import os
import gradio as gr
import pillow_heif
import torch
import devicetorch
import subprocess
import gc
import logging

from PIL import Image
from pathlib import Path
from datetime import datetime
from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoProcessor

from refiners.fluxion.utils import manual_seed
from refiners.foundationals.latent_diffusion import Solver, solvers

from enhancer import ESRGANUpscaler, ESRGANUpscalerCheckpoints

import warnings
# Filter out the timm deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
# Filter the GenerationMixin inheritance warning
warnings.filterwarnings("ignore", message=".*has generative capabilities.*")
# Filter the PyTorch flash attention warning
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")



pillow_heif.register_heif_opener()
pillow_heif.register_avif_opener()

device = devicetorch.get(torch)

save_path = "outputs"  # Can be changed to a preferred directory: "C:\path\to\save_folder"


CHECKPOINTS = ESRGANUpscalerCheckpoints(
    unet=Path(
        hf_hub_download(
            repo_id="refiners/juggernaut.reborn.sd1_5.unet",
            filename="model.safetensors",
            revision="347d14c3c782c4959cc4d1bb1e336d19f7dda4d2",
        )
    ),
    clip_text_encoder=Path(
        hf_hub_download(
            repo_id="refiners/juggernaut.reborn.sd1_5.text_encoder",
            filename="model.safetensors",
            revision="744ad6a5c0437ec02ad826df9f6ede102bb27481",
        )
    ),
    lda=Path(
        hf_hub_download(
            repo_id="refiners/juggernaut.reborn.sd1_5.autoencoder",
            filename="model.safetensors",
            revision="3c1aae3fc3e03e4a2b7e0fa42b62ebb64f1a4c19",
        )
    ),
    controlnet_tile=Path(
        hf_hub_download(
            repo_id="refiners/controlnet.sd1_5.tile",
            filename="model.safetensors",
            revision="48ced6ff8bfa873a8976fa467c3629a240643387",
        )
    ),
    esrgan=Path(
        hf_hub_download(
            repo_id="philz1337x/upscaler",
            filename="4x-UltraSharp.pth",
            revision="011deacac8270114eb7d2eeff4fe6fa9a837be70",
        )
    ),
    negative_embedding=Path(
        hf_hub_download(
            repo_id="philz1337x/embeddings",
            filename="JuggernautNegative-neg.pt",
            revision="203caa7e9cc2bc225031a4021f6ab1ded283454a",
        )
    ),
    negative_embedding_key="string_to_param.*",
    
    loras={
        "more_details": Path(
            hf_hub_download(
                repo_id="philz1337x/loras",
                filename="more_details.safetensors",
                revision="a3802c0280c0d00c2ab18d37454a8744c44e474e",
            )
        ),
        "sdxl_render": Path(
            hf_hub_download(
                repo_id="philz1337x/loras",
                filename="SDXLrender_v2.0.safetensors",
                revision="a3802c0280c0d00c2ab18d37454a8744c44e474e",
            )
        ),
    },
)

device = torch.device(devicetorch.get(torch))
dtype = devicetorch.dtype(torch) 
enhancer = ESRGANUpscaler(checkpoints=CHECKPOINTS, device=device, dtype=dtype)


def generate_prompt(image: Image.Image, caption_detail: str = "<DETAILED_CAPTION>") -> str:
    """
    Generate a detailed caption for the image using Florence-2, with optimized memory usage.
    """
    if image is None:
        return gr.Warning("Please load an image first!")
        
    try:
        device = torch.device(devicetorch.get(torch))
        torch_dtype = devicetorch.dtype(torch)
        
        gc.collect()
        devicetorch.empty_cache(torch)

        # Load model in eval mode immediately
        model = AutoModelForCausalLM.from_pretrained(
            "multimodalart/Florence-2-large-no-flash-attn", 
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).eval()
        
        processor = AutoProcessor.from_pretrained(
            "multimodalart/Florence-2-large-no-flash-attn",
            trust_remote_code=True
        )

        # Move model to device after eval mode
        model = devicetorch.to(torch, model)

        # Process the image
        inputs = processor(
            text=caption_detail, 
            images=image.convert("RGB"), 
            return_tensors="pt"
        )
        
        # Convert inputs to the correct dtype and move to device
        inputs = {
            k: v.to(device=device, dtype=torch_dtype if v.dtype == torch.float32 else v.dtype) 
            for k, v in inputs.items()
        }

        # Generate caption with no grad
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=2
            )
            
            # Move generated_ids to CPU immediately
            generated_ids = generated_ids.cpu()

        # Clear inputs from GPU
        inputs = {k: v.cpu() for k, v in inputs.items()}
        devicetorch.empty_cache(torch)
        
        # Process the generated text
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=caption_detail,
            image_size=(image.width, image.height)
        )
        
        # Clean up the caption and add enhancement-specific terms
        caption_text = clean_caption(parsed_answer[caption_detail])
        enhanced_prompt = f"masterpiece, best quality, highres, {caption_text}"

        # Aggressive cleanup
        del generated_ids
        del inputs
        model.cpu()
        del model
        del processor
        gc.collect()
        devicetorch.empty_cache(torch)
            
        return enhanced_prompt
        
    except Exception as e:
        # Ensure cleanup happens even on error
        gc.collect()
        devicetorch.empty_cache(torch)
        return gr.Warning(f"Error generating prompt: {str(e)}")
        
        
def clean_caption(text: str) -> str:
    """
    Clean up the caption text by removing common prefixes, filler phrases, and dangling descriptions.
    """
    # Common prefixes to remove
    replacements = [
        "The image shows ",
        "The image is ",
        "The image depicts ",
        "This image shows ",
        "This image depicts ",
        "The photo shows ",
        "The photo depicts ",
        "The picture shows ",
        "The picture depicts ",
        "The overall mood ",
        "The mood of the image ",
        "There is ",
        "We can see ",
    ]
    
    cleaned_text = text
    for phrase in replacements:
        cleaned_text = cleaned_text.replace(phrase, "")
    
    # Remove mood/atmosphere fragments
    mood_patterns = [
        ". The mood is ",
        ". The atmosphere is ",
        ". of the image is ",
        ". The overall feel is ",
        ". The tone is ",
    ]
    
    for pattern in mood_patterns:
        if pattern in cleaned_text:
            cleaned_text = cleaned_text.split(pattern)[0]
    
    # Remove trailing fragments
    while cleaned_text.endswith((" is", " are", " and", " with", " the")):
        cleaned_text = cleaned_text.rsplit(" ", 1)[0]
    
    return cleaned_text.strip()

    
def process(
    input_image: Image.Image,
    prompt: str = "masterpiece, best quality, highres",
    negative_prompt: str = "worst quality, low quality, normal quality",
    seed: int = 42,
    upscale_factor: int = 2,
    controlnet_scale: float = 0.6,
    controlnet_decay: float = 1.0,
    condition_scale: int = 6,
    tile_width: int = 112,
    tile_height: int = 144,
    denoise_strength: float = 0.35,
    num_inference_steps: int = 18,
    solver: str = "DDIM",
    auto_save_enabled: bool = True,  
) -> tuple[Image.Image, Image.Image]:
    try:
        # Clear memory before processing
        gc.collect()
        devicetorch.empty_cache(torch)
        
        manual_seed(seed)
        solver_type: type[Solver] = getattr(solvers, solver)

        # Use no_grad context
        with torch.no_grad():
            enhanced_image = enhancer.upscale(
                image=input_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                upscale_factor=upscale_factor,
                controlnet_scale=controlnet_scale,
                controlnet_scale_decay=controlnet_decay,
                condition_scale=condition_scale,
                tile_size=(tile_height, tile_width),
                denoise_strength=denoise_strength,
                num_inference_steps=num_inference_steps,
                loras_scale={"more_details": 0.5, "sdxl_render": 1.0},
                solver_type=solver_type,
            )

        # Store the latest result for auto-save functionality
        global latest_result
        latest_result = enhanced_image
        
        if auto_save_enabled:
            save_output(enhanced_image, True)
        
        # Clear memory after processing
        gc.collect()
        devicetorch.empty_cache(torch)
        
        return (input_image, enhanced_image)
        
    except Exception as e:
        gc.collect()
        devicetorch.empty_cache(torch)
        return gr.Warning(f"Error during processing: {str(e)}")
        

def open_output_folder():
    folder_path = os.path.abspath(save_path)
    
    try:
        os.makedirs(folder_path, exist_ok=True)
    except Exception as e:
        return gr.Warning(f"Error creating folder: {str(e)}")
        
    try:
        if os.name == 'nt':  # Windows
            os.startfile(folder_path)
        elif os.name == 'posix':  # macOS and Linux
            subprocess.run(['xdg-open' if os.name == 'posix' else 'open', folder_path])
        return gr.Info(f"Opened folder: {folder_path}")
    except Exception as e:
        return gr.Warning(f"Error opening folder: {str(e)}")

def save_output(image: Image.Image = None, auto_saved: bool = False) -> str:
    """
    Save the enhanced image to the output directory.
    
    Args:
        image: The image to save. If None, uses the latest_result
        auto_saved: Whether this is an automatic save
    
    Returns:
        str: Success/error message for the UI
    """
    if image is None:
        if not globals().get('latest_result'):
            return gr.Warning("No image to save! Please enhance an image first.")
        image = latest_result
        
    try:
        # Create output directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"enhanced_{timestamp}.png"
        filepath = os.path.join(save_path, filename)
        
        # Save the image
        image.save(filepath, "PNG")
        
        save_type = "auto-saved" if auto_saved else "saved"
        message = f"Image {save_type} as: {filename}"
        
        # Return different types of notifications based on save type
        if auto_saved:
            print(message)  # Log auto-saves to console
            return None
        else:
            return gr.Info(message)
            
    except Exception as e:
        error_msg = f"Error saving image: {str(e)}"
        if auto_saved:
            print(f"Auto-save failed: {error_msg}")
            return None
        else:
            return gr.Warning(error_msg)

        
css = """

/* Specific adjustments for Image */
.image-container .image-custom {
    max-width: 100% !important;
    max-height: 80vh !important;
    width: auto !important;
    height: auto !important;
}

/* Center the ImageSlider container and maintain full width for slider */
.image-container .image-slider-custom {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
}

/* Style for the slider container */
.image-container .image-slider-custom > div {
    width: 100% !important;
    max-width: 100% !important;
    max-height: 80vh !important;
}

/* Ensure both before/after images maintain aspect ratio */
.image-container .image-slider-custom img {
    max-height: 80vh !important;
    width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
}

/* Style for the slider handle */
.image-container .image-slider-custom .image-slider-handle {
    width: 2px !important;
    background: white !important;
    border: 2px solid rgba(0, 0, 0, 0.6) !important;
}

"""

# Store the latest processing result
latest_result = None

with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Column(elem_classes="image-container"):
            input_image = gr.Image(type="pil", label="Input Image", elem_classes=["image-custom"])
            with gr.Row():
                run_button = gr.ClearButton(components=None, value="Enhance Image", variant="primary")
           
        with gr.Column(elem_classes="image-container"):
            output_slider = ImageSlider(
                interactive=False,
                label="Before / After",
                elem_classes=["image-slider-custom"]
            )
            run_button.add(output_slider)
            with gr.Row():
                save_result = gr.Button("Save Result", scale=2)
                auto_save = gr.Checkbox(
                    label="Auto-save", 
                    value=True,
                    info="Automatically save images after processing"
                )
                open_folder_button = gr.Button("Open Outputs Folder", scale=2)

    with gr.Accordion("Advanced Options", open=False):
        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="masterpiece, best quality, highres",
                show_label=True
            )
            with gr.Column(scale=1):
                caption_detail = gr.Radio(
                    choices=["<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"],
                    value="<DETAILED_CAPTION>",
                    label="Florence-2 Caption Detail",
                    info="Choose level of detail for image analysis"
                )
                generate_prompt_btn = gr.Button("📝 Generate Prompt")
        
        negative_prompt = gr.Textbox(
            label="Negative Prompt",
            placeholder="worst quality, low quality, normal quality",
        )
        upscale_factor = gr.Slider(
            minimum=1,
            maximum=4,
            value=2,
            step=0.2,
            label="Upscale Factor",
        )
        denoise_strength = gr.Slider(
            minimum=0,
            maximum=1,
            value=0.35,
            step=0.05,
            label="Denoise Strength",
        )
        num_inference_steps = gr.Slider(
            minimum=1,
            maximum=30,
            value=18,
            step=1,
            label="Number of Inference Steps",
        )
        seed = gr.Slider(
            minimum=0,
            maximum=10_000,
            value=42,
            step=1,
            label="Seed",
        )
        controlnet_scale = gr.Slider(
            minimum=0,
            maximum=1.5,
            value=0.6,
            step=0.1,
            label="ControlNet Scale",
        )
        controlnet_decay = gr.Slider(
            minimum=0.5,
            maximum=1,
            value=1.0,
            step=0.025,
            label="ControlNet Scale Decay",
        )
        condition_scale = gr.Slider(
            minimum=2,
            maximum=20,
            value=6,
            step=1,
            label="Condition Scale",
        )
        tile_width = gr.Slider(
            minimum=64,
            maximum=200,
            value=112,
            step=1,
            label="Latent Tile Width",
        )
        tile_height = gr.Slider(
            minimum=64,
            maximum=200,
            value=144,
            step=1,
            label="Latent Tile Height",
        )
        solver = gr.Radio(
            choices=["DDIM", "DPMSolver"],
            value="DDIM",
            label="Solver",
        )

    # Event handlers
    
    generate_prompt_btn.click(
        fn=generate_prompt,
        inputs=[input_image, caption_detail],
        outputs=[prompt]
    )
    
    run_button.click(
        fn=process,
        inputs=[
            input_image,
            prompt,
            negative_prompt,
            seed,
            upscale_factor,
            controlnet_scale,
            controlnet_decay,
            condition_scale,
            tile_width,
            tile_height,
            denoise_strength,
            num_inference_steps,
            solver,
            auto_save, 
        ],
        outputs=output_slider,
    )
    
    save_result.click(
        fn=save_output,
        inputs=None,
        outputs=gr.Text(visible=False)
    )
    
    open_folder_button.click(
        fn=open_output_folder,
        inputs=None,
        outputs=gr.Text(visible=False) 
    )

demo.launch(share=False)
