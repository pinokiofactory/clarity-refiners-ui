import os
import sys
import gradio as gr
import pillow_heif
import torch
import devicetorch
import subprocess
import gc
import psutil  # for system stats - gpu/cpu etc

from PIL import Image
from pathlib import Path
from datetime import datetime
from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoProcessor

from refiners.fluxion.utils import manual_seed
from refiners.foundationals.latent_diffusion import Solver, solvers
from enhancer import ESRGANUpscaler, ESRGANUpscalerCheckpoints
from system_monitor import SystemMonitor
from message_manager import MessageManager

import warnings
# Filter out the timm deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
# Filter the GenerationMixin inheritance warning
warnings.filterwarnings("ignore", message=".*has generative capabilities.*")
# Filter the PyTorch flash attention warning
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

pillow_heif.register_heif_opener()
pillow_heif.register_avif_opener()

message_manager = MessageManager()
save_path = "../outputs"   # Can be changed to a preferred directory: "C:\path\to\save_folder"


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


def generate_prompt(image: Image.Image, caption_detail: str = "<CAPTION>") -> str:
    """
    Generate a detailed caption for the image using Florence-2.
    """
    if image is None:
        message_manager.add_warning("No image loaded for captioning")
        return gr.Warning("Please load an image first!")
        
    try:
        message_manager.add_message(f"Starting Florence-2 caption generation with detail level: {caption_detail}")
        device = torch.device(devicetorch.get(torch))
        torch_dtype = devicetorch.dtype(torch)
        
        gc.collect()
        devicetorch.empty_cache(torch)
        message_manager.add_message("Loading Florence-2 model...")

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
        message_manager.add_success("Florence-2 model loaded successfully")

        # Move model to device after eval mode
        model = devicetorch.to(torch, model)
        message_manager.add_message("Processing image with Florence-2...")

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
        raw_caption = parsed_answer[caption_detail]
        caption_text = clean_caption(raw_caption)
        enhanced_prompt = f"masterpiece, best quality, highres, {caption_text}"
        
        message_manager.add_message("Raw caption: " + raw_caption)
        message_manager.add_success(f"Generated prompt: {enhanced_prompt}")

        # Aggressive cleanup
        del generated_ids
        del inputs
        model.cpu()
        del model
        del processor
        gc.collect()
        devicetorch.empty_cache(torch)
        message_manager.add_message("Cleaned up Florence-2 resources")
            
        return enhanced_prompt
        
    except Exception as e:
        # Ensure cleanup happens even on error
        gc.collect()
        devicetorch.empty_cache(torch)
        message_manager.add_error(f"Error in caption generation: {str(e)}")
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
        message_manager.add_message(f"Starting enhancement with seed {seed}")
        message_manager.add_message(f"Upscale factor: {upscale_factor}x")
        
        # Clear memory before processing
        gc.collect()
        devicetorch.empty_cache(torch)
        message_manager.add_message("Cleared GPU memory")
        
        manual_seed(seed)
        solver_type: type[Solver] = getattr(solvers, solver)

        # Use no_grad context
        with torch.no_grad():
            message_manager.add_message("Processing image...")
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

        global latest_result
        latest_result = enhanced_image
        message_manager.add_success("Enhancement complete!")
        
        if auto_save_enabled:
            save_output(enhanced_image, True)
        
        # Clear memory after processing
        gc.collect()
        devicetorch.empty_cache(torch)
        message_manager.add_message("Cleaned up resources")
        
        return (input_image, enhanced_image)
        
    except Exception as e:
        message_manager.add_error(f"Error during processing: {str(e)}")
        gc.collect()
        devicetorch.empty_cache(torch)
        return gr.Warning(f"Error during processing: {str(e)}")
        

def open_output_folder() -> None:
    folder_path = os.path.abspath(save_path)
    
    try:
        os.makedirs(folder_path, exist_ok=True)
    except Exception as e:
        message_manager.add_error(f"Error creating folder: {str(e)}")
        return
        
    try:
        if os.name == 'nt':  # Windows
            os.startfile(folder_path)
        elif os.name == 'posix':  # macOS and Linux
            subprocess.run(['xdg-open' if os.name == 'posix' else 'open', folder_path])
        message_manager.add_success(f"Opened outputs folder: {folder_path}")
    except Exception as e:
        message_manager.add_error(f"Error opening folder: {str(e)}")


def save_output(image: Image.Image = None, auto_saved: bool = False) -> None:  # Changed return type to None
    if image is None:
        if not globals().get('latest_result'):
            message_manager.add_warning("No image to save! Please enhance an image first.")
            return
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
        
        # Add to message manager
        message_manager.add_success(message)
        
    except Exception as e:
        error_msg = f"Error saving image: {str(e)}"
        message_manager.add_error(error_msg)


        
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

.console-scroll textarea {
    max-height: 12em !important;  /* Approximately 8 lines of text */
    overflow-y: auto !important;  /* Enables vertical scrolling */
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
                auto_save = gr.Checkbox(label="Auto-save", value=True)
                open_folder_button = gr.Button("Open Outputs Folder", scale=2)

    with gr.Accordion("Prompting", open=False):
        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="masterpiece, best quality, highres",
                show_label=True
            )
            with gr.Column(scale=1):
                caption_detail = gr.Radio(
                    choices=["<CAPTION>","<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"],
                    value="<CAPTION>",
                    label="Florence-2 Caption Detail",
                    info="Choose level of detail for image analysis"
                )
                generate_prompt_btn = gr.Button("📝 Generate Prompt", variant="primary")
        with gr.Row():
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="worst quality, low quality, normal quality",
            )
            seed = gr.Slider( minimum=1, maximum=10_000, value=42, step=1, label="Seed")
        
    with gr.Accordion("Options", open=False):
        with gr.Row():    
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
                value=0.25,
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
    with gr.Accordion("Advanced Options", open=False):
        with gr.Row(): 
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
        with gr.Row(): 
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
            
    with gr.Row():       
        # Status Info (for cpu/gpu monitor)
        resource_monitor = gr.Textbox(
            label="Monitor",
            lines=8,
            interactive=False,
            # value=get_welcome_message()
        )  
        console_out = gr.Textbox(
            label="Console",
            lines=8,
            interactive=False,
            show_copy_button=True,
            autoscroll=True,    # Enables automatic scrolling to newest messages
            elem_classes="console-scroll"  # Add custom class for styling
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
    
    def update_console():
        return message_manager.get_messages()
    
    # Initialize the timer and set up its tick event
    demo.load(
        fn=lambda: (SystemMonitor.get_system_info(), update_console()),
        outputs=[resource_monitor, console_out],
        every=1  # Updates every 1 second
    )
    
    
demo.launch(share=False)
