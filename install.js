module.exports = {
  run: [
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
        }
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env", 
        message: [
          "pip install gradio devicetorch",
          "pip install -r requirements.txt"
        ]
      }
    }
  ]
}
