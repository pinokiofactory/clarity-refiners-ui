module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    method: "shell.run",
    params: {
      path: "app",
      message: "git pull"
    }
  }, {
     method: "shell.run",
     params: {
       venv: "env",
      message: "pip install -r requirements.txt",
      path: "app"
    }
  }]
}