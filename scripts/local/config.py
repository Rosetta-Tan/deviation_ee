config = {
    "hqi401b": {
        "exec": "docker run --runtime=nvidia --gpus all -v /home/yitan/Coding/deviation_ee:/home/dnm gdmeyer/dynamite:latest-cuda python",
        "data_dir": "/home/yitan/Coding/deviation_ee/data"
    }
}