from huggingface_hub import snapshot_download

# This downloads the model to a folder named 'jetmoe-hf'
snapshot_download(repo_id="jetmoe/jetmoe-8b-chat", local_dir="jetmoe-local")