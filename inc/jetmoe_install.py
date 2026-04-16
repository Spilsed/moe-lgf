from huggingface_hub import snapshot_download

# This downloads the model to a folder named 'jetmoe-local'
snapshot_download(repo_id="jetmoe/jetmoe-8b-chat", local_dir="jetmoe-local-chat")
snapshot_download(repo_id="jetmoe/jetmoe-8b",local_dir="jetmoe-local")