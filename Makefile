# To build the untrained chat model, run `$make local-model` in the base directory
# To remove the model(to clear space), run `$make clean` in the base directory

local-model:
	cd inc/ && python jetmoe_install.py
	cd inc/ && mv jetmoe-local-chat/*.safetensors jetmoe-local/ && mv jetmoe-local-chat/*token* jetmoe-local/
	rm -r inc/jetmoe-local-chat

clean:
	rm -rf inc/jetmoe-local
