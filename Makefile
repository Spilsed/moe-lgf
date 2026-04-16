

local-model:
	cd inc/ && python jetmoe_install.py
	cd inc/ && mv jetmoe-local-chat/*.safetensors jetmoe-local/ && mv jetmoe-local-chat/*token* jetmoe-local/
	rm -r inc/jetmoe-local-chat

clean:
	rm -rf inc/jetmoe-local
