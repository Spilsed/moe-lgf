# Run all `make` commands in the base directory

# To build the untrained chat model, run `$ make local-model`
# To remove the model(to clear space), run `$ make clean`
# To run a chat session, run `$ make test`
# To run default benchmarking, run `$ make bench`

.ONESHELL:
ACTIVATE_VENV := source .venv/bin/activate

local-model:
	uv sync && $(ACTIVATE_VENV)
	cd inc/ && python jetmoe_install.py
	cd inc/ && mv jetmoe-local-chat/*.safetensors jetmoe-local/ && mv jetmoe-local-chat/*token* jetmoe-local/
	rm -r inc/jetmoe-local-chat

test:
	uv sync && $(ACTIVATE_VENV)
	accelerate launch src/benchmarking/test.py

bench:
	uv sync && $(ACTIVATE_VENV)
	accelerate launch src/benchmarking/benchmarking.py

clean:
	rm -rf inc/jetmoe-local
