# Variables
ENV_NAME = talent
ENV_FILE = environment.yml

# Targets
.PHONY: all env setup clean

# Set up everything
all: env setup

# Create or update Conda environment
env:
	conda env create -f $(ENV_FILE) || conda env update -f $(ENV_FILE)

# Run the setup script with pip (avoiding deprecated `setup.py install`)
setup:
	conda run -n $(ENV_NAME) pip install .

# Clean up environment and GloVe files
clean:
	conda remove --name $(ENV_NAME) --all
	rm -rf glove.6B.zip glove.6B.*
