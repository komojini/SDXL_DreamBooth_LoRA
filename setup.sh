
ENV_DIR="venv"  # name of your virtual environment directory

# Check if the environment directory exists
if [ -d "$ENV_DIR" ]; then
    echo "Activating virtual environment..."
    source "$ENV_DIR/bin/activate"
else
    echo "Virtual environment not found. Creating one..."
    virtualenv $ENV_DIR
    source "$ENV_DIR/bin/activate"
fi

pip install git+https://github.com/huggingface/diffusers

pip install torchvision invisible_watermark transformers accelerate safetensors gradio xformers
accelerate config

