from PIL import Image
import gradio as gr
from diffusers import DiffusionPipeline
import torch
from datetime import datetime
import os 
import argparse
from train_dreambooth_lora_sdxl import parse_args as parse_train_args, main as train_dreambooth_lora



OUTPUT_ROOT = "checkpoints"
DATASETS_DIR = "datasets"
MODEL_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
RESOLUTION = 512
MAX_TRAIN_STEP = 1000
CHECKPOINTING_STEPS = 200
LORA_ROOT_PATH = "checkpoints"

title = """SDXL Lora DreamBooth"""

description = """#### Generate images of your own pet."""
pipe: DiffusionPipeline = None

lora_paths = []

def reload_lora_paths(exclude_personal_models=True):
    global lora_paths
    lora_paths = []
    for lora_set_directory in os.listdir(LORA_ROOT_PATH):
        if "bn" in str(lora_set_directory) and exclude_personal_models:
            continue
        for lora_directory in os.listdir(os.path.join(LORA_ROOT_PATH, lora_set_directory)):
            lora_paths.append(str(os.path.join(LORA_ROOT_PATH, lora_set_directory, lora_directory)))
    print(f"LoRA Paths = {lora_paths}")

def load_model(lora_path):
    global pipe
    pipe = DiffusionPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            )
    pipe.to("cuda")
    pipe.load_lora_weights(lora_path)


def crop_image(image_path, save_path):
    # Open the image
    image = Image.open(image_path)

    # Crop the image (example cropping to a square in the center)
    width, height = image.size
    min_dimension = min(width, height)
    left = (width - min_dimension) / 2
    top = (height - min_dimension) / 2
    right = (width + min_dimension) / 2
    bottom = (height + min_dimension) / 2
    cropped_image = image.crop((left, top, right, bottom))

    # Resize the cropped image to 512x512 pixels
    new_size = (RESOLUTION, RESOLUTION)
    resized_image = cropped_image.resize(new_size)

    # Save the cropped and resized image
    resized_image.save(save_path)
    return save_path



def get_image(lora_path, prompt, negative_prompt, inference_steps):
    payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": RESOLUTION,
            "height": RESOLUTION,
            #    "samples": "1",
            "num_inference_steps": inference_steps,
            "guidance_scale": 7.5
            }
    image = pipe(**payload).images[0]
    image.save(os.path.join(lora_path, f"{datetime.now().timestamp()}.png"))
    return image

def preview(files, sd: gr.SelectData):
    return files[sd.index].name

def create_dataset(files, pet_name):
    resized_images = []
    dataset_dir = os.path.join(DATASETS_DIR, pet_name)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    for i, file in enumerate(files):
        image_path = file.name
        save_image_path = os.path.join(dataset_dir, str(i) + ".png")
        crop_image(image_path, save_image_path)
        resized_images.append(save_image_path)

    return resized_images


def empty_gpu_memory():
    global pipe
    if not pipe:
        return
    del pipe
    torch.cuda.empty_cache()


def train(pet_name, class_name, token, progress=gr.Progress()):
    empty_gpu_memory()

    instance_prompt = f"A photo of {token} {class_name}"
    output_dir = os.path.join(OUTPUT_ROOT, pet_name)
    instance_data_dir = os.path.join(DATASETS_DIR, pet_name)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    train_input_args = [
        f"--pretrained_model_name_or_path={MODEL_PATH}",
        f"--instance_data_dir={instance_data_dir}",
        f"--output_dir={output_dir}",
        "--mixed_precision=fp16",
        f"--instance_prompt='{instance_prompt}'",
        f"--class_prompt='{class_name}'",
        # f"--validation_prompt='{instance_prompt}'",
        # "--num_validation_images=4",
        # "--validation_epochs=50",
        "--center_crop",
        "--resume_from_checkpoint=latest",
        f"--resolution={RESOLUTION}",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=4",
        "--learning_rate=1e-4",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        f"--checkpointing_steps={CHECKPOINTING_STEPS}",
        f"--max_train_steps={MAX_TRAIN_STEP}",
        "--seed=0",
        "--checkpoints_total_limit=10",
    ]

    print("Input args:", train_input_args)

    train_args = parse_train_args(
        input_args=train_input_args
    )
    train_dreambooth_lora(train_args)

    # return os.system(train_command)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    return args 


def main(args):
    reload_lora_paths(
        exclude_personal_models=args.share
    )
   
    demo = gr.Blocks()

    with demo:
        gr.Markdown("""# SDXL LoRA DreamBooth""")
        with gr.Tabs():
            with gr.TabItem("Generation"):
                with gr.Row(equal_height=False):
                    lora_path_dropdown = gr.Dropdown(
                        choices=lora_paths,
                        label="LoRA Path",
                        scale=10,
                    )
                    lora_path_refresh_btn = gr.Button(
                        value="",
                        size="sm",
                        scale=1,
                        icon="assets/refresh_icon.png"
                    )
                prompt_input = gr.Textbox(
                        label="Positive Prompt",
                        value="A photo of zwc cat"
                        )
                negative_prompt_input = gr.Textbox(
                        label="Negative Prompt",
                        ) 
                with gr.Row():
                    with gr.Column():
                        num_inference_steps_input = gr.Number(
                                label="Num Inference Steps",
                                value=30,
                                )
                    with gr.Column():
                        generate_image_btn = gr.Button("Generate")
                
                image_output = gr.Image(type="pil")

            with gr.TabItem("Train"):
                with gr.Row():
                    images = gr.File(file_types=["image"], file_count="multiple")
                    preview_images = gr.Image()
                
                with gr.Row():
                    instance_name_input = gr.Textbox(
                        value="",
                        label="Instance Name",
                        )
                    class_name_input = gr.Textbox(
                        value="cat",
                        label="Class Name"
                    )
                    token_input = gr.Textbox(
                        value="zwc",
                        label="Token Prompt",
                    )
                images.select(preview, images, preview_images)
                prepare_dataset_btn = gr.Button("Prepare Dataset")
                dataset_images = gr.Gallery(
                        label="Train Images",
                        columns=[3],
                        rows=[2],
                        object_fit="contain",
                        height="auto",
                        )
                with gr.Row():
                    train_btn = gr.Button("Train")
                    train_log = gr.Textbox(
                        label="Train Log",
                        max_lines=20,
                    )

        prepare_dataset_btn.click(
            create_dataset,
            inputs=[
                images,
                instance_name_input,
                ],
            outputs=[
                dataset_images,
                ],
            )
        generate_image_btn.click(
            get_image,
            inputs=[
                lora_path_dropdown,
                prompt_input,
                negative_prompt_input,
                num_inference_steps_input,
                ],
            outputs=[
                image_output,
                ]
            )
        train_btn.click(
            train,
            inputs=[
                instance_name_input,
                class_name_input,
                token_input,
            ],
            outputs=[
                train_log,
            ]
        ) 
        lora_path_dropdown.select(
            load_model,
            inputs=[lora_path_dropdown]
        )
        lora_path_refresh_btn.click(reload_lora_paths)


    demo.launch(debug=args.debug, share=args.share)

 

if __name__ == "__main__":
    args = parse_args()
    main(args)