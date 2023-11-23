from PIL import Image
import gradio as gr
from io import BytesIO
from diffusers import DiffusionPipeline
import torch
from datetime import datetime
import os 

DATASETS_DIR = "datasets/"


title = """SDXL Lora DreamBooth"""

description = """#### Generate images of your own pet."""
lora_path = "checkpoints/minsuck_checkpoints/checkpoint-1000"

# pipe = DiffusionPipeline.from_pretrained(
#         "stabilityai/stable-diffusion-xl-base-1.0",
#         torch_dtype=torch.float16,
#         )
# pipe.to("cuda")
# pipe.load_lora_weights(lora_path)



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
    new_size = (512, 512)
    resized_image = cropped_image.resize(new_size)

    # Save the cropped and resized image
    resized_image.save(save_path)
    return save_path


def empty_cache():
    del pipe
    torch.cuda.empty_cache() # PyTorch thing
    print("Closing Gradio")


def get_image(prompt, negative_prompt, inference_steps):
    payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": 512,
            "height": 512,
            #    "samples": "1",
            "num_inference_steps": inference_steps,
            "guidance_scale": 7.5
            }
    image = pipe(**payload).images[0]
    image.save(f"outputs/minsuck_{datetime.now().timestamp()}.png")
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


def train(pet_name):
    pass

if __name__ == "__main__":
    demo = gr.Blocks()

    with demo:
        gr.Markdown("""# SDXL LoRA DreamBooth""")
        with gr.Tabs():
            with gr.TabItem("Generation"):
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

                pet_name_input = gr.Textbox(
                        value=str(datetime.now().timestamp),
                        label="Pet Name"
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
                train_btn = gr.Button("Train")

        prepare_dataset_btn.click(
                create_dataset,
                inputs=[
                    images,
                    pet_name_input,
                    ],
                outputs=[
                    dataset_images,
                    ],
                )
        generate_image_btn.click(
                get_image,
                inputs=[
                    prompt_input,
                    negative_prompt_input,
                    num_inference_steps_input,
                    ],
                outputs=[
                    image_output,
                    ]
                )


    demo.launch(debug=True, share=True)


# try:
#     demo = gr.Interface(fn=get_image,
#                         inputs = [
#                             gr.Textbox(label="Enter the Prompt", value="A photo of zwc cat"), 
#                             gr.Textbox(label="Negative Prompt"),
#                             gr.Number(label="Enter number of steps", value=30),
#                             ],
#                         outputs = gr.Image(type='pil'),
#                         title = title,
#                         description = description)
# 
#     demo.launch(debug='True', share=True)
# except Exception as e:
#     del pipe
#     torch.cuda.empty_cache() # PyTorch thing
#     print("Closing Gradio")
