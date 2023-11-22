from diffusers import DiffusionPipeline
import torch


lora_path = "checkpoints/minsuck_checkpoints/checkpoint-1000"
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)

pipe.to("cuda")
pipe.load_lora_weights(lora_path)

prompt = "A photo of zwc cat wearing santa clothes, cristmas background, cristmas tree, snowing"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("outputs/minsuck_35234.png")



