from threading import Thread
from io import BytesIO
from typing import Optional

import ray
from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor
from transformers.generation.streamers import TextIteratorStreamer
from PIL import Image
import requests

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

# Set PyTorch configs outside the class to avoid serialization issues
torch.backends.cudnn.benchmark = True

weights = "./weights"

app = FastAPI()

class ImageDescriptionRequest(BaseModel):
    image: str  # Public URL of the image
    prompt: str  # Prompt for the model
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 128

@serve.deployment(
    ray_actor_options={"num_gpus": 1, "num_cpus": 8},
    num_replicas=8  # Create 8 replicas to utilize all GPUs on g6e.24xlarge
)
@serve.ingress(app)
class ImageDescriptionService:
    def __init__(self):
        disable_torch_init()
        print(f"Loading custom LLaVA lora model: {weights}...")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            weights,
            model_name="llava-v1.5-13b-custom-lora",
            model_base="liuhaotian/llava-v1.5-13b",
            load_8bit=False,
            load_4bit=False
        )

        # Move model to GPU and optimize for inference
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def load_image(self, image_file):
        if image_file.startswith('http') or image_file.startswith('https'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image

    @app.post("/describe")
    async def describe(self, body: ImageDescriptionRequest):
        try:
            conv_mode = "llava_v1"
            conv = conv_templates[conv_mode].copy()

            # Add user prompt with image token
            image_token_str = DEFAULT_IMAGE_TOKEN
            prompt = body.prompt
            if image_token_str not in prompt:
                prompt = f"{image_token_str}\n{prompt}"
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)

            # Process image
            image = self.load_image(body.image)
            image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()

            # Create prompt with image token
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()

            # Set up stopping criteria
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

            # Use streamer for efficient token generation
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            # Run generation in a separate thread
            with torch.no_grad():
                generation_kwargs = dict(
                    images=image_tensor,
                    input_ids=input_ids,
                    streamer=streamer,
                    stopping_criteria=[stopping_criteria],
                    max_new_tokens=body.max_tokens,
                    temperature=body.temperature,
                    top_p=body.top_p,
                    use_cache=True,
                )

                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                # Collect generated text
                generated_text = ""
                for text in streamer:
                    generated_text += text

                thread.join()

                # Clean up the generated text by removing the stop string if present
                if generated_text.endswith(stop_str):
                    generated_text = generated_text[:-len(stop_str)]

                return {"generated_text": generated_text.strip()}

        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}

deployment = ImageDescriptionService.bind()