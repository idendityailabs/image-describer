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


weights = "./weights"


app = FastAPI()

class ImageDescriptionRequest(BaseModel):
    image: str  # Public URL of the image
    prompt: str  # Prompt for the model
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 128

@serve.deployment(
    ray_actor_options={"num_gpus": 1, "num_cpus": 10},
    num_replicas=8
)
@serve.ingress(app)
class ImageDescriptionService:
    def __init__(self):
        disable_torch_init()
        print(f"Loading custom LLaVA lora model: {weights}...")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(weights, model_name="llava-v1.5-13b-custom-lora", model_base="liuhaotian/llava-v1.5-13b", load_8bit=False, load_4bit=False)

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

            image_data = self.load_image(str(body.image))
            image_tensor = self.image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].half().cuda()

            inp = DEFAULT_IMAGE_TOKEN + '\n' + body.prompt
            conv.append_message(conv.roles[0], inp)

            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=20.0)

            with torch.inference_mode():
                thread = Thread(target=self.model.generate, kwargs=dict(
                    inputs=input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=body.temperature,
                    top_p=body.top_p,
                    max_new_tokens=body.max_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]))
                thread.start()
                prepend_space = False

                result = []

                for new_text in streamer:
                    if new_text == " ":
                        prepend_space = True
                        continue
                    if new_text.endswith(stop_str):
                        new_text = new_text[:-len(stop_str)].strip()
                        prepend_space = False
                    elif prepend_space:
                        new_text = " " + new_text
                        prepend_space = False
                    if len(new_text):
                        result.append(new_text)
                if prepend_space:
                    result.append(" ")
                thread.join()

            result = "".join(result)
            return {"description": result}
        except Exception as e:
            return {"error": str(e)}

deployment = ImageDescriptionService.bind()
