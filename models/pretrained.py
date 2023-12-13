from PIL import Image
from .model import BaseModel
from transformers import pipeline

from loguru import logger


class PretrainedModel(BaseModel):
    def __init__(self):
        super().__init__("pretrained")
        self.model = pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base",
            max_new_tokens=20,
        )
        self.warmup()

    def warmup(self) -> None:
        logger.info(f"Warming up {self.model_name} model...")
        for img in self.get_warmup_imgs():
            _ = self.model(img)
        logger.info(f"Finished {self.model_name} model warmup!")

    def inference(self, img: Image.Image) -> str:
        caption = self.model(img)[0]["generated_text"]
        return caption
