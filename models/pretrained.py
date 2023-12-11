from PIL.Image import Image
from .model import Model
from transformers import pipeline


class PretrainedModel(Model):
    def __init__(self):
        super().__init__("pretrained")
        self.model = pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base",
            max_new_tokens=20,
        )
        self.warmup()

    def warmup(self) -> None:
        print(f"Warming up {self.model_name} model...")
        for img in self.get_warmup_imgs():
            _ = self.model(img)
        print(f"Finished {self.model_name} model warmup!")

    def inference(self, img: Image) -> str:
        caption = self.model(img)[0]["generated_text"]
        return caption
