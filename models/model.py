from PIL.Image import Image
import os


class Model:
    def __init__(self, name: str):
        self.model_name = name

    def get_warmup_imgs(self) -> Image:
        for filename in os.listdir(os.path.join("imgs"))[:3]:
            img = Image.open(filename)
            yield img

    def warmup(self):
        raise NotImplementedError

    def inference(self, img: Image) -> str:
        raise NotImplementedError
