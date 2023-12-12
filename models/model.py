from PIL import Image
import os

IMGS_FOLDER = "imgs"


class Model:
    def __init__(self, name: str):
        self.model_name = name

    def get_warmup_imgs(self) -> Image.Image:
        for filename in os.listdir(os.path.join(IMGS_FOLDER))[:3]:
            img = Image.open(os.path.join(IMGS_FOLDER, filename))
            yield img

    def warmup(self) -> None:
        raise NotImplementedError

    def inference(self, img: Image) -> str:
        raise NotImplementedError
