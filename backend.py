from fastapi import FastAPI
from pydantic import BaseModel, Field

import base64
from PIL import Image
from io import BytesIO
from enum import Enum

from models.pretrained import PretrainedModel

from models.vgg_lstm import VggLstmModel

models = {"pretrained": PretrainedModel(), "vgg+lstm": VggLstmModel()}


# All models available
class ModelName(str, Enum):
    pretrained = "pretrained"
    vgg_lstm = "vgg+lstm"


# Expected Body definition
class TargetImage(BaseModel):
    filename: str = Field(description="File name")
    content: str = Field(description="Image content encoded base64")


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Be My Eyes API is running"}


@app.post(
    "/inference",
    summary="Get image caption",
    description="Retrieve the caption for a specified image from a designated model",
)
async def inference(model_name: ModelName, image: TargetImage):
    img = Image.open(BytesIO(base64.b64decode(image.content)))
    default_message = "CAPTION  GENERATED BY MODEL, RESPONSE FROM THE API."

    model = models.get(model_name, None)
    if model is not None:
        return {"caption": model.inference(img)}

    return default_message
