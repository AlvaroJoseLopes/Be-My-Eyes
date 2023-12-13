from PIL import Image
from .model import BaseModel
import pickle
from loguru import logger
import os

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.saving import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np

START_TOKEN = "stsq"
END_TOKEN = "endsq"
MODEL_WEIGHTS = os.path.join("weights", "BeMyEyes_checkpoint.keras")
MAX_LENGTH = 35


class VggLstmModel(BaseModel):
    def __init__(self):
        super().__init__("trained")
        logger.info(f"Loading VGG model ...")
        self.vgg = VGG16()
        self.vgg = Model(inputs=self.vgg.inputs, outputs=self.vgg.layers[-2].output)
        self.lstm = load_model(MODEL_WEIGHTS)
        with open(os.path.join("weights", "tokenizer.pkl"), "rb") as handle:
            self.tokenizer = pickle.load(handle)
        self.idx_to_word = self.create_map_token_idx_to_word(self.tokenizer)
        self.warmup()

    def warmup(self) -> None:
        logger.info(f"Warming up {self.model_name} model...")
        for img in self.get_warmup_imgs():
            _ = self.inference(img)
        logger.info(f"Finished {self.model_name} model warmup!")

    def create_map_token_idx_to_word(self, tokenizer: Tokenizer) -> dict[int, str]:
        idx_to_word = {}
        for word, index in tokenizer.word_index.items():
            idx_to_word[index] = word
        return idx_to_word

    def inference(self, img: Image.Image) -> str:
        img = img.resize((224, 224))
        img = img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

        # Preprocess image to feed into VGG and produce its features
        img = preprocess_input(img)
        img_feature = self.vgg.predict(img, verbose=0)

        in_text = START_TOKEN
        for _ in range(MAX_LENGTH):
            [sequence] = self.tokenizer.texts_to_sequences([in_text])
            sequence = pad_sequences([sequence], MAX_LENGTH)

            # Inference to predict the next word
            yhat = self.lstm.predict([img_feature, sequence], verbose=0)
            yhat = np.argmax(yhat)

            # Concatenate the results to get a new sequence
            word = self.idx_to_word[yhat]
            in_text = " ".join([in_text, word])
            if word == END_TOKEN:
                break

        return in_text.replace(START_TOKEN, "").replace(END_TOKEN, "").strip()

        return "teste"
