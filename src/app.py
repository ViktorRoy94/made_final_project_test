import os
import sys
import logging
import gdown
import torch
import torchaudio
import numpy as np

from typing import List, Union, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .model import MaleFemaleModel

DEFAULT_MODEL_URL = "https://drive.google.com/file/d/1i2QBfIdI1pPz3LUnfqNT3iinrfli8vrC/view?usp=sharing"
DEFAULT_HELLO_MESSAGE = "I'm glad to see you here!"
DEFAULT_SAMPLE_RATE = 16000
MIN_AUDIO_LENGTH = 1
MAX_AUDIO_LENGTH = 60

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


app = FastAPI()
model: Optional[torch.nn.Module] = None


class MaleFemaleModelRequest(BaseModel):
    data: List[List[Union[float, int, None]]]
    sample_rate: int


class MaleFemaleResponse(BaseModel):
    sex: str


def download_model(url: str, model_path: str):
    gdown.download(url, model_path, fuzzy=True)


def load_model(path_to_model: str):
    model = MaleFemaleModel(input_size=64, time_size=101)
    with open(path_to_model, "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)
    return model


@app.on_event("startup")
def create_model():
    global model
    model_url = os.getenv("MODEL_URL")
    model_path = 'model.pth'

    if model_url is None:
        model_url = DEFAULT_MODEL_URL

    download_model(model_url, model_path)
    model = load_model(model_path)


@app.get("/")
def root():
    return DEFAULT_HELLO_MESSAGE


@app.get("/health")
def health() -> bool:
    print(model)
    return not (model is None)


@app.get("/predict/", response_model=MaleFemaleResponse)
def predict(request: MaleFemaleModelRequest):
    waveform = np.array(request.data)
    sr = request.sample_rate

    if waveform.shape[1] < sr * MIN_AUDIO_LENGTH:
        msg = f"Audio file length should be at least {MIN_AUDIO_LENGTH}s"
        raise HTTPException(status_code=400, detail=msg)
    if waveform.shape[1] > sr * MAX_AUDIO_LENGTH:
        msg = "Audio file is too big"
        raise HTTPException(status_code=400, detail=msg)

    input_transform = torch.nn.Sequential(
        torchaudio.transforms.Resample(orig_freq=sr,
                                       new_freq=DEFAULT_SAMPLE_RATE),
        torchaudio.transforms.MelSpectrogram(
            sample_rate=DEFAULT_SAMPLE_RATE,
            n_fft=512,
            win_length=int(0.025 * DEFAULT_SAMPLE_RATE),
            hop_length=int(0.01 * DEFAULT_SAMPLE_RATE),
            n_mels=64
        )
    )

    try:
        # Run model on multiple parts of audio file for better accuracy
        i = 0
        predictions = []
        while i < waveform.shape[1] and waveform.shape[1] - i >= sr * MIN_AUDIO_LENGTH:
            data = waveform[:, i:i + sr * MIN_AUDIO_LENGTH]
            data = torch.Tensor(data)

            data = input_transform(data)
            pred = model(data.unsqueeze(0))[0]
            predictions.append(pred > 0.5)
            i += sr * MIN_AUDIO_LENGTH

        # Check most popular answer
        male_count = sum(predictions)
        female_count = len(predictions) - male_count
        ans = "Male" if male_count > female_count else "Female"
        return MaleFemaleResponse(sex=ans)

    except RuntimeError as err:
        msg = f"Runtime error in prediction - {err}"
        raise HTTPException(status_code=500, detail=msg)
