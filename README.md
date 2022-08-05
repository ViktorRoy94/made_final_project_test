## Task

* Create binary classifier Male/Female voices using LibriTTS dataset

## Train

* Please see notebooks/EDA.ipynb

## Demo

Besides model training I created python demo that run small RestApi server and can predict voice in \*.wav files.
File should be at least 1s, but less than 60s. File should have one mono channel.

You may run demo with commands:

* ``` docker build -t online_male_female:v1 .```
* ``` docker run -p 8000:8000 online_male_female:v1 ```
* ``` python3 src/make_request.py --data /path/to/file.wav ```
