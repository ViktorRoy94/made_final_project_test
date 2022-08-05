FROM pytorch/pytorch
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/app.py app.py
COPY src/model.py model.py

WORKDIR .

ENV MODEL_URL="https://drive.google.com/file/d/1i2QBfIdI1pPz3LUnfqNT3iinrfli8vrC/view?usp=sharing"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]