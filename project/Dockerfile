FROM python:3.9
WORKDIR /code
RUN apt-get update && apt-get install -y libgl1-mesa-glx
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code
RUN mkdir -p /code/models && chmod 777 /code/models
RUN mkdir -p /code/uploads && chmod 777 /code/uploads

CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app", "--timeout", "120"]
