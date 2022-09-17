FROM python:3.9-slim

# os level reqs
RUN apt-get update -y \
     && apt install libgl1-mesa-glx -y \
     && apt-get install 'ffmpeg' 'libsm6' 'libxext6' -y \
     && apt-get install -y build-essential libzbar-dev

RUN pip install install --upgrade pip

# install large requirements to save time in next builds
RUN pip install --no-cache-dir tensorflow \ 
     && pip install --no-cache-dir numpy \
     && pip install --no-cache-dir opencv-python \
     && pip install --no-cache-dir matplotlib 

COPY . /app
WORKDIR /app/src

RUN pip install -r ../requirements.txt

EXPOSE 8000

CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]