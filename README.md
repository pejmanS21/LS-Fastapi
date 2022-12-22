# Lung Segmentation with fastapi

This app uses `FastAPI` as backend.

## Usage for `app.py`

First install required libraries by running:

    pip install -r requirements.txt

To run the application run following command in `src` dir:

    uvicorn app:app --reload

or

    chmod +x app.sh
    ./app.sh

## Tutorial for `app.py`

![app.gif](images/app.gif)

## Tutorial

### `main page`

    http://localhost:8000/

![main.png](./images/main.png)

### `fastapi documentation`

    http://localhost:8000/docs

![docs.png](./images/docs.png)

### `show results`

    http://localhost:8000/imshow

![imshow.png](./images/imshow.png)

---

## DVC

```bash
pip install dvc dvc-gdrive

# pull weights from Google Drive
dvc pull
```
in [weights](./weights/) directory
```
weights
├── cxr_resunet.tflite
├── cxr_resunet.tflite.dvc
├── cxr_unet.tflite
└── cxr_unet.tflite.dvc
```

---

## Docker

```bash
# Build image
docker build -t IMAGE_NAME:TAG_NAME .
docker run -p 8000:8000 -d IMAGE_NAME:TAG_NAME
```

Or

```bash
# for amd64 systems
docker run -d -p 8000:8000 pejmans21/ls-fastapi:0.1.0

#### OR

# for arm64 systems
docker run -d -p 8000:8000 pejmans21/ls-fastapi:aarch64
```

---

## Kubernetes

```bash
kubectl apply -f ls-fastapi-k8s-config.yaml
```
to see output
```bash
kubectl port-forward service/lsapi-service 8000
```

>Now check http://127.0.0.1:8000/

### Stop process
```bash
kubectl delete -f ls-fastapi-k8s-config.yaml
```