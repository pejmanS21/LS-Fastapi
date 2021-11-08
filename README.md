# Lung Segmentation with fastapi

This app uses `FastAPI` as backend.

## Usage

First install required libraries by running:

    pip install -r requirements.txt

To run the application run following command in `src` dir:

    uvicorn main:app --reload

or

    chmod +x run.sh
    ./run.sh

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
