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

## Usage for `main.py`

First install required libraries by running:

    pip install -r requirements.txt

To run the application run following command in `src` dir:

    uvicorn main:app --reload

or

    chmod +x main.sh
    ./main.sh

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
