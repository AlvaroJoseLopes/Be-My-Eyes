# Be My Eyes

FastAPI + Streamlit

## Running the app

Firstly, install all the dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Backend

```bash
uvicorn backend:app
```

For default, the backend will be running at port `127.0.0.1:8000`.

## Frontend

```bash
streamlit
```

For default, the frontend will be running at port `8501`.
