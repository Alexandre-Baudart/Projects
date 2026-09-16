from fastapi import FastAPI
from src.api.routes import ml

app = FastAPI(
    title="ML API",
    version="1.0.0",
)

app.include_router(ml.router)

# uvicorn src.api.api_main:app --reload
# Open Swagger : http://127.0.0.1:8000/docs


