
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="AgriVision AI")

@app.get("/")
def health_check():
    return {"status": "ok", "project": "AgriVision AI"}

print("FastAPI app created successfully!")
print("Uvicorn version:", uvicorn.__version__)
print("API title:", app.title)
print("API routes:", [route.path for route in app.routes])
