from typing import Dict, List

import uvicorn
from fastapi import BackgroundTasks, FastAPI

from inference import GenerateRequest, Generator

app = FastAPI()
generator = Generator()

@app.post("/generate")
async def generate(
    requests: List[GenerateRequest], background_tasks: BackgroundTasks
) -> Dict[str, str]:
    background_tasks.add_task(generator.generate, requests)
    return {"message": "Task is being processed", "status": "accepted"}


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8006,
        log_level="info",
        # reload=True,
    )
