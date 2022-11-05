import logging

from fastapi import FastAPI

from app.api import prediction

log = logging.getLogger("uvicorn")


def create_application() -> FastAPI:
    application = FastAPI()
    application.include_router(
        prediction.router, prefix="/prediction", tags=["prediction"]
    )

    return application


app = create_application()


@app.on_event("startup")
async def startup_event():
    log.info("Starting up...")


@app.on_event("shutdown")
async def shutdown_event():
    log.info("Shutting down...")


"""
Uncomment for testing
"""
# import uvicorn
#
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
