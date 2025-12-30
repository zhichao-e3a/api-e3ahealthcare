from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

def install_middleware(app: FastAPI) -> None:

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://api.ai.e3ahealth.com",
            "https://35.240.213.6:8000/",
            "http://35.240.213.6:8000/"
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[
            "api.ai.e3ahealth.com",
            "35.240.213.6",
            "localhost",
            "127.0.0.1",
            "[::1]",
            "*.local",
            "host.docker.internal",
            "host.docker.internal:8000",
            "172.17.0.1"
        ],
    )