from fastapi import FastAPI
from starlette.middleware.trustedhost import TrustedHostMiddleware

def install_middleware(app: FastAPI) -> None:

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
            "172.17.0.1"
        ],
    )