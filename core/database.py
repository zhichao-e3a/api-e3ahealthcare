from fastapi import Request

def get_mongo(request: Request):
    return request.app.state.mongo
