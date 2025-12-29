from fastapi import Request

def get_mongo(request: Request):
    return request.app.state.mongo

def get_sql(request: Request):
    return request.app.state.sql