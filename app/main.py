from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def okok():
    return {'message': "hello world wow boy!!"}

@app.get('/a')
def okok():
    return {'message': "hello world wow boy!! aaaa"}