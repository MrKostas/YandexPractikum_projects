import uvicorn
import argparse
import logging
import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from model.preprocessor import Preprocessor
from model.model import Model

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)
app_handler = logging.StreamHandler()
app_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
app_handler.setFormatter(app_formatter)
app_logger.addHandler(app_handler)

@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse('start_form.html', context={'request': request})

@app.get("/start")
async def main(request: Request):
    return templates.TemplateResponse('start_form.html', context={'request': request})

@app.post("/submit")
async def process_request(request: Request,
                          file_path: str = Form(...),
                          file_extension: str = Form(...),
                          prob: float = Form(...)):
    if not os.path.exists(file_path):
        return templates.TemplateResponse('error_not_found.html', context={'request': request})
    else:
        preprocessor = Preprocessor(file_path, file_extension)
        if preprocessor.status == 'fail':
            return templates.TemplateResponse('error_preprocessing.html', context={'request': request})
        else:
            model = Model(data = preprocessor.data,
                          id = preprocessor.id,
                          threshold = prob,
                          file_path = file_path,
                          file_extension = file_extension)
            model.save_predictions()
            return templates.TemplateResponse('success.html', context={'request': request})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())

    uvicorn.run(app, **args)