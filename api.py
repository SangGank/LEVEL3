from fastapi import APIRouter, Form, File, UploadFile, Request
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from transformers import GPT2LMHeadModel
from tokenizer import get_custom_tokenizer
import torch
import os, shutil
from symusic import Score
import logging


# 로깅 설정
logging.basicConfig(level=logging.INFO)

from pydantic import BaseModel

class TextData(BaseModel):
    text: str

router = APIRouter()
templates = Jinja2Templates(directory="templates")

MODEL_DIR = "./model/"
TEMP_DIR = "./temp/"

model_path = os.path.join(MODEL_DIR, "bar4-ch4-checkpoint-8100")
model = GPT2LMHeadModel.from_pretrained(model_path) 
tokenizer = get_custom_tokenizer()

## 추가되는 부분
from transformers import AutoTokenizer

## frontModel.py 파일과 pickle 필요
from frontModelFunction import customBertForSequenceClassification, id2labelData_labels

front_model_path = './my_awesome_model/checkpoint-300'
front_tokenizer = AutoTokenizer.from_pretrained(front_model_path)
front_model = customBertForSequenceClassification.from_pretrained(front_model_path)

pickle_path = './labels.pkl'
emotion_dict , tempo_dict, genre_dict = id2labelData_labels(pickle_path)

##여기 까지



@router.post("/generate_midi/")
async def generate_midi(text: str = Form(...)):

    # 로그에 텍스트 출력
    logging.info("Received text: %s", text)
    
    ## generation midi
    initial_token = "BOS_None"
    generated_ids = torch.tensor([[tokenizer[initial_token]]])

    iteration_number = 0

    input_ids = generated_ids
    eos_token_id = tokenizer["Track_End"]
    temperature = 0.8
    generated_ids = model.generate(
        input_ids,
        max_length=1024,
        do_sample=True,
        temperature=temperature,
        eos_token_id=eos_token_id,
    ).cpu()

    midi_data = tokenizer.tokens_to_midi(generated_ids[0])

    file_path = os.path.join(TEMP_DIR, "temp_gen.mid")
    midi_data.dump_midi(file_path)

    return FileResponse(file_path, media_type="audio/midi")
    return StreamingResponse(open(file_path, "rb"), media_type="audio/midi")

@router.post("/upload_midi/")
async def upload_midi(midi_file: UploadFile = File(...)):
    temp_file_path = os.path.join(TEMP_DIR, "temp_upload.mid")
    try:
        # 업로드된 파일을 임시 폴더에 저장
        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(midi_file.file, temp_file)
        
        # 임시 파일 경로 반환
        return {"status": "success", "temp_file_path": temp_file_path}
    except Exception as e:
        return {"status": "failed", "message": str(e)}
    


## 추가되는 부분

@router.post("/model1/")
async def categoryModel(text: str = Form(...)):

    # 로그에 텍스트 출력
    logging.info("Received text: %s", text)

    inputs = tokenizer(text, return_tensors='pt')
    result = model(**inputs).logits

    emotion_id = int(result[0].detach().argmax())
    tempo_id = int(result[1].detach().argmax())
    genre_id = int(result[2].detach().argmax())

    emotion , tempo, genre = emotion_dict[emotion_id], tempo_dict[tempo_id], genre_dict[genre_id]
    
    # 로그에 텍스트 출력
    logging.info("emotion : %s,  tempo : %s,  genre : %s", emotion, tempo, genre)
    
    
    return emotion , tempo, genre

## 여기까지