from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from google import genai
from google.genai import types
# from google.generativeai import types
from fastapi import FastAPI, File, UploadFile, APIRouter,Form
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import asyncio
import base64
import os
import io
import uvicorn
from typing import Annotated

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vertexai_key.json'

# Define project information
PROJECT_ID = ""  # @param {type:"string"} removed for security purposes
LOCATION = "us-central1"  # @param {type:"string"}
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_headers=['*'],
    allow_methods=['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS'], # allow_methods=['*'],
)

def generate_content(photo, photo_type, text_prompt):
    # Define generation config to improve reproducibility
    # gemini_model = "gemini-2.0-flash-exp"
    gemini_model = "gemini-2.0-flash-preview-image-generation"
    generation_config = types.GenerateContentConfig(
        temperature=0.5,
        top_p=0.8,
        top_k=10,
        candidate_count=1,
        max_output_tokens=1024,
        response_modalities = ["Text", "Image"],
    )
    prompt_photo = types.Part.from_bytes(
        data=photo,
        mime_type=photo_type, #mine_type = photo_type
    )
    prompt_text = types.Part.from_text(text=text_prompt)
    contents = [types.Content(
        role="user",
        parts=[prompt_photo, prompt_text]
    )]

    responses = client.models.generate_content(
        model=gemini_model,
        contents=contents, #contents=text_prompt,
        config=generation_config,
    )
    # print("checking response:", responses)


    return responses


def find_total_on_reciept(photo, photo_type, text_prompt):
    gemini_model = "gemini-2.0-flash-exp"

    generation_config = types.GenerateContentConfig(
        temperature=0.5,
        top_p=0.8,
        top_k=10,
        candidate_count=1,
        max_output_tokens=1024,
        response_modalities=["Text"],
    )

    prompt_photo = types.Part.from_bytes(
        data=photo,
        mime_type=photo_type,
    )
    prompt_text = types.Part.from_text(text=text_prompt)
    
    contents = [types.Content(
        role="user",
        parts=[prompt_photo, prompt_text]
    )]

    responses = client.models.generate_content(
        model=gemini_model,
        contents=contents,
        config=generation_config,
    )

    
    text_response = responses.candidates[0].content.parts[0].text
    return text_response

class PromptRequest(BaseModel):
    prompt: str
    language: str = "English"

@app.exception_handler(Exception)
async def validation_exception_handler(request, err):
    return JSONResponse(status_code=500, content={
        'detail': [
            {'msg': str(err), 'type': type(err).__name__}
        ]
    })


@app.get("/")# get is for fast api but route is for flask
async def root():
    # return{"message": "Hello all"}
    return FileResponse('index.html')

@app.post("/annotate_receipt")
async def long_operation(photo: UploadFile,  tax_rate: Annotated[float, Form()]):
    photo_content = await photo.read()
    
    
    
   
    total_tax = float(find_total_on_reciept(photo=photo_content, photo_type=photo.content_type, text_prompt="What's the total tax on the image? only return the number"))
    #double check picture in terminal just in cas
    print(f"total_amount: {total_tax}")

    total_amount = float(find_total_on_reciept(photo=photo_content, photo_type=photo.content_type, text_prompt="What's the total amount on the image? only return the number"))

    print(f"total_amount: {total_amount}")

    
    final_amount = total_amount - total_tax

    # new_total_amount = final_amount*(1+total_tax/100)
    new_total_amount = final_amount*(1+tax_rate/100)

    # photo_result = generate_content(photo=photo_content, photo_type=photo.content_type, text_prompt=f"on a horizontal yellow background across the image with red text on top, place{final_amount} on the image ")
    photo_result = generate_content(photo=photo_content, photo_type=photo.content_type, text_prompt=f"preserve the original image to the best of your ability. the original image should not change in anyway. then across the original image, place a horizontal yellow background as an overlay, with red text write 'If tax were {tax_rate}%, the total amount would be ${new_total_amount}.' that text ")
    image = photo_result.candidates[0].content.parts[0].inline_data.data
    print(image)
    buffer = io.BytesIO(image)
    Image.open(io.BytesIO(image)).save(buffer,'PNG')
    image=base64.b64encode(buffer.getvalue())
    
    # img = image.new('RGB',(400,500), "red")


    # return dict(image=base64.b64encode(buffer.getvalue()))#return dict(image=base64.b4encode(buffer.getvalue()))

    return {
    "amount_before_tax" : final_amount,
    "receipt" : { 
     "data" :   image,
    "mimeType": "image/PNG",
    
    }}


    # with open('data.png', "wb") as fo:
    #     fo.write(image)


#last task: Annotate Image: overlay (annotate) the image with the following text: "If tax were {tax_rate}%, the total amount would be ${new_total_amount}."
# app.include_router(router)