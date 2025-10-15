from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import os
import uuid
from pathlib import Path
from PIL import Image
import pdf2image
import logging
import base64
import httpx
from datetime import datetime
from io import BytesIO
import time
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from enum import Enum



LOG_LEVEL=os.getenv("LOG_LEVEL", "ERROR")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("uploads")
IMAGE_DIR = UPLOAD_DIR / "images"
PDF_DIR = UPLOAD_DIR / "pdfs"
CONVERTED_DIR = UPLOAD_DIR / "converted"

# Create directories
for directory in [IMAGE_DIR, PDF_DIR, CONVERTED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


VLLM_URL = os.getenv("VLLM_URL")
VLLM_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "30")) #default 30 sec




# Allowed file types
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".bmp"}
ALLOWED_PDF_EXTENSION = ".pdf"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


SYSTEM_PROMPT = "You are a helpful assistant."

'''
You are a reliable data extraction engine. Your sole purpose is to analyze the provided image and extract information. Your entire response must be a single, valid JSON object, and you must include **no other text, explanations, or conversational filler**.
'''

'''
{
  "model": "your-openai-compatible-model-name",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "**[YOUR FULL PROMPT/INSTRUCTIONS FROM STEP 3 GO HERE]**\n\nFor example: \"You are a helpful AI assistant. Always follow these rules: 1. Use concise language. 2. Output must be valid JSON only. Analyze the provided image and identify the main product and its price.\""
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,**[YOUR_BASE64_ENCODED_IMAGE_STRING_GOES_HERE]**"
          }
        }
      ]
    }
  ],
  "max_tokens": 1024,
  "temperature": 0.1,
  
  // *** OPTIONAL: To force JSON output (if supported by the server/model) ***
  "response_format": {
    "type": "json_object"
  }
}


You must first encode your image file (e.g., JPEG, PNG) into a Base64 string. This string then replaces [YOUR_BASE64_ENCODED_IMAGE_STRING_GOES_HERE] and is prefixed with the Data URL mime type (data:image/jpeg;base64, or data:image/png;base64,).


'''

class PersonalDocType(str, Enum):
    passport = "passport"
    national_id = "national_id"
    military_id = "military_id"
    defence_forces = "defence_forces"


class KraDocType(str, Enum):
    business = "business"
    individual = "individual"


class OwnershipDocType(str, Enum):
    title_deed = "title_deed"
    lease_agreement = "lease_agreement"
    shares_certificate = "shares_certificate"
    allotment_letter = "allotment_letter"

class CertificateDocType(str, Enum):
    registration = "registration"
    incorporations = "incorporations"



class PersonalDocument(BaseModel):
    firstName: str
    middleName: str
    surname: str
    gender: str
    country: str
    documentType: PersonalDocType
    documentNumber: str
    birthDate: str


#https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html

'''

PERSONAL_DOC (PASSPORT, ID, MILITARY ID)
	firstName: z.string(Errors.REQUIRED).min(1, Errors.REQUIRED),
	middleName: z.string().optional(),
	surname: z.string(Errors.REQUIRED).min(1, Errors.REQUIRED),
	gender: z.string(Errors.REQUIRED).min(1, Errors.REQUIRED),
	country: z.string(Errors.REQUIRED).min(1, Errors.REQUIRED),
	documentType: z.string(Errors.REQUIRED).min(1, Errors.REQUIRED),
	documentNumber: z.string(Errors.REQUIRED).min(1, Errors.REQUIRED),
	birthDate: z.date(Errors.REQUIRED),


KRA PIN (BUSINESS OR INDIVIDUAL)
	pin: z.string(Errors.REQUIRED).min(1, Errors.REQUIRED),
	email: z.email(Errors.INVALID_EMAIL),
	phone: z.string(Errors.REQUIRED).regex(PHONE_REGEX, Errors.INVALID_PHONE),
	postalAddress: z.string(Errors.REQUIRED).min(1, Errors.REQUIRED),
	postalCode: z.string(Errors.REQUIRED).min(1, Errors.REQUIRED),
  

Certificate of Registration/Incorporations
	businessName: z.string(Errors.REQUIRED).min(1, Errors.REQUIRED),
	country: z.string(Errors.REQUIRED).min(1, Errors.REQUIRED),
	documentType: z.string(Errors.REQUIRED).min(1, Errors.REQUIRED),
	documentNumber: z.string(Errors.REQUIRED).min(1, Errors.REQUIRED),

'''

DOC_TYPE_TEMPLATES = {  
  'DT0002': 'Extract data in json format', 
  'DT0049': 'PASSPORT',
  'DT0081': 'MILITARY_ID',
  'DT0030': 'CERT_OF_REGISTRATION',
  'DT0075': 'CERT_OF_INCORPORATIONS',
  
  'DT0074': 'BUSINESS KRA PIN', 
  'DT0083': 'INDIVIDUAL KRA PIN',

  'DT0076': 'TITLE_DEED', 
  'DT0077':	'LEASE_AGREEMENT',
  'DT0078':	'SHARES_CERTIFICATE',
  'DT0079':	'ALLOTMENT_LETTER'
}



'''
DT0002 - NATIONAL_ID
DT0049 - PASSPORT
DT0081 - MILITARY_ID
DT0030 - CERT_OF_REGISTRATION
DT0075 - CERT_OF_INCORPORATIONS
DT0074 - BUSINESS KRA PIN 
DT0083 - INDIVIDUAL KRA PIN
DT0076 - TITLE_DEED 
DT0077 - LEASE_AGREEMENT
DT0078 - SHARES_CERTIFICATE
DT0079 - ALLOTMENT_LETTER
'''



def validate_file_size(file: UploadFile) -> bool:
    """Validate file size"""
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    return file_size <= MAX_FILE_SIZE


def get_file_extension(filename: str) -> str:
    """Get file extension in lowercase"""
    return Path(filename).suffix.lower()


def save_uploaded_file(file: UploadFile, directory: Path) -> Path:
    """Save uploaded file with unique name"""
    file_ext = get_file_extension(file.filename)
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = directory / unique_filename
    
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    
    return file_path


def image_to_base64(file: UploadFile) -> str:
    try:
        return base64.b64encode(file.file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image encoding failed: {str(e)}")


def pil_image_to_base64(image: Image) -> str:
    try:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image encoding failed: {str(e)}")


def pdf_to_images(file: UploadFile, max_pages: int = None) -> List[Image.Image]:
    try:
        buffer = BytesIO()
        file.save(buffer)
        bytes = buffer.getvalue()
        images = pdf2image.convert_from_bytes(
            bytes,
            dpi=300,
            fmt='png'
        )
        
        # Limit to max_pages if specified
        if max_pages:
            images = images[:max_pages]
        
        return images

    except Exception as e:
        logger.error(f"Error converting PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF conversion failed: {str(e)}")



async def analyze_images(images: List[str], doc_type: str) -> dict:
    
    if doc_type not in DOC_TYPE_TEMPLATES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported document type"
        )

    try:
        image_contents = []
        for img in images:
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": img
                }
            })
        
        print(image_contents)
        # Construct the prompt based on doc_type
        #prompt = DOC_TYPE_TEMPLATES[doc_type]
        prompt = "what is the capital of the world?"

        # Build messages
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text", 
                        "text": SYSTEM_PROMPT
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    #*image_contents
                ]
            }
        ]


        
        # Make API request
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        json_schema = PersonalDocument.model_json_schema()

        payload = {
            "messages": messages,
            "extra_body": {
                "guided_json": json_schema
            }
        }
        
        async with httpx.AsyncClient(timeout=VLLM_TIMEOUT) as client:
            response = await client.post(
                VLLM_URL,
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"API error: {response.text}"
                )
            
            result = response.json()

            '''
            {
                'id': 'chatcmpl-09d1b5bfb937420097e118f09c9a6dca',
                'object': 'chat.completion',
                'created': 1760552858,
                'model': 'google/gemma-3-1b-it',
                'choices': [{
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': 'That\'s a fantastic and delightfully tricky question! There isn\'t one single, official "capital of the world." It\'s a really interesting concept! \n\nHowever, if you\'re looking for the most commonly cited and historically significant answer, it’s **Paris, France.** \n\nIt’s become a popular symbol of global influence and a frequent topic of discussion. \n\nBut, it’s worth noting that it\'s a bit of a playful answer! 😊',
                        'refusal': None,
                        'annotations': None,
                        'audio': None,
                        'function_call': None,
                        'tool_calls': [],
                        'reasoning_content': None
                    },
                    'logprobs': None,
                    'finish_reason': 'stop',
                    'stop_reason': 106,
                    'token_ids': None
                }],
                'service_tier': None,
                'system_fingerprint': None,
                'usage': {
                    'prompt_tokens': 24,
                    'total_tokens': 124,
                    'completion_tokens': 100,
                    'prompt_tokens_details': None
                },
                'prompt_logprobs': None,
                'prompt_token_ids': None,
                'kv_transfer_params': None
            }

            '''
            
            return {
                "analysis": result["choices"][0]["message"]["content"],
                "model": result["model"],
                "usage": result.get("usage", {}),
                "finish_reason": result["choices"][0]["finish_reason"]
            }
            
    except httpx.TimeoutException:
        logger.error("API request timed out")
        raise HTTPException(status_code=504, detail="API request timed out")
    except httpx.RequestError as e:
        logger.error(f"API request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error calling API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"API error: {str(e)}")



@app.post("/process")
#@limiter.limit("1/minute")
async def process_file(
    request: Request,
    file: UploadFile = File(...),
    doc_type: str = Form(...)
):

    try:
        start_time = time.perf_counter()
        file_ext = get_file_extension(file.filename)
        
        if not validate_file_size(file):
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024)}MB"
            )

        images_to_analyze: List[str] = []
        task_uuid = uuid.uuid4()
        
        if file_ext == ".jpg":
            base64 = image_to_base64(file)
            images_to_analyze.append(f"data:image/jpg;base64,{base64}")

        elif file_ext == ".jpeg":
            base64 = image_to_base64(file)
            images_to_analyze.append(f"data:image/jpeg;base64,{base64}")

        elif file_ext == ".bmp":
            base64 = image_to_base64(file)
            images_to_analyze.append(f"data:image/bmp;base64,{base64}")

        elif file_ext == ".pdf":
            pil_images = pdf_to_images(file, max_pages=5)
            
            for img in pil_images:
                base64 = pil_image_to_base64(img)
                images_to_analyze.append(f"data:image/png;base64,{base64}")

            if len(images_to_analyze) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unable to process PDF file"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)} and .pdf"
            )


        logger.info(f"Processing file: {file.filename} as {doc_type}")

        vllm_result = await analyze_images(images_to_analyze, doc_type)
        
        end_time = time.perf_counter()
        processing_time = end_time - start_time

        result = {}
        result["analysis"] = vllm_result
        result["processing_time"] = processing_time
        

        return JSONResponse(content={
            "success": True,
            "message": f"File processed and analyzed successfully as {doc_type}",
            "result": result
        })

    except HTTPException as er:
        logger.error(f"Error processing file: {str(er)}", exc_info=True)
        raise er

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "file-upload-api",
        "upload_directory": str(UPLOAD_DIR.absolute()),
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6060)
