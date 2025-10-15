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
VLLM_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "120"))




# Allowed file types
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".bmp"}
ALLOWED_PDF_EXTENSION = ".pdf"
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB



DOC_TYPE_TEMPLATES = {  
  'DT0002': 'NATIONAL_ID',
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
    
    try:
        image_contents = []
        for img in images:
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": img
                }
            })
        
        # Construct the prompt based on doc_type
        prompt = f"Analyze this {doc_type} document and extract all relevant information. Provide a structured summary of the key details found in the document."
        
        # Build messages
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text", 
                        "text": "You are a helpful assistant."
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
                    *image_contents
                ]
            }
        ]
        
        # Make API request
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        payload = {
            "messages": messages,
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
@limiter.limit("1/minute")
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
        
        if doc_type not in DOC_TYPE_TEMPLATES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported document type"
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
    uvicorn.run(app, host="0.0.0.0", port=8000)