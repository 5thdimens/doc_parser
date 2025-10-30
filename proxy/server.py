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
import json


system = "You are a reliable data extraction engine. Your sole purpose is to analyze the provided image and extract information."

user = """
You are an expert document processing assistant. Analyze the provided image of a personal identification document (e.g., passport, national ID, or driver's license) and extract the following information **only if clearly visible and legible**. Return your response **strictly as a valid JSON object** with the fields below. If a field is not present or unreadable, use `null`.

Fields to extract:
- document_type: one of ["passport", "national_id", "driver_license", "military_id", "other"]
- full_name
- document_number
- national_id
- nationality
- date_of_birth
- issue_date
- expiry_date
- gender
- issuing_country
- issuing_authority
- confidence: float (on a scale of 0.0 to 1.0)

Do not invent or guess any values. Only extract what is explicitly shown in the document.
"""







passport = '''
You MUST return your output in the following JSON format:
{
	"document_type": "string ('passport' or 'other')",
	"name": "string",
	"gender": "string",
	"country": "string",
	"date_of_birth": "string (usualy is the date with the smallest year)",
	"passport_number": "string",
	"confidence_score": "float (on a scale of 0.0 to 1.0)"
}

Please analyze this image and generate the JSON. In case a field is missing, field value should be empty or null.
'''

national_id = '''
You MUST return your output in the following JSON format:
{
	"document_type": "string ('kenya_national_id' or 'other')",
	"name": "string",
	"gender": "string",
	"date_of_birth": "string (usualy is the date with the smallest year)",
	"id_number": "string (usualy 8 digit number)",
    "serial_number": "string (usually 9 digit number)",
	"confidence_score": "float (on a scale of 0.0 to 1.0)"
}

Please analyze this image and generate the JSON. In case a field is missing, field value should be empty or null.
'''


military_id = '''
You MUST return your output in the following JSON format:
{
	"document_type": "string ('military_id' or 'other')",
	"name": "string",
    "service_number": "string",
    "rank": "string",
    "service": "string",
    "height": "string",
    "blood_group": "string",
    "national_id": "string (usualy 8 digit number)",
    "date_of_issue": "string",
	"confidence_score": "float (on a scale of 0.0 to 1.0)"
}

Please analyze this image and generate the JSON. In case a field is missing, field value should be empty or null.
'''




kra_pin = '''
You MUST return your output in the following JSON format:
{
	"document_type": "string ('kra_pin' or 'other')",
	"pin": "string",
    "email": "string",
    "phone": "string",
    "po_box": "string",
    "postal_code": "string",
    "county": "string",
    "district": "string",
    "city": "string",
    "street": "string",
    "building": "string",
	"confidence_score": "float (on a scale of 0.0 to 1.0)"
}

Please analyze this image and generate the JSON. In case a field is missing, field value should be empty or null.
'''


cert_of_reg = '''
You MUST return your output in the following JSON format:
{
	"document_type": "string ('certificate_of_registration' or 'other')",
	"business_name": "string",
    "country": "string",
    "registration_number": "string",
	"confidence_score": "float (on a scale of 0.0 to 1.0)"
}

Please analyze this image and generate the JSON. In case a field is missing, field value should be empty or null.
'''



cert_of_incorp = '''
You MUST return your output in the following JSON format:
{
	"document_type": "string ('certificate_of_incorporation' or 'other')",
	"business_name": "string",
    "country": "string",
    "registration_number": "string",
	"confidence_score": "float (on a scale of 0.0 to 1.0)"
}

Please analyze this image and generate the JSON. In case a field is missing, field value should be empty or null.
'''




owner_ship_cert = '''
You MUST return your output in the following JSON format:
{
    "document_type": "string ('title_deed', 'lease_agreement', 'shares_certificate', 'allotment_letter' or 'other')",
    "confidence_score": "float (on a scale of 0.0 to 1.0)"
}

Please analyze this image and generate the JSON. In case a field is missing, field value should be empty or null.
'''



collection = {
    "DT0002": {
        "doc_type": "kenya_national_id", 
        "prompt": national_id
    },
    "DT0049": {
        "doc_type": "passport", 
        "prompt": passport
    },
    "DT0081": {
        "doc_type": "military_id", 
        "prompt": military_id
    },
    "DT0030": {
        "doc_type": "certificate_of_registration", 
        "prompt": cert_of_reg
    },
    "DT0075": {
        "doc_type": "certificate_of_incorporation", 
        "prompt": cert_of_incorp
    },
    "DT0074": {
        "doc_type": "kra_pin", 
        "prompt": kra_pin
    },
    "DT0083": {
        "doc_type": "kra_pin", 
        "prompt": kra_pin
    },
    "DT0076": {
        "doc_type": "title_deed", 
        "prompt": owner_ship_cert
    },
    "DT0077": {
        "doc_type": "lease_agreement", 
        "prompt": owner_ship_cert
    },
    "DT0078": {
        "doc_type": "shares_certificate", 
        "prompt": owner_ship_cert
    },
    "DT0079": {
        "doc_type": "allotment_letter", 
        "prompt": owner_ship_cert
    },
}




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


SYSTEM_PROMPT = "You are a reliable data extraction engine. Your sole purpose is to analyze the provided image and extract information. Your entire response must be a single, valid JSON object, and you must include **no other text, explanations, or conversational filler**."



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
        image.save(buffer, format="JPEG")
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
            dpi=200,
            fmt='jpeg'
        )
        
        # Limit to max_pages if specified
        if max_pages:
            images = images[:max_pages]
        
        return images

    except Exception as e:
        logger.error(f"Error converting PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF conversion failed: {str(e)}")



async def analyze_images(images: List[str], doc_type: str) -> dict:
    
    if doc_type not in collection:
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
        
        prompt = collection.get(doc_type).get("prompt")

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
        
        payload = {
            "messages": messages,
        }
        
        async with httpx.AsyncClient(timeout=VLLM_TIMEOUT) as client:
            '''
            response = await client.post(
                VLLM_URL,
                headers=headers,
                json=payload
            )
            '''
            response = await client.get(
                "http://weather:8001/weather",
                headers=headers
            )
            
            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"API error: {response.text}"
                )
            
            response = response.json()
            '''content = result["choices"][0]["message"]["content"]
            if content.startswith("```json") and content.endswith("```"):
                content = content[8:-4]
            response = json.loads(content)'''

            return response
            
    except httpx.TimeoutException:
        logger.error("API request timed out")
        raise HTTPException(status_code=504, detail="API request timed out")
    except httpx.RequestError as e:
        logger.error(f"API request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error calling API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"API error: {str(e)}")


def normalize_image_for_gemma_vlm(image_url: str) -> str | None:
    """
    Downloads an image from a URL, resizes it to the Gemma 3 VLM's required 
    896x896 resolution, normalizes the pixel data, and encodes the result
    into a Base64 string (PNG format).

    Note: This function performs generic preprocessing (resize and 0-1 scaling).
    For production use with the Gemma 3 VLM, always use the official 
    image processor from the model's library (e.g., Hugging Face's transformers) 
    to ensure all normalization steps (mean/std, specific color conversions) are
    exactly correct.

    Args:
        image_url: The public URL of the image to process.

    Returns:
        A Base64-encoded string of the preprocessed image in PNG format, 
        or None if an error occurs.
    """
    try:
        # 1. Download the image data
        logging.info(f"Attempting to download image from: {image_url}")
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes

        # 2. Open the image using PIL (Pillow)
        image_data = io.BytesIO(response.content)
        img = Image.open(image_data).convert("RGB")
        logging.info(f"Original image size: {img.size}")

        # 3. Resize the image to the VLM's fixed input resolution (896x896)
        # Using Image.Resampling.LANCZOS for high quality downsampling
        img_resized = img.resize(GEMMA_VLM_SIZE, Image.Resampling.LANCZOS)
        logging.info(f"Resized image size: {img_resized.size}")

        # 4. Convert the resized image to a NumPy array
        img_array = np.array(img_resized, dtype=np.float32)

        # 5. Normalize pixel values (scaling from [0, 255] to [0.0, 1.0])
        # This step is often performed internally by the model's processor, 
        # but we include it here for completeness before encoding the final image.
        # Note: If saving to PNG/JPEG, we must revert to 0-255 scale first.
        
        # 6. Prepare image for Base64 encoding (convert back to 8-bit and PIL)
        # We ensure the data is in the expected 0-255 range and unit8 type before
        # converting back to a savable PIL Image object.
        img_8bit = (img_array).astype(np.uint8)
        final_img = Image.fromarray(img_8bit, 'RGB')
        
        # 7. Save to an in-memory buffer as PNG and encode to Base64
        buffer = io.BytesIO()
        final_img.save(buffer, format="PNG")
        base64_encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        logging.info("Image successfully preprocessed, normalized, and Base64 encoded.")
        logging.debug(f"Encoded string length: {len(base64_encoded_image)}")

        return base64_encoded_image

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching image from URL: {e}")
        return None
    except IOError as e:
        logging.error(f"Error reading or processing image file: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None




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

        
        return JSONResponse(content={
            "success": True,
            "processing_time": processing_time,
            "result": vllm_result
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
    return JSONResponse(content={
        "status": "healthy",
        "service": "file-upload-api",
        "upload_directory": str(UPLOAD_DIR.absolute()),
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024)
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6060)
