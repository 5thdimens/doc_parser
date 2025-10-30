from vllm import LLM, SamplingParams
#from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoTokenizer


# Create model instance
llm = LLM(
    model="OpenGVLab/InternVL3_5-2B-Instruct",
    #model="Qwen/Qwen3-VL-2B-Instruct",
    max_model_len=4096,
    max_num_seqs=2,
    dtype="half",
    trust_remote_code=True,
    #enable_prefix_caching=False,
    #mm_processor_cache_gb=0,
    #logits_processors=[NGramPerReqLogitsProcessor]
)



from vllm import LLM, SamplingParams
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoTokenizer

#response = requests.get("https://scontent-otp1-1.xx.fbcdn.net/v/t1.6435-9/94386732_3111029968930958_1209861092935729152_n.jpg?_nc_cat=108&ccb=1-7&_nc_sid=833d8c&_nc_ohc=eewcPGDSpcsQ7kNvwECk7Ef&_nc_oc=Adne7dlZo0DiARiOImRE_Vtn1UYQAr1S01QVLI_2MwnYSqkkpHaOrRumo5DplP80J6Q&_nc_zt=23&_nc_ht=scontent-otp1-1.xx&_nc_gid=2alJJykF3jwpQVktszskuQ&oh=00_Afez7jbOyNgFHXuzwXRk3CtO3jQOCr__Fb55dscmcmlsAQ&oe=692AF63F")
#response.raise_for_status()  # Check if the request was successful

#image_1 = Image.open(BytesIO(response.content)).convert("RGB")


# Prepare batched input with your image file
image_1 = Image.open("/kaggle/input/docsss/docs/business-cert-of-registration.jpg").convert("RGB")
#image_2 = Image.open("path/to/your/image_2.png").convert("RGB")


#prompt =  "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>extract data in json format<|im_end|>\n<|im_start|>assistant\n"
system = "You are a reliable data extraction engine. Your sole purpose is to analyze the provided image and extract information. Your entire response must be a single, valid JSON object, and you must include **no other text, explanations, or conversational filler**."

#system = "You are a reliable data extraction engine."

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

user_2 = '''
You are an expert document processing assistant. Analyze the provided image and determine if it is **clearly and unambiguously** one of the following supported document types:
- "passport"
- "national_id"
- "driver_license"

If the document does **not match any of these types with high confidence**, set `document_type` to `"unsupported"` and leave all other fields as `null`.

Only extract data if `document_type` is one of the three supported types. Otherwise, return:
{
  "document_type": "unsupported",
  "full_name": null,
  "document_number": null,
  ...
}
'''

#qwen3 vl
prompt =  f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{user}<|im_end|>\n<|im_start|>assistant\n"

    
prompt_input = {
        "prompt": prompt,
        "multi_modal_data": {"image": image_1}
    }


# 
tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3_5-2B-Instruct", trust_remote_code=True)
messages = [
        [{"role": "system", "content": f"{system}"},{"role": "user", "content": f"<image>\n{user}"}]
    ]
prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )[0]

prompt_input = {
        "prompt": prompt,
        "multi_modal_data": {"image": image_1}
    }


sampling_param = SamplingParams(
            temperature=0.7,              # Your temperature value
            #top_p=0.8,                    # Your top_p value
            #top_k=20,                     # Your top_k value
            #repetition_penalty=1.0,       # Your repetition_penalty value
            #presence_penalty=1.5,         # Your presence_penalty value
            max_tokens=8192,             # Your out_seq_length value
        )
# Generate output
model_outputs = llm.generate(prompt_input, sampling_param)

# Print output
for output in model_outputs:
    print(output.outputs[0].text)