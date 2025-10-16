

class Document:
    def __init__(self, doc_type, prompt):
        self.doc_type = doc_type
        self.prompt = prompt

class DocumentCollection:
    def __init__(self, documents):
        for key, doc_data in documents.items():
            setattr(self, key, Document(doc_data['doc_type'], doc_data['prompt']))



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




documents_data = {
    "DT0002": {"doc_type": "kenya_national_id", "prompt": national_id},
    "DT0049": {"doc_type": "passport", "prompt": passport},
    "DT0081": {"doc_type": "military_id", "prompt": military_id},
    "DT0030": {"doc_type": "certificate_of_registration", "prompt": cert_of_reg},
    "DT0075": {"doc_type": "certificate_of_incorporation", "prompt": cert_of_incorp},
    "DT0074": {"doc_type": "kra_pin", "prompt": kra_pin},
    "DT0083": {"doc_type": "kra_pin", "prompt": kra_pin},
    "DT0076": {"doc_type": "title_deed", "prompt": owner_ship_cert},
    "DT0077": {"doc_type": "lease_agreement", "prompt": owner_ship_cert},
    "DT0078": {"doc_type": "shares_certificate", "prompt": owner_ship_cert},
    "DT0079": {"doc_type": "allotment_letter", "prompt": owner_ship_cert},
}
collection = DocumentCollection(documents_data)


