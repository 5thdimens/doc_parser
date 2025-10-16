


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
    "document_type": "string ('title_deed', 'lease_agreement', 'share_certificate', 'allotment_letter' or 'other')",
    "confidence_score": "float (on a scale of 0.0 to 1.0)"
}

Please analyze this image and generate the JSON. In case a field is missing, field value should be empty or null.
'''