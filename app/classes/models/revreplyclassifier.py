from pydantic import BaseModel
from typing import Optional, Literal


class RevReplyClassifierRequest(BaseModel):
    email_thread: str
    prospect_name: Optional[str] = None
    sender_name: Optional[str] = None
    thread_id: Optional[str] = None
    user_id: Optional[int] = None
    persona_id: Optional[int] = None
    email_send_log_id: Optional[int] = None
    ooo_toggle: Optional[bool] = False  # When True, treat OOO with delegation as DELEGATED
    check_same_day_phone_request: Optional[bool] = False  # When True, run phone extractor and team alerter
    disqualification_criteria: Optional[str] = ""  # Qualification rules - disqualify if prospect mentions these factors

    #New filed to track days since last reply
    days_since_last_reply: Optional[int] = None
 
#Response model with tag support
class ClassificationTag(BaseModel):
    ###Sub Tags for quaified and discovery classifications
    tag: Optional[Literal["FOLLOW_UP", "CHECK_BACK_LATER", "REJECTED"]] = None
    reason: Optional[str] = None


# COMMENTED OUT: GEPA JSON Classifier request model - replaced by MIPRO-based classifier
# from pydantic import field_validator
# from typing import Dict, Any
# import json
#
# class GEPAJSONClassifierRequest(BaseModel):
#     """
#     Request model for the GEPA JSON Classifier (v2 API).
#
#     The email_thread_json must be a JSON string that will be parsed into a dictionary.
#
#     Expected structure after parsing:
#     {
#         "email_thread": {
#             "subject": "Re: Subject line",
#             "messages": [
#                 {
#                     "order": 1,
#                     "direction": "inbound" | "outbound",
#                     "from": "sender@email.com",
#                     "to": "recipient@email.com",
#                     "subject": "Subject",
#                     "datetime": "2026-02-04T10:00:00.000000Z",
#                     "body": "Email body text",
#                     "has_quoted_content": true | false
#                 }
#             ]
#         }
#     }
#     """
#     email_thread_json: str  # JSON string that will be parsed to dict
#     thread_id: Optional[str] = None
#     user_id: Optional[int] = None
#     persona_id: Optional[int] = None
#     ooo_toggle: Optional[bool] = False  # When True, treat OOO with delegation as DELEGATED
#     check_same_day_phone_request: Optional[bool] = False  # When True, run phone extractor and team alerter
#
#     @field_validator('email_thread_json', mode='before')
#     @classmethod
#     def validate_json_string(cls, v):
#         """
#         Validate that email_thread_json is a valid JSON string.
#         If it's already a dict (from tests), convert it to JSON string.
#         """
#         if isinstance(v, dict):
#             # Allow dicts for testing - convert to JSON string
#             return json.dumps(v)
#         elif isinstance(v, str):
#             # Validate it's parseable JSON
#             try:
#                 json.loads(v)
#                 return v
#             except json.JSONDecodeError as e:
#                 raise ValueError(f"Invalid JSON string for email_thread_json: {e}")
#         else:
#             raise ValueError(f"email_thread_json must be a JSON string, got {type(v)}")
