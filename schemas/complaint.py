from pydantic import BaseModel, Field

class ComplaintRequest(BaseModel):
    text: str= Field(..., min_length= 10, max_length= 1000)

class ComplaintResponse(BaseModel):
    category: str
    sentiment: str
    urgency: str
    risk: str
    reply: str