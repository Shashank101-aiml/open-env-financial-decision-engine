from pydantic import BaseModel

class Observation(BaseModel):
    input_text: str

class Action(BaseModel):
    decision: str  # approve / reject / block / alert
    risk: str      # low / medium / high
    reason: str

class Reward(BaseModel):
    score: float
