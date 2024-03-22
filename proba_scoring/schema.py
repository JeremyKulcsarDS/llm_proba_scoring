from pydantic import BaseModel
from typing import Optional

class CheckerPrompt(BaseModel):
    task: Optional[str] = None
    header: Optional[str] = None
    content: Optional[str] = None
    to_return: Optional[str] = None
    example: Optional[str] = None