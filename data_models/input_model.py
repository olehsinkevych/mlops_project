from pydantic import BaseModel


class InputData(BaseModel):
    Gender: str
    Age: int
    DrivingLicense: int
    RegioniD: int
    # ....



