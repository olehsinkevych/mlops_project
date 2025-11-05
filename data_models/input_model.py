from pydantic import BaseModel


class InputData(BaseModel):
    Gender: str
    Age: int
    DrivingLicense: int
    RegionCode: int
    Previously_Insured: int
    Vehicle_Age: str
    Vehicle_Damage: int
    Annual_Premium: float
    Policy_Sales_Channel: int
    Vintage: int
