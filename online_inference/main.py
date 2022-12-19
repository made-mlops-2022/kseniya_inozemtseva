from enum import Enum
from typing import List

from pydantic import BaseModel, Field, root_validator, ValidationError
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse, JSONResponse

from ms import app
from ms.functions import get_model_response, model_ready


model_name = "Cleveland"
version = "v1.0.0"


class Sex(int, Enum):
    female = 0
    male = 1


class ChestPain(int, Enum):
    typical_angina = 0
    atypical_angina = 1
    non_anginal_pain = 2
    asymptomatic = 3


class ECG(int, Enum):
    normal = 0
    ST_T_wave_abn = 1
    left_ventr_hypertr = 2


class Slope(int, Enum):
    upsloping = 0
    flat = 1
    downsloping = 2


class Thal(int, Enum):
    normal = 0
    fixed = 1
    reversable = 2


class Input(BaseModel):
    age: int = Field(..., ge=0, lt=130)
    sex: Sex
    cp: ChestPain
    trestbps: int = Field(..., gt=50, lt=400)  # resting blood pressure (in mm Hg on admission to the hospital)
    chol: int = Field(..., gt=50, lt=800)  # serum cholestoral in mg / dl
    fbs: bool  # (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
    restecg: ECG  # resting electrocardiographic results
    thalach: int = Field(..., gt=10, lt=600)  # maximum heart rate achieved
    exang: bool  # exercise induced angina (1 = yes; 0 = no)
    oldpeak: float = Field(..., ge=0, lt=14)  # ST depression induced by exercise relative to rest
    slope: Slope
    ca: int = Field(..., ge=0, le=3)  # number of major vessels (0-3) colored by flourosopy
    thal: Thal

    @root_validator(pre=True)
    def custom_check(cls, values):
        if values['age'] < 2 and values['cp'] != ChestPain.asymptomatic:
            # I guess, you cannot know symptoms of the chest pain in children who do not speak
            raise ValidationError
        return values

    class Config:
        use_enum_values = True
        schema_extra = {
            "age": 56,
            "sex": 1,
            "cp": 0,
            "trestbps": 160,
            "chol": 234,
            "fbs": 1,
            "restecg": 2,
            "thalach": 131,
            "exang": 0,
            "oldpeak": 0.1,
            "slope": 1,
            "ca": 1,
            "thal": 0
        }


class Output(BaseModel):
    label: List[str]
    prediction: List[int]


@app.get('/')
async def model_info():
    """Return model information, version, how to call"""
    return {
        "name": model_name,
        "version": version
    }


@app.get('/health')
async def service_health():
    """Return service health"""

    if model_ready():
        return 200
    else:
        return 400


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(exc.json(), status_code=400)


@app.post('/predict', response_model=Output)
async def model_predict(input: Input):
    """Predict with input"""
    response = get_model_response(input)
    return response
