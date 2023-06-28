from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Definir la estructura del cuerpo de la solicitud
class Patient(BaseModel):
    gender: str
    age: int
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    hba1c_level: float
    blood_glucose_level: float

# Cargar el modelo entrenado
model = joblib.load("modelo_entrenado.joblib")

# Ruta para la predicción de diabetes
@app.post("/predict_diabetes")
def predict_diabetes(patient: Patient):
    # Convertir los datos del paciente en un arreglo numpy
    data = [[patient.gender, patient.age, patient.hypertension, patient.heart_disease,
             patient.smoking_history, patient.bmi, patient.hba1c_level, patient.blood_glucose_level]]
    
    # Realizar la predicción utilizando el modelo cargado
    prediction = model.predict(data)
    
    # Obtener el resultado de la predicción (0 para negativo, 1 para positivo)
    if prediction[0] == 0:
        result = "Negativo"
    else:
        result = "Positivo"
    
    return {"diabetes_prediction": result}
