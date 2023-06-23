import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from rest_framework.decorators import api_view
from rest_framework.response import Response
import os

@api_view(["GET"])
def predict_fuel_trainer(request):
    if request.method == 'GET':
        #Leer el archivo CSV media/fuel_consumption.csv
        data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'media', 'fuel_consumption.csv'))
        #Convertir la columna fechas de csv a tipo datetime
        data['Fecha'] = pd.to_datetime(data['Fecha'])

        #Obtener el número de días transcurridos desde la fecha inicial
        data['Dias'] = (data['Fecha'] - data['Fecha'].min()).dt.days

        # Obtener las características (X) y el consumo de combustible (y)
        X = data[['Dias', 'Peso', 'Potencia']].values
        y = data['Consumo'].values

        # Crear un modelo de regresión lineal
        model = LinearRegression()

        # Entrenar el modelo con los datos de entrenamiento
        model.fit(X, y)

        # Guardar el modelo entrenado en un archivo
        np.save(os.path.join(os.path.dirname(__file__), 'media','model.npy'), model.coef_)
        np.save(os.path.join(os.path.dirname(__file__), 'media', 'intercept.npy'), model.intercept_)

        # Crear una respuesta JSON con el modelo entrenado
        response = {'message': 'Modelo entrenado y guardado en el servidor'}

        # Devolver la respuesta JSON
        return Response(response)

@api_view(["GET"])
def predict_fuel_consumption(request):
    if request.method == 'GET':
        # # Obtener los datos de prueba desde la solicitud JSON
        dias = request.GET.get('dias')
        peso = request.GET.get('peso')
        potencia = request.GET.get('potencia')
        # Obtener las características de prueba

        X_test = np.array([[dias,peso,potencia]])
        
        # Realizar predicciones utilizando el modelo entrenado desde el archivo
        model = LinearRegression()
        model.coef_ = np.load(os.path.join(os.path.dirname(__file__), 'media','model.npy'))
        model.intercept_ = np.load(os.path.join(os.path.dirname(__file__), 'media', 'intercept.npy'))
        predictions = model.predict(X_test)
        
        #Crear una respuesta JSON con las predicciones multiplicado por dias
        prediction = predictions[0] * int(dias)
        prediction = round(prediction, 2)
        response = {'predictions': 'El consumo de combustible es de: ' + str(prediction) + ' Galones en '+ str(dias) + ' dias'}

        
        # Devolver la respuesta JSON
        return Response(response)