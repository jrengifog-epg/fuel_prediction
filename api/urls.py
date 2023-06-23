from django.urls import path, include
from .router import router
from .views import *

urlpatterns = [
    path('predict/', predict_fuel_consumption, name='predict'),
    path('train/', predict_fuel_trainer, name='train'),
]