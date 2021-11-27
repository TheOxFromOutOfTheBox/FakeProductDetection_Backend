from django.urls import path

from detection.views import *
urlpatterns=[
    path('company-list/',ListCompany.as_view()),
    path('imgupload/',UploadImage.as_view()),
]