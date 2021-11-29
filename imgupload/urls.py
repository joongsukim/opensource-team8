from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import *
from . import views

urlpatterns = [
    path('image_upload/', views.dog_image_view),
    path('success/', views.success),
    path('dog_image/', views.display_dog_images),
]

# if settings.DEBUG:
#    urlpatterns += static(settings.MEDIA_URL,
#                          document_root=settings.MEDIA_ROOT)
