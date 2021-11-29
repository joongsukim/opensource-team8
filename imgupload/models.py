from django.db import models

# Create your models here.
class Photo(models.Model):
    name = models.CharField(max_length=50)
    dog_image = models.ImageField(upload_to='images/')

