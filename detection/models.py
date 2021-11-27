from django.db import models

# Create your models here.
class Product(models.Model):
    productName=models.CharField(max_length=100)
    productImg=models.URLField(max_length=512)
    def __str__(self):
        return self.productName[:20]

class Company(models.Model):
    companyName=models.CharField(max_length=100)
    companyProducts=models.ManyToManyField(Product)
    def __str__(self):
        return self.companyName[:20]
