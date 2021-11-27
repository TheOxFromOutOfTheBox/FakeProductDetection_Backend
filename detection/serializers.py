from rest_framework import serializers

from .models import Company, Product

class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = '__all__'

class CompanySerializer(serializers.ModelSerializer):
    companyProducts=ProductSerializer(read_only=True,many=True)
    class Meta:
        model = Company
        fields = '__all__'