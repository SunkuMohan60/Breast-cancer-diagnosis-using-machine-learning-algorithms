from django.db import models

# Create your models here.


# Create your models here.
class DiagnosisUserRegistrationModel(models.Model):
    name = models.CharField(max_length=100)
    loginid = models.CharField(unique=True, max_length=100)
    password = models.CharField(max_length=100)
    mobile = models.CharField(unique=True, max_length=100)
    email = models.CharField(unique=True, max_length=100)
    locality = models.CharField(max_length=100)
    address = models.CharField(max_length=1000)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    status = models.CharField(max_length=100)

    def __str__(self):
        return self.loginid

    class Meta:
        db_table = 'DiagnosisUsers'


class PredictUserDataModel(models.Model):
    radius_mean= models.DecimalField(decimal_places=8, max_digits=19)	
    texture_mean= models.DecimalField(decimal_places=8, max_digits=19)	
    perimeter_mean= models.DecimalField(decimal_places=8, max_digits=19)	
    area_mean= models.DecimalField(decimal_places=8, max_digits=19)	
    smoothness_mean= models.DecimalField(decimal_places=8, max_digits=19)	
    compactness_mean= models.DecimalField(decimal_places=8, max_digits=19)	
    concavity_mean= models.DecimalField(decimal_places=8, max_digits=19)	
    concave_points_mean= models.DecimalField(decimal_places=8, max_digits=19)	
    symmetry_mean= models.DecimalField(decimal_places=8, max_digits=19)	
    fractal_dimension_mean= models.DecimalField(decimal_places=8, max_digits=19)	
    radius_se= models.DecimalField(decimal_places=8, max_digits=19)	
    texture_se= models.DecimalField(decimal_places=8, max_digits=19)	
    perimeter_se= models.DecimalField(decimal_places=8, max_digits=19)	
    area_se= models.DecimalField(decimal_places=8, max_digits=19)	
    smoothness_se= models.DecimalField(decimal_places=8, max_digits=19)	
    compactness_se= models.DecimalField(decimal_places=8, max_digits=19)	
    concavity_se= models.DecimalField(decimal_places=8, max_digits=19)	
    concave_points_se= models.DecimalField(decimal_places=8, max_digits=19)	
    symmetry_se= models.DecimalField(decimal_places=8, max_digits=19)	
    fractal_dimension_se= models.DecimalField(decimal_places=8, max_digits=19)	
    radius_worst= models.DecimalField(decimal_places=8, max_digits=19)	
    texture_worst= models.DecimalField(decimal_places=8, max_digits=19)	
    perimeter_worst= models.DecimalField(decimal_places=8, max_digits=19)	
    area_worst= models.DecimalField(decimal_places=8, max_digits=19)	
    smoothness_worst= models.DecimalField(decimal_places=8, max_digits=19)	
    compactness_worst= models.DecimalField(decimal_places=8, max_digits=19)	
    concavity_worst= models.DecimalField(decimal_places=8, max_digits=19)	
    concave_points_worst= models.DecimalField(decimal_places=8, max_digits=19)	
    symmetry_worst= models.DecimalField(decimal_places=8, max_digits=19)	
    fractal_dimension_worst= models.DecimalField(decimal_places=8, max_digits=19)