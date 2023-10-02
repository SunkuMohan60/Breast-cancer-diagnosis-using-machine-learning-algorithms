from django import forms
from .models import DiagnosisUserRegistrationModel,PredictUserDataModel

class DiagnosisUserRegistrationForm(forms.ModelForm):
    name = forms.CharField(widget=forms.TextInput(attrs={'pattern':'[a-zA-Z]+'}), required=True,max_length=100)
    loginid = forms.CharField(widget=forms.TextInput(attrs={'pattern':'[a-zA-Z]+'}), required=True,max_length=100)
    password = forms.CharField(widget=forms.PasswordInput(attrs={'pattern':'(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}','title':'Must contain at least one number and one uppercase and lowercase letter, and at least 8 or more characters'}), required=True,max_length=100)
    mobile = forms.CharField(widget=forms.TextInput(attrs={'pattern':'[56789][0-9]{9}'}), required=True,max_length=100)
    email = forms.CharField(widget=forms.TextInput(attrs={'pattern':'[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$'}), required=True,max_length=100)
    locality = forms.CharField(widget=forms.TextInput(), required=True,max_length=100)
    address = forms.CharField(widget=forms.Textarea(attrs={'rows':4, 'cols': 22}), required=True,max_length=250)
    city = forms.CharField(widget=forms.TextInput(attrs={'autocomplete': 'off','pattern':'[A-Za-z ]+', 'title':'Enter Characters Only '}), required=True,max_length=100)
    state = forms.CharField(widget=forms.TextInput(attrs={'autocomplete': 'off','pattern':'[A-Za-z ]+', 'title':'Enter Characters Only '}), required=True,max_length=100)
    status = forms.CharField(widget=forms.HiddenInput(), initial='waiting' ,max_length=100)


    class Meta():
        model = DiagnosisUserRegistrationModel
        fields='__all__'

class PredictUserDataForm(forms.ModelForm):
    radius_mean= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    texture_mean= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    perimeter_mean= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    area_mean= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    smoothness_mean= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    compactness_mean= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    concavity_mean= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    concave_points_mean= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    symmetry_mean= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    fractal_dimension_mean= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    radius_se= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    texture_se= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    perimeter_se= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    area_se= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    smoothness_se= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    compactness_se= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    concavity_se= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    concave_points_se= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    symmetry_se= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    fractal_dimension_se= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    radius_worst= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    texture_worst= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    perimeter_worst= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    area_worst= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    smoothness_worst= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    compactness_worst= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    concavity_worst= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    concave_points_worst= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    symmetry_worst= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)	
    fractal_dimension_worst= forms.CharField(widget=forms.TextInput(), required=True,max_length=100)
    class Meta():
        model = PredictUserDataModel
        fields='__all__'