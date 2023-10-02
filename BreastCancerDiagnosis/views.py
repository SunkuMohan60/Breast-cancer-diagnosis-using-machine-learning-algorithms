from django.shortcuts import render,HttpResponse
from users.forms import DiagnosisUserRegistrationForm
def index(request):
    return render(request,'index.html',{})


def logout(request):
    return render(request, 'index.html', {})


def UserLogin(request):
    return render(request, 'UserLogin.html', {})


def AdminLogin(request):
    return render(request, 'AdminLogin.html', {})


def UserRegister(request):
    form = DiagnosisUserRegistrationForm()
    return render(request, 'DiagnosisRegister.html', {'form':form})
