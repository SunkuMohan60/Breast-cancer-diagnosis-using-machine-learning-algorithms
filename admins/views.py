from django.shortcuts import render
from django.contrib import messages
from users.models import DiagnosisUserRegistrationModel
from.BreastCancerDiagnosis import DiagnosisModels
from django.conf import  settings
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('TkCairo')

# Create your views here.
def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')
        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})

def AdminHome(request):
    return render(request,'admins/AdminHome.html',{})

def DiagnosisUsers(request):
    data = DiagnosisUserRegistrationModel.objects.all()
    return render(request, 'admins/DiagnosisUsers.html',{'data':data})

def AdminActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        DiagnosisUserRegistrationModel.objects.filter(id=id).update(status=status)
        data = DiagnosisUserRegistrationModel.objects.all()
        return render(request,'admins/DiagnosisUsers.html',{'data':data})



def AdminClassificationReports(request):
    path = settings.MEDIA_ROOT + "\\" + 'data.csv'
    obj = DiagnosisModels()
    myDict = obj.classificationmodelExecutions(path=path)

    return render(request,"admins/AdminClassificationReport.html",{'myDict':myDict})


def AdminNeuralNetworks(request):
    path = settings.MEDIA_ROOT + "\\" + 'data.csv'
    obj = DiagnosisModels()
    dict_perc = obj.multiLayerPerceptron(path=path)
    dict_dnn = obj.DeepNeuralNetwork(path=path)
    dict_perc.update(dict_dnn)
    print("FInal Result ", dict_perc)
    return render(request,"admins/AdminDNNReport.html",{'myDict':dict_perc})