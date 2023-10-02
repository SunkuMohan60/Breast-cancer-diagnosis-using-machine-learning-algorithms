from django.conf import settings
# load and evaluate a saved model
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
# load model
path = settings.MEDIA_ROOT + "\\" + 'model.h5'
model = load_model(path)
dataset = settings.MEDIA_ROOT + "\\" + 'data.csv'

#Preprocess Data
def predictResp(data):
    for k, v in data.items():
        data[k] = float(v)
    data = pd.DataFrame([data])
    print(data)
    X = data.iloc[:, :].values
    #X = np.array(X[0])
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    dataScale = pd.read_csv(dataset)
    del dataScale['diagnosis']
    del dataScale['id']
    del dataScale['Unnamed: 32']
    #print(dataScale)
    dataScale = dataScale.iloc[:, :].values
    sc.fit(dataScale)
    sc.transform(dataScale)
    X=sc.transform(X)
    print(X)
    op=(model.predict(X))
    op = np.transpose(op)[0]
    op = list(map(lambda X: 'Benign(B)' if X<0.5 else 'Malignant(M)', op))
    print(op)
    return "The data entered is predicted as " + op[0]



