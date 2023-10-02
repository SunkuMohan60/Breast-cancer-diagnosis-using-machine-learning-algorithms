
# load and evaluate a saved model
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
# load model
model = load_model('model.h5')

#Preprocess Data
def startPreprocess(path):
    data = pd.read_csv(path)
    del data['Unnamed: 31']
    del data['id']
    print(data)
    X = data.iloc[:, :].values
    from sklearn.preprocessing import StandardScaler   
    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X

testData = startPreprocess("./media/test.csv")

op=(model.predict(testData))
op = np.transpose(op)[0]
op = list(map(lambda x: 'B' if x<0.5 else 'M', op))
print(op)

