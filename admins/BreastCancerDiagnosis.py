## Import packages
import pandas as pd
from pandas import set_option
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import SelectKBest
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.cm as cm
from matplotlib import rcParams
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, LabelEncoder
from sklearn.pipeline import Pipeline
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

rcParams['xtick.major.pad'] = 1
rcParams['ytick.major.pad'] = 1
import matplotlib
#matplotlib.use('TkCairo')
matplotlib.use('Agg')

class DiagnosisModels:

    def classificationmodelExecutions(self, path):
        myResult ={}
        bcdf = pd.read_csv(path)
        print("Data path ", bcdf.shape)
        diagnosis_coder = {'M': 1, 'B': 0}
        bcdf.diagnosis = bcdf.diagnosis.map(diagnosis_coder)
        # Drop unecessary columns
        bcdf.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
        # Reorder columsn so diagnosis is right-most
        # First define a diagnosis series object
        diagnosis = bcdf.diagnosis
        # Then drop diagnosis from dataframe
        bcdf.drop('diagnosis', axis=1, inplace=True)
        # Then append diagnsis to end of dataframe
        bcdf['Diagnosis'] = diagnosis
        # Take a quick glimpse of the dataset
        print(bcdf.head())
        # Quick glimpse of tumor features (mean values) in relation to diagnosis
        bcdf.groupby('Diagnosis').mean()
        # For visual comparisons of differential diagnosis...
        # create to dataframes - one for benign, one for malignant tumor data
        bcdf_n = bcdf[bcdf['Diagnosis'] == 0]
        bcdf_y = bcdf[bcdf['Diagnosis'] == 1]
        # Create list of features related to mean tumor characteristics
        features_means = list(bcdf.columns[0:10])
        outcome_count = bcdf.Diagnosis.value_counts()
        outcome_count = pd.Series(outcome_count)
        outcome_count = pd.DataFrame(outcome_count)
        outcome_count.index = ['Benign', 'Malignant']
        outcome_count['Percent'] = 100 * outcome_count['Diagnosis'] / sum(outcome_count['Diagnosis'])
        outcome_count['Percent'] = outcome_count['Percent'].round().astype('int')
        print('The Perecentage of tumors classified as \'malignant\' in this data set is: {}'.format(
            100 * float(bcdf.Diagnosis.value_counts()[1]) / float((len(bcdf)))))
        print(
            '\nA good classifier should therefore outperform blind guessing knowing the proportions i.e. > 62% accuracy')
        print(outcome_count)

        # Split data into testing and training set. Use 80% for training
        X_train, X_test, y_train, y_test = train_test_split(bcdf.iloc[:, :-1], bcdf['Diagnosis'], train_size=.8)
        # The normalize features to account for feature scaling
        # Instantiate
        norm = Normalizer()
        # Fit
        norm.fit(X_train)
        # Transform both training and testing sets
        X_train_norm = norm.transform(X_train)
        X_test_norm = norm.transform(X_test)
        # Model testing
        # Define parameters for optimization using dictionaries {parameter name: parameter list}
        SVM_params = {'C': [0.001, 0.1, 10, 100], 'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}
        LR_params = {'C': [0.001, 0.1, 1, 10, 100]}
        LDA_params = {'n_components': [None, 1, 2, 3], 'solver': ['svd'], 'shrinkage': [None]}
        KNN_params = {'n_neighbors': [1, 5, 10, 20, 50], 'p': [2], 'metric': ['minkowski']}
        RF_params = {'n_estimators': [10, 50, 100]}
        DTC_params = {'criterion': ['entropy', 'gini'], 'max_depth': [10, 50, 100]}
        #MLP_param = {'max_iter':1000,'alpha':0.1, 'activation':'logistic', 'solver':'adam', 'random_state':42}

        # Append list of models with parameter dictionaries
        models_opt = []
        # models_opt.append(('LR', LogisticRegression(), LR_params))
        # models_opt.append(('LDA', LinearDiscriminantAnalysis(), LDA_params))
        # models_opt.append(('KNN', KNeighborsClassifier(), KNN_params))
        models_opt.append(('DTC', DecisionTreeClassifier(), DTC_params))
        models_opt.append(('RFC', RandomForestClassifier(), RF_params))
        models_opt.append(('SVM', SVC(), SVM_params))
        #models_opt.append(("MLP",MLPClassifier(),MLP_param))
        results = []
        names = []
        def estimator_function(parameter_dictionary, scoring='accuracy'):
            for name, model, params in models_opt:
                s = len(X_train_norm)
                print("Size is = ",s)
                # kfold = KFold(, n_splits=5, random_state=2, shuffle=True)
                # kfold = KFold(len(X_train_norm))
                kfold = KFold(n_splits=5, random_state=2, shuffle=True)
                model_grid = GridSearchCV(model, params)
                cv_results = cross_val_score(model_grid, X_train_norm, y_train, cv=kfold, scoring=scoring)
                results.append(cv_results)
                names.append(name)
                myResult.update({name:cv_results.mean()})
                msg = "Cross Validation Accuracy %s: Accarcy: %f SD: %f" % (name, cv_results.mean(), cv_results.std())
                print(msg)
        estimator_function(models_opt, scoring='accuracy')
        # Guassian Naive Bayes does not require optimization so we will run it separately without
        # gridsearch and append the performance results to the results and names lists.
        # Instantiate model
        # Define kfold - this was done above but not as a global variable
        # kfold = KFold(len(X_train_norm), n_folds=5, random_state=2, shuffle=True)
        kfold = KFold(n_splits=5, random_state=2, shuffle=True)
        # Run cross validation
        # Ensemble Voting
        from sklearn.ensemble import VotingClassifier
        # Create list for estimatators
        estimators = []
        # Create estimator object
        # model1 = LogisticRegression()
        # Append list with estimator name and object
        # estimators.append(("logistic", model1))
        model2 = DecisionTreeClassifier()
        estimators.append(("cart", model2))
        model3 = SVC()
        estimators.append(("svm", model3))
        model4 = KNeighborsClassifier()
        estimators.append(("KNN", model4))
        model5 = RandomForestClassifier()
        estimators.append(("RFC", model5))
        model7 = LinearDiscriminantAnalysis()
        estimators.append(("LDA", model7))
        voting = VotingClassifier(estimators)
        results_voting = cross_val_score(voting, X_train_norm, y_train, cv=kfold)
        results.append(results_voting)
        names.append('Voting')
        print('Accuracy: {} SD: {}'.format(results_voting.mean(), results_voting.std()))
        myResult.update({'RF-50':results_voting.mean()})
        # Visualize model accuracies for comparision - boxplots will be appropriate to visualize
        # data variation
        plt.boxplot(results, labels=names)
        plt.title('Breast Cancer Diagnosis Accuracy using Various Machine Learning Models')
        plt.ylabel('Model Accuracy %')
        sns.set_style("whitegrid")
        plt.ylim(0.8, 1)
        plt.show()
        # Instantiate a new LDA model
        '''lda_2 = LinearDiscriminantAnalysis()
        # Fit LDA model to the entire training data
        lda_2.fit(X_train_norm, y_train)
        # Test LDA model on test data
        lda_2_predicted = lda_2.predict(X_test_norm)
        # Use sklearn's 'accuracy_score' method to check model accuracy during testing
        print('Linear discriminant model analyis Accuracy is: {}'.format(accuracy_score(y_test, lda_2_predicted)))
        confusion_matrix_lda = pd.DataFrame(confusion_matrix(y_test, lda_2_predicted),
                                            index=['Actual Negative', 'Actual Positive'],
                                            columns=['Predicted Negative', 'Predicted Postive'])

        print('Linear discriminant Model Confusion Matrix')
        print(confusion_matrix_lda)
        print('Linear discriminant Model Classification Report')
        print(classification_report(y_test, lda_2_predicted))'''
        # Parameters
        RF_params = {'n_estimators': [10, 50, 100, 200]}
        # Instantiate RFC
        RFC_2 = RandomForestClassifier(random_state=42)
        # Instantiate gridsearch using RFC model and dictated parameters
        RFC_2_grid = GridSearchCV(RFC_2, RF_params)
        # Fit model to training data
        RFC_2_grid.fit(X_train_norm, y_train)
        # Print best parameters
        print('Optimized number of estimators: {}'.format(RFC_2_grid.best_params_.values()))
        # Train RFC on whole training set
        # Instantiate RFC with optimal parameters
        RFC_3 = RandomForestClassifier(n_estimators=50, random_state=42)
        # Fit RFC to training data
        RFC_3.fit(X_train_norm, y_train)
        # Predict on training data using fitted RFC
        # Evalaute RFC with test data
        RFC_3_predicted = RFC_3.predict(X_test_norm)
        print('Model accuracy on test data: {}'.format(accuracy_score(y_test, RFC_3_predicted)))
        myResult.update({'RF-100': format(accuracy_score(y_test, RFC_3_predicted))})
        # Create dataframe by zipping RFC feature importances and column names
        rfc_features = pd.DataFrame(zip(RFC_3.feature_importances_, bcdf.columns[:-1]),
                                    columns=['Importance', 'Features'])

        # Sort in descending order for easy organization and visualization
        rfc_features = rfc_features.sort_values(['Importance'], ascending=False)
        # Visualize RFC feature importances
        sns.barplot(x='Importance', y='Features', data=rfc_features, )
        plt.title('Feature Importance for Breast Cancer Diagnosis')
        sns.set_style("whitegrid")
        plt.show()
        rfc_features.Features[:5]

        # Instantiate PCA
        pca_var = PCA()

        # Fit PCA to training data
        pca_var.fit(X_train_norm)

        # Visualize explained variance with an increasing number of components
        plt.plot(pca_var.explained_variance_, 'bo-', markersize=8)
        plt.title("Explained Variance ")
        plt.ylabel('Explained Variance')
        plt.xlabel('Component Number')
        sns.set_style("whitegrid")
        plt.show()


        return myResult
    def multiLayerPerceptron(self,path):
        myDict = {}
        breast_cancer = pd.read_csv(path)
        breast_cancer.head()
        for field in breast_cancer.columns:
            amount = np.count_nonzero(breast_cancer[field] == 0)
            if amount > 0:
                print('Number of 0-entries for "{field_name}" feature: {amount}'.format(
                    field_name=field,
                    amount=amount
                ))
        # Features "id" and "Unnamed: 32" are not useful
        feature_names = breast_cancer.columns[2:-1]
        X = breast_cancer[feature_names]
        # "diagnosis" feature is our class which I wanna predict
        y = breast_cancer.diagnosis
        class_le = LabelEncoder()
        # M -> 1 and B -> 0
        y = class_le.fit_transform(breast_cancer.diagnosis.values)
        sns.heatmap(
            data=X.corr(),
            annot=True,
            fmt='.2f',
            cmap='RdYlGn'
        )
        fig = plt.gcf()
        fig.set_size_inches(20, 16)
        plt.show()
        pipe = Pipeline(steps=[
            ('preprocess', StandardScaler()),
            ('classification', MLPClassifier())
        ])
        random_state = 42
        mlp_activation = ['identity', 'logistic', 'tanh', 'relu']
        mlp_solver = ['lbfgs', 'sgd', 'adam']
        mlp_max_iter = range(1000, 10000, 1000)
        mlp_alpha = [1e-4, 1e-3, 0.01, 0.1, 1]
        preprocess = [Normalizer(), MinMaxScaler(), StandardScaler(), RobustScaler(), QuantileTransformer()]
        mlp_param_grid = [
            {
                'preprocess': preprocess,
                'classification__activation': mlp_activation,
                'classification__solver': mlp_solver,
                'classification__random_state': [random_state],
                'classification__max_iter': mlp_max_iter,
                'classification__alpha': mlp_alpha
            }
        ]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            random_state=42,
            test_size=0.32
        )

        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        scaler = StandardScaler()
        print('\nData preprocessing with {scaler}\n'.format(scaler=scaler))
        X_train_scaler = scaler.fit_transform(X_train)
        X_test_scaler = scaler.transform(X_test)
        mlp = MLPClassifier(
            max_iter=1000,
            alpha=0.1,
            activation='logistic',
            solver='adam',
            random_state=42
        )
        mlp.fit(X_train_scaler, y_train)
        mlp_predict = mlp.predict(X_test_scaler)
        mlp_predict_proba = mlp.predict_proba(X_test_scaler)[:, 1]
        print('MLP Accuracy: {:.2f}%'.format(accuracy_score(y_test, mlp_predict) * 100))

        print('MLP AUC: {:.2f}%'.format(roc_auc_score(y_test, mlp_predict_proba) * 100))
        print('MLP Classification report:\n\n', classification_report(y_test, mlp_predict))
        print('MLP Training set score: {:.2f}%'.format(mlp.score(X_train_scaler, y_train) * 100))
        print('MLP Testing set score: {:.2f}%'.format(mlp.score(X_test_scaler, y_test) * 100))

        myDict.update({'Perceptron':accuracy_score(y_test, mlp_predict) * 100})
        myDict.update({'Perceptron AUC': roc_auc_score(y_test, mlp_predict_proba) * 100})
        outcome_labels = sorted(breast_cancer.diagnosis.unique())
        # Confusion Matrix for MLPClassifier
        sns.heatmap(
            confusion_matrix(y_test, mlp_predict),
            annot=True,
            fmt="d",
            xticklabels=outcome_labels,
            yticklabels=outcome_labels
        )
        # ROC for MLPClassifier
        fpr, tpr, thresholds = roc_curve(y_test, mlp_predict_proba)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.rcParams['font.size'] = 12
        plt.title('ROC curve for MLPClassifier')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.grid(True)
        strat_k_fold = StratifiedKFold(
            n_splits=10,
            random_state=42
        )

        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        fe_score = cross_val_score(
            mlp,
            X_std,
            y,
            cv=strat_k_fold,
            scoring='f1'
        )
        print("MLP: F1 after 10-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
            fe_score.mean() * 100,
            fe_score.std() * 2
        ))
        return myDict

    def DeepNeuralNetwork(self,path):
        myDitc= {}
        data = pd.read_csv(path)
        del data['Unnamed: 32']
        X = data.iloc[:, 2:].values
        y = data.iloc[:, 1].values
        # Encoding categorical data
        from sklearn.preprocessing import LabelEncoder
        labelencoder_X_1 = LabelEncoder()
        y = labelencoder_X_1.fit_transform(y)
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        # Initialising the ANN
        classifier = Sequential()
        # Adding the input layer and the first hidden layer
        classifier.add(Dense(output_dim=16, init='uniform', activation='relu', input_dim=30))
        # Adding dropout to prevent overfitting
        classifier.add(Dropout(p=0.1))
        # Adding the second hidden layer
        classifier.add(Dense(output_dim=16, init='uniform', activation='relu'))
        # Adding dropout to prevent overfitting
        classifier.add(Dropout(p=0.1))
        # Adding the output layer
        classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
        # Compiling the DNN
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # Fitting the ANN to the Training set
        classifier.fit(X_train, y_train, batch_size=100, nb_epoch=150)
        # Long scroll ahead but worth
        # The batch size and number of epochs have been set using trial and error. Still looking for more efficient ways. Open to suggestions.

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        y_pred = (y_pred > 0.5)
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1]) / 57) * 100))
        myDitc.update({'DNN':((cm[0][0] + cm[1][1]) / 57)*100})
        sns.heatmap(cm, annot=True)
        plt.show()
        plt.savefig('h.png')
        return  myDitc




