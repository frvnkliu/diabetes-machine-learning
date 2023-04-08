#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 18:17:18 2023

@author: frankliu
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('diabetes.csv')
print(df.head())
df.info()

# A mix of bar charts and histograms of each column
fig, axes = plt.subplots(5,5, figsize=(18,9))
fig.delaxes(axes[4][4])
fig.delaxes(axes[4][3])
fig.delaxes(axes[4][2])
for i, col in enumerate(df.columns):
    ax = axes[i//5, i%5]
    counts = df[col].value_counts()
    if(len(counts)<30):
        ax.bar(counts.index, counts.values)
        ax.set_xticks(counts.index)
        ax.set_xticklabels(counts.index)
    else:
        ax.hist(df[col], bins=10)
    ax.set_title(col)  
plt.tight_layout()
plt.show()

#create train and test data



#%% Model Comparison Tool
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn import metrics
from sklearn import model_selection

class model_tester:
    def __init__(self, df, target):
        train_data, test_data = model_selection.train_test_split(df, test_size = 0.2) 
        self.X_train = train_data.drop(target, axis =1)
        self.y_train = train_data[target]
        self.X_test = test_data.drop(target, axis =1)
        self.y_test = test_data[target]
        self.models = {}
    
    def test_model(self, model, model_type):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average="macro")
        recall =  recall_score(self.y_test, y_pred, average="macro")
        f1 =  f1_score(self.y_test, y_pred, average="macro")
        
        y_prob = model.predict_proba(self.X_test)
        roc_auc = roc_auc_score(self.y_test, y_prob[:,1], average="macro")
        print(f'{model_type} model performance')
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print(f'ROC-AUC: {roc_auc}')
        metrics.RocCurveDisplay.from_predictions(self.y_test, y_prob[:,1], name = model_type)
        self.models[model_type] = {'model': model,'AUC': roc_auc, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    
    def calc_scores(self, model_type):
        scores = []
        model = self.models[model_type]['model']
        #Note that everytime .fit() is called the parameters are reset so there's no influence from previously trained models
        ## Identifying the best predictor, see which variable when removed affects our model the most
        for i, var_name in enumerate(self.X_train.columns):    
            X_train_rem = self.X_train.drop(var_name, axis =1)
            X_test_rem = self.X_test.drop(var_name, axis =1)
            model.fit(X_train_rem, self.y_train)
            y_pred = model.predict(X_test_rem)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average="macro")
            recall =  recall_score(self.y_test, y_pred, average="macro")
            f1 =  f1_score(self.y_test, y_pred, average="macro")
            y_prob = model.predict_proba(X_test_rem)
            roc_auc = roc_auc_score(self.y_test, y_prob[:,1], average="macro")
            print(f'{model_type} model without {var_name} trained. ({i+1}/{len(self.X_train.columns)})')
            scores.append({'name': var_name, 'AUC': roc_auc, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1})
        self.scores = scores
        self.models[model_type]['scores'] = scores
    
    def find_best_predictors(self, model_type):
        print(f'===Lowest Performance Scores of each Metric of {model_type} model===')
        print(f'Lowest AUC: {sorted(self.scores, key = lambda x: x["AUC"])[0]["name"]}')
        print(f'Lowest Precision: {sorted(self.scores, key = lambda x: x["precision"])[0]["name"]}')
        print(f'Lowest Accuracy: {sorted(self.scores, key = lambda x: x["accuracy"])[0]["name"]}')
        print(f'Lowest Recall: {sorted(self.scores, key = lambda x: x["recall"])[0]["name"]}')
        print(f'Lowest F1: {sorted(self.scores, key = lambda x: x["f1"])[0]["name"]}')
#%%
# Initialize our Model Tester
tester = model_tester(df, 'Diabetes')
#%%
# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
LR_model = LogisticRegression(class_weight = 'balanced', max_iter = 3000)
tester.test_model(LR_model, "Logistic Regression")
#%%
tester.calc_scores("Logistic Regression")
tester.find_best_predictors("Logistic Regression")

#%%
# Support Vector Machine Model
from sklearn.svm import LinearSVC 
from sklearn.calibration import CalibratedClassifierCV
#CalibratedClassifierCV allows us to use .predict_proba
SVM_model = CalibratedClassifierCV(LinearSVC(dual = False, class_weight = 'balanced'))
tester.test_model(SVM_model, 'SVM')
#%%
tester.calc_scores("SVM")
tester.find_best_predictors("SVM")

#%%
# Individual Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(class_weight = 'balanced')
tester.test_model(tree_model, "Tree")

#%%
tester.calc_scores("Tree")
tester.find_best_predictors("Tree")

#%%
# Random Forest Model
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(class_weight = 'balanced', max_samples = 0.4, max_features = 0.3) 
tester.test_model(RF_model, "Random Forest")
#%%
tester.calc_scores("Random Forest")
tester.find_best_predictors("Random Forest")

#%%
from sklearn.ensemble import AdaBoostClassifier
AB_model = AdaBoostClassifier()
tester.test_model(AB_model, "Adaboost")
#%%
tester.calc_scores("Adaboost")
tester.find_best_predictors("Adaboost")

#%%
corr_matrix = df.corr()
#%% 
corr_matrix['Diabetes'].sort_values(key = lambda x: abs(x), ascending = False)
#%% 
# Some exploration Topics, does Mental Health correlate with Physical Health?
corr_matrix['MentalHealth'].sort_values(key = lambda x: abs(x), ascending = False)

#%%
tester2 = model_tester(df, 'Stroke')
#Use Logisitic Regression and find best predictor for Strokes
tester2.test_model(LR_model, "Logistic Regression")
#%%
tester2.calc_scores("Logistic Regression")
tester2.find_best_predictors("Logistic Regression")