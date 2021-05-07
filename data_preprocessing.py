import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


class Preprocessing:
    def __init__(self, file_path, num_col,cat_col ):
        self.path=file_path
        self.data = self.readData()
        self.num_col=num_col
        self.cat_col=cat_col
    def readData(self):
        data=pd.read_csv(file_path)
        return data
    def proveNullValues(self):
        nullValues=self.data.isna().values.sum()

        if nullValues>0:
            print("Anzahl null-werte:",nullValues)
        else:
            print("keine null-werte")
        return nullValues
    def getDataInfo(self, target_val):
        self.proveNullValues()
        print(self.data.head())
        print(self.data.info())
        print(self.data.describe())

    def proofCorrelation(self):
        print(self.data.corr()[target_val].sort_values(ascending=False))

        print('--------------Absolut----------------')
        print(self.data.corr()[target_val].abs().sort_values())

    def getUniqueValues(self):
        for i in cat_col:
            print(i, ':\n', self.data[i].value_counts())
            print(i, ':\n', self.data[i].value_counts() / len(self.data))
            print('-----------------------------------------------')

    def dataPlotting(self, target_val):
        plt.figure(figsize=(10, 10))
        sns.boxplot(x="variable", y="value", data=pd.melt(self.data[num_col]))
        plt.show()
        fig, axis = plt.subplots(7, 2, figsize=(10, 15))
        self.data.hist(ax=axis)
        plt.show()
        plt.figure(figsize=(15, 10))
        sns.heatmap(self.data.corr(), cmap='autumn', annot=True)
        plt.show()#######
        #strlst=[target_val]
        #sns.pairplot(self.data[cat_col+strlst], hue=target_val, corner=True, diag_kind='hist')
    def cleanData(self, col1,val1,col2,val2):
        for x in self.data.index:
            if self.data.loc[x, col1] == val1:
                self.data.drop(x, inplace=True)
            elif self.data.loc[x, col2] == val2:
                self.data.drop(x, inplace=True)




class TransformData:
    def __init__(self, cleanedDataset,X_col, y_col):
        self.data=cleanedDataset
        self.X_col=X_col
        self.y_col=y_col
        #self.X_train
        #self.X_test
        #self.y_train
        #self.y_test
        #self.X_transformed
    def stratifyData(self, test_size, random_state_num):
        X = self.data[self.X_col]
        y = self.data[self.y_col]
        split = StratifiedShuffleSplit(test_size=test_size, random_state=random_state_num)
        for train_index, test_index in split.split(X,y):

            self.X_train, self.X_test = X.iloc[train_index], X.iloc[test_index]
            self.y_train, self.y_test =y.iloc[train_index], y.iloc[test_index]
        return self.X_train, self.X_test,self.y_train, self.y_test
    def columnTransformer(self,columns_category,columns_num,strategy,remainder):

        if strategy=="standard_scaler":
            res = [(val, StandardScaler(), [val]) for val in columns_num]
            self.cf_scaled=ColumnTransformer(res,remainder=remainder)
            return self.cf_scaled
        elif strategy=="standard_and_one_hot":
            scaler=[( val, StandardScaler(), [val]) for val in columns_num ]
            oneHot=[( val, OneHotEncoder(), [val]) for val in columns_category]
            self.cf_scaled=ColumnTransformer(scaler+oneHot,remainder=remainder)
            return self.cf_scaled
    def fit(self,X):
        self.X_train_fitted= self.cf_scaled.fit(X)

    def transform(self, X):
        self.X_transformed = self.cf_scaled.transform(X)
        return self.X_transformed








class MyLogisticRegression:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test

    def createModel(self,balanced,class_weight_dict,dict):
        if balanced==True:
            self.model=LogisticRegression(class_weight = "balanced")
        elif dict==True:
            self.model=LogisticRegression(class_weight = class_weight_dict)
        else:
            self.model = LogisticRegression()
    def crossValidation(self):
        self.score_cv = cross_val_score(self.model, self.X_train, self.y_train)
        print("score_cv:",self.score_cv,"\nscore_cv_mean:", self.score_cv.mean(),"\nscore_cv_std:", self.score_cv.std())

    def fit(self):
       self.model.fit(self.X_train,self.y_train)
       #print(self.model.score(X_train, y_train))
    def getScore(self,X,y,threshold,tune):
        score=self.model.score(X,y)
        print("Accurance:",score)
        y_test_pred = self.model.predict(X)
        print("recall:", recall_score(y, y_test_pred))
        print("precision:", precision_score(y, y_test_pred))
        matrix=confusion_matrix(y, y_test_pred)
        print(matrix)
        plot_confusion_matrix(self.model, X, y)
        plt.show()
        plot_confusion_matrix(self.model, X, y, normalize="all")
        plt.show()
        print("---------ROC Kurve-------------")
        fpr, tpr, thresholds = roc_curve(y, y_test_pred)
        # print(fpr, tpr, thresholds)
        plt.plot(fpr, tpr)
        plt.xlabel("fpr")
        plt.ylabel("tpr")
        # plt.legend(loc='upper right')
        plt.show()
        # die Fläche unter der Kurve berechnen
        roc_area = roc_auc_score(y, y_test_pred)
        print("Fläsche unter der ROC Kurve:", roc_area)
        if tune==True:
            print("----------------------")
            print("Threshold has changed")
            print("----------------------")
            y_test_pred = self.model.predict_proba(X_test)[:, 1] >= threshold
            print("recall:", recall_score(y, y_test_pred))
            print("precision:", precision_score(y, y_test_pred))
            matrix = confusion_matrix(y, y_test_pred)
            print(matrix)
            print("---------ROC Kurve-------------")
            fpr, tpr, thresholds = roc_curve(y, y_test_pred)
            # print(fpr, tpr, thresholds)
            plt.plot(fpr, tpr)
            plt.xlabel("fpr")
            plt.ylabel("tpr")
            # plt.legend(loc='upper right')
            plt.show()
            # die Fläche unter der Kurve berechnen
            roc_area = roc_auc_score(y, y_test_pred)
            print("Fläsche unter der ROC Kurve:", roc_area)

        return score





file_path="./heart.csv" #file path

cl1="ca" #column1 for data cleaning
val1=4 #value1 for data cleaning
cl2="thal" #column2 for data cleaning
val2=0 #value2 for data cleaning
num_col=['age','trestbps','chol','thalach','oldpeak'] #numeric columns
cat_col=['sex','cp','fbs','restecg','exang','slope','ca','thal'] #categorial columns
target_val="target"  #target value

test= Preprocessing("./heart.csv",num_col,cat_col)
test.getDataInfo(target_val)
test.dataPlotting(target_val)
test.getUniqueValues()
test.cleanData(cl1,val1,cl2,val2)
test.getUniqueValues()
clean_data=test.data
print(clean_data.shape)
########




######Testing
test_size=0.2
random_state_num=42
X_col=['age','trestbps','chol','thalach','oldpeak','sex','cp','fbs','restecg','exang','slope','ca','thal']
y_col = 'target'
######

test_transform=TransformData(clean_data,X_col,y_col)
X_train, X_test, y_train, y_test=test_transform.stratifyData(test_size,random_state_num)
print(X_train)
print(y_train)
num_columns=['age','trestbps','chol','thalach','oldpeak']
cat_col=['cp','restecg','slope','ca','thal']

test_transform.columnTransformer(cat_col,num_columns,"standard_and_one_hot","passthrough")
test_transform.fit(X_train)
X_train=test_transform.transform(X_train) #X-train transformed
print(X_train.shape)
#test_transform.columnTransformer(cat_col,num_columns,"standard_scaler","passthrough")
X_test=test_transform.transform(X_test) #X-train transformed
print(X_test.shape)
regression=MyLogisticRegression(X_train, X_test,y_train, y_test)
weights={0:1, 1:1}
regression.createModel(False,weights,False)
regression.crossValidation()
regression.fit()
regression.getScore(X_test, y_test,0.4 ,True)





