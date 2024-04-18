import numpy as np
import pandas as pd 
from scipy.stats import skew, kurtosis
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.stats import chi2_contingency, f_oneway,ttest_ind,stats
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Veri setini okuma
data = pd.read_csv("emails.csv")


# Veri setinin başlığını gösterme
data.head()

#describe the data
data.describe()

#info
data.info

#shape of the data
data.shape

#check for null values
data.isnull().sum()

# handle useless cols
data.drop(columns=['Email No.'],inplace=True)

# sample data
data.sample(5)

#count plot

plt.figure(figsize=(14,6))
sns.set_style('darkgrid')
sns.countplot(x='Prediction',data=data)
plt.title('Number of Spam and not')

#describe the columns
data.describe().columns


# import os
# import zipfile
# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, classification_report

# # Dosyanın tam yolu
# csv_file_path = "spam_ham_dataset.csv"



# # Veri setini yükleme
# data = pd.read_csv(csv_file_path)

# # Metin verilerini ve etiketleri ayırma
# X = data["Category"]
# y = data["Message"]

# # Metin verilerini vektörlere dönüştürme
# vectorizer = CountVectorizer()
# X_vectorized = vectorizer.fit_transform(X)

# # Eğitim ve test setlerini ayırma
# X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# # Naive Bayes sınıflandırma modelini oluşturma ve eğitme
# model = MultinomialNB()
# model.fit(X_train, y_train)

# # Modeli test seti üzerinde değerlendirme
# y_pred = model.predict(X_test)

# # Modelin performansını değerlendirme
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))
# # import pandas as pd

# # # Veri setini yükleme
# # csv_file_path = "C:\\Users\\emir7\\OneDrive\\Masaüstü\\spam.csv"  # Veri setinizin yolunu doğru olarak ayarlayın
# # data = pd.read_csv(csv_file_path)

# # # Metin verisi temizleme
# # data["Category"] = data["Category"].str.lower()  # Küçük harfe dönüştürme

# # # Veri setini yeniden kaydetme (isteğe bağlı)
# # data.to_csv("cleaned_spam.csv", index=False)  # Temizlenmiş veriyi yeni bir CSV dosyasına kaydetme (isteğe bağlı)
