import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import dump, load

df = pd.read_csv("./data/train.csv")
df.drop(columns=["id","author"], axis=1, inplace=True)

vectorizer = TfidfVectorizer()
X = df["title"] + ' ' + df["text"]
y = df["label"].values

X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.2)

train_x_cv = vectorizer.fit_transform(X_train.apply(lambda x: np.str_(x)))
train_y =Y_train.astype('int')

test_x_cv = vectorizer.transform(x_test.apply(lambda x: np.str_(x)))

mn = MultinomialNB()
mn.fit(train_x_cv,train_y)

pred = mn.predict(test_x_cv)
# dump(mn, "./model/FakeNewsDetector.joblib")
mse = mean_squared_error(y_test,pred)
rmse = np.sqrt(mse)
print(rmse)