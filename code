import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

#import data
with open("./data.csv", "r") as f:
        df = pd.read_csv(f)
df.head()

#data overview
df.info()

df["Bankrupt?"] = df["Bankrupt?"].astype(bool)
df["Bankrupt?"].value_counts(normalize=True).plot(
    kind = "bar",
    xlabel= "Bankrupt",
    ylabel = "Frequency",
    title = "Class Balance"
)

corr = df.drop(columns="Bankrupt?").corr()
sns.heatmap(corr)

#split
target = "Bankrupt?"
X = df.drop(columns = target)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

under_sampler = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = under_sampler.fit_resample(X_train, y_train)
print("X_train_under shape:",X_train_under.shape)
X_train_under.head()

over_sampler = RandomOverSampler(random_state=42)
X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)
print("X_train_over shape:", X_train_over.shape)
X_train_over.head()

#building model
# Fit on `X_train`, `y_train`
model_reg = RandomForestClassifier(random_state=42)
model_reg.fit(X_train, y_train)

# Fit on `X_train_under`, `y_train_under`
model_under = RandomForestClassifier(random_state=42)
model_under.fit(X_train_under, y_train_under)

# Fit on `X_train_over`, `y_train_over`
model_over = RandomForestClassifier(random_state=42)
model_over.fit(X_train_over, y_train_over)

# Fit on `X_train_over`, `y_train_over` with DecisionTreeClassifier
model_dover = DecisionTreeClassifier(random_state=42)
model_dover.fit(X_train_over, y_train_over)

# Fit on `X_train_over`, `y_train_over` with DecisionTreeClassifier
model_dreg = DecisionTreeClassifier(random_state=42)
model_dreg.fit(X_train, y_train)

for m in [model_reg, model_under, model_over, model_dover, model_dreg]:
    acc_train = m.score(X_train, y_train)
    acc_test = m.score(X_test, y_test)

    print("Training Accuracy:", round(acc_train, 4))
    print("Test Accuracy:", round(acc_test, 4))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ConfusionMatrixDisplay.from_estimator(model_over, X_test, y_test).plot(ax=ax1)
ConfusionMatrixDisplay.from_estimator(model_dover, X_test, y_test).plot(ax=ax2)
ax1.set_title('Random Forest Classifier')
ax2.set_title('Decision Tree Classifier')
plt.show()

params = {
    "n_estimators": range(25,100,25),
    "max_depth":range(10,50,10)
}
model = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=params,
    cv=5,
    n_jobs = -1,
    verbose=1
)

model.fit(X_train_over, y_train_over)

params = {
    "max_depth":range(10,50,10)
}
model_d = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=params,
    cv=5,
    n_jobs = -1,
    verbose=1
)

model_d.fit(X_train_over, y_train_over)


#Report
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test).plot(ax=ax1)
ConfusionMatrixDisplay.from_estimator(model_d, X_test, y_test).plot(ax=ax2)
ax1.set_title('Random Forest Classifier')
ax2.set_title('Decision Tree Classifier')
plt.show()

# Generate predictions--Randomforest
y_test_pred = model.predict(X_test)
# Put predictions into Series with name "bankrupt", and same index as X_test
y_test_pred = pd.Series(y_test_pred, index=X_test.index, name="bankrupt")
print(classification_report(y_test, y_test_pred))

# Generate predictions--Decisiontree
y_test_pred = model_d.predict(X_test)
# Put predictions into Series with name "bankrupt", and same index as X_test
y_test_pred = pd.Series(y_test_pred, index=X_test.index, name="bankrupt")
print(classification_report(y_test, y_test_pred))

with open("./check.csv", encoding="utf-8") as f:
    df1 = pd.read_csv(f)
y_check = model_d.predict(df1.drop(columns="Bankrupt?"))
print(y_check)
