import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

df = pd.read_csv('data/processed.cleveland.data', header = None)


df.columns = ['age', 'sex', 'cp', 'restbp','chol','fbs','restecg','thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'hd'  ]
#print(len(df.loc[(df['ca'] == '?')))

df_no_missing = df.loc[(df['ca'] != '?') & (df['thal'] != '?')]

#print(len(df_no_missing))
x = df_no_missing.drop('hd', axis = 1).copy()
#print(x.head())
y = df_no_missing['hd'].copy()
#print(y.head())

x_encoding = pd.get_dummies(x, columns=['cp', 'restecg', 'slope', 'thal'])
#print(x_encoding.head())

#print(x['sex'].unique())
y_not_zero_index = y > 0
y[y_not_zero_index] = 1


x_train, x_test , y_train , y_test = train_test_split(x_encoding, y , random_state=42)
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt = clf_dt.fit(x_train, y_train)

#plt.figure(figsize=(15, 7.5))
#plot_tree(clf_dt, filled = True, rounded = True, class_names=['No HD', 'Yes HD'], feature_names=x_encoding.columns);
#plt.show()

y_pred = clf_dt.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
     confusion_matrix=cm,
     display_labels=["does not have hd", "does have hd "]
 )

disp.plot()
plt.show()

path = clf_dt.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

clf_dts = []

for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha= ccp_alpha)
    clf_dt.fit(x_train,y_train)
    clf_dts.append(clf_dt)

train_scores = [clf_dt.score(x_train,y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(x_test,y_test) for clf_dt in clf_dts]

#/fig, ax = plt.subplots()
# ax.set_xlabel("alphas")
# ax.set_ylabel("accuracy")
# ax.set_title("alphas vs accuracy graph for training and testing dataset")
# ax.plot(ccp_alphas, train_scores, marker = 'o', label = "train", drawstyle = "steps-post")
# ax.plot(ccp_alphas, test_scores, marker = 'o',label = 'test', drawstyle = "steps-post" )
# ax.legend
#plt.show()

# clf_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=0.016)
# scores = cross_val_score(clf_dt, x_train,y_train, cv = 5 )
# df  = pd.DataFrame(data={'tree': range(5), 'accuracy': scores})
# df.plot(x = 'tree', y = 'accuracy', marker= 'o', linestyle= '--' )
# plt.show()

alpha_loop_scores = []

for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha= ccp_alpha)
    scores = cross_val_score(clf_dt,x_train, y_train, cv = 5)
    alpha_loop_scores.append([ccp_alpha,np.mean(scores),np.std(scores)])

alpha_results = pd.DataFrame(alpha_loop_scores,columns = ['alpha', 'mean_accuracy', 'std'])

# alpha_results.plot(x = 'alpha', y = 'mean_accuracy',yerr = 'std', marker = 'o', linestyle= '--')
# plt.show()

ideal_ccp_alpha = alpha_results[(alpha_results['alpha'] > 0.014) & (alpha_results['alpha'] < 0.015)]
ideal_ccp_alpha = ideal_ccp_alpha['alpha'].iloc[0]  # first match

clf_dt_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha= ideal_ccp_alpha)
clf_dt_pruned = clf_dt_pruned.fit(x_train,y_train)

y_pred = clf_dt_pruned.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["does not have hd", "does have hd "]
)

#disp.plot()
#plt.show()

plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt_pruned, filled = True, rounded = True, class_names=['No HD', 'Yes HD'], feature_names=x_encoding.columns);
plt.show()