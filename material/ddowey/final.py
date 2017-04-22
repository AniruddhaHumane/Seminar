#Using Whole dataset

X_values = flt2008wtr[Xcols]
Y_values = flt2008wtr[Ycol]

X_values['UniqueCarrier'] = pd.factorize(X_values['UniqueCarrier'])[0]
X_values['MC_DEP'] = pd.factorize(X_values['MC_DEP'])[0]
X_values['MC_ARR'] = pd.factorize(X_values['MC_ARR'])[0]

if direction == 'Origin':
    X_values['Dest'] = pd.factorize(X_values['Dest'])[0]
else:
    X_values['Origin'] = pd.factorize(X_values['Origin'])[0]

sm = SMOTE(random_state=42)
temp1 = np.where(Y_values>delay_threshold,1,0)
Counter(temp1)
# Counter({0: 3908966, 1: 914401})

X_sampled, Y_sampled = sm.fit_sample(X_values,temp1)
Counter(Y_sampled)
# Counter({0: 3908966, 1: 3908966})
Counter(Y_sampled)[0]+Counter(Y_sampled)[1]
# 7817932

#Splitting data
TrainX, TestX, TrainY, TestY = train_test_split(X_sampled,Y_sampled, test_size=0.50, random_state=0)

#Training data
clf_rf = RandomForestClassifier(n_estimators=70, n_jobs=-1)
clf_rf.fit(TrainX, TrainY)

# Evaluate on test set
pred = clf_rf.predict(TestX)

# print results
cm_rf = confusion_matrix(TestY, pred)
print("Confusion matrix")
print(pd.DataFrame(cm_rf))
report_rf = precision_recall_fscore_support(list(TestY), list(pred), average='binary')
print ("\nprecision = %0.2f, recall = %0.2f, F1 = %0.2f, accuracy = %0.2f\n" % \
        (report_rf[0], report_rf[1], report_rf[2], accuracy_score(list(TestY), list(pred))))
print(pd.DataFrame(cm_rf.astype(np.float64) / cm_rf.sum(axis=1)))
show_confusion_matrix(cm_rf)

# Confusion matrix
#          0        1
# 0  1858155    95399
# 1   326796  1628616

# precision = 0.94, recall = 0.83, F1 = 0.89, accuracy = 0.89

#           0         1
# 0  0.951166  0.048787
# 1  0.167283  0.832876

#Calculating ROC
prob = clf_rf.predict_proba(TestX)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = metrics.roc_curve(TestY, prob[:,1])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])
print (metrics.roc_auc_score(TestY, prob[:,1]))
plt.figure()
plt.plot(fpr[1], tpr[1])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()

# 0.946695065577

#Calculating cross validation score

RF_scores = cross_val_score(clf_rf, TrainX, TrainY, cv=10, scoring='accuracy')
print (RF_scores.min(), RF_scores.mean(), RF_scores.max())