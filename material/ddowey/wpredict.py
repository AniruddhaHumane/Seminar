# Once again, we select Chicago and look at departure delays. We can use the first exploration above as a point of reference.
X_values = flt2008wtr[flt2008wtr[direction] == 'ORD'][Xcols]
Y_values = flt2008wtr[flt2008wtr[direction] == 'ORD'][Ycol]

X_values.head().T
#                 37    52    67    68    106
# Month             1     1     1     1     1
# DayOfWeek         4     4     4     4     4
# DepHr            17    19    20    20    18
# ArrHr            20    22    22    22    21
# UniqueCarrier    CO    AA    UA    AA    UA
# Dest            EWR   SAT   SFO   SFO   LAX
# CRSElapsedTime  159   180   288   275   270
# Distance        719  1041  1846  1846  1745
# Origin_Demand   152   170   153   153   169
# MC_DEP            V     V     V     V     V
# VISIBLE_DEP      10    10    10    10    10
# TEMP_DEP         17    19    21    21    18
# Dest_Demand      76    22    43    43   103
# MC_ARR            V     V     I     I     V
# VISIBLE_ARR      10    10     5     5    10
# TEMP_ARR         18    44    53    53    57

# This clearly indicates that Origin/Dest, UniqueCarrier, MC_ARR, MC_DEP needs Factorization 
# Factorise the qualitative variables
X_values['UniqueCarrier'] = pd.factorize(X_values['UniqueCarrier'])[0]
X_values['MC_DEP'] = pd.factorize(X_values['MC_DEP'])[0]
X_values['MC_ARR'] = pd.factorize(X_values['MC_ARR'])[0]

if direction == 'Origin':
    X_values['Dest'] = pd.factorize(X_values['Dest'])[0]
else:
    X_values['Origin'] = pd.factorize(X_values['Origin'])[0]
    
#Selecting 20000 Samples at random and splitting it into training and testing data
rows = np.random.choice(X_values.index.values, 20000)
sampled_X = X_values.ix[rows]
sampled_Y = Y_values.ix[rows]
TrainX, TestX, TrainY, TestY = train_test_split(sampled_X, sampled_Y, test_size=0.50, random_state=42)

# Create Random Forest classifier with 70 trees
clf_rf = RandomForestClassifier(n_estimators=70, n_jobs=-1)
clf_rf.fit(TrainX, np.where(TrainY >= delay_threshold,1,0))

# Evaluate on test set
pred = clf_rf.predict(TestX)
# print results
cm_rf = confusion_matrix(np.where(TestY >= delay_threshold,1,0), pred)
print("Confusion matrix")
print(pd.DataFrame(cm_rf))
report_rf = precision_recall_fscore_support(list(np.where(TestY >= delay_threshold,1,0)), list(pred), average='binary')
print ("\nprecision = %0.2f, recall = %0.2f, F1 = %0.2f, accuracy = %0.2f\n" % \
        (report_rf[0], report_rf[1], report_rf[2], accuracy_score(list(np.where(TestY >= delay_threshold,1,0)), list(pred))))
print(pd.DataFrame(cm_rf.astype(np.float64) / cm_rf.sum(axis=1)))
show_confusion_matrix(cm_rf)
# plt.savefig("25.png", transparent=True)
# plt.show()

# Confusion matrix
#       0     1
# 0  6130   683
# 1  1595  1592

# precision = 0.70, recall = 0.50, F1 = 0.58, accuracy = 0.77

#           0         1
# 0  0.899750  0.214308
# 1  0.234111  0.499529

#Accuracy now has improved to 77%, which is a good gain. Precision has also improved to 70% from 49% before and the F1 has also jumped to 58% from 39%. The new model seems to perform better.

#Finding Feature Importance
importances = pd.Series(clf_rf.feature_importances_, index=Xcols)
importances.sort_values(inplace=True)

plt.barh(np.arange(len(importances)), importances, alpha=0.7)
plt.yticks(np.arange(.5, len(importances),1), importances.index)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature importance')
remove_border()
plt.savefig("26.png", transparent=True)
plt.show()

# The most important features are the Demand and Temperature attributes at the Departure and Arrival airports, followed by the duration and distance. Departure airport attributes are more important generally than the arrival airport ones - something that should reverse in the analysis of arrival delays. Visibility and MC are low in imporance, as is Carrier.

# Calculating cross-validation scores, using 10-fold CV and 70 trees.
RF_scores = cross_val_score(clf_rf, TrainX, np.where(TrainY >= delay_threshold,1,0), cv=10, scoring='accuracy')
print (RF_scores.min(), RF_scores.mean(), RF_scores.max())
# 0.745 0.760801251901 0.783
#
#The model seems robust. The previous accuracy score 78% is among the highest.

#Plottung ROC Curve
test = np.where(TestY >= delay_threshold,1,0)
prob = clf_rf.predict_proba(TestX)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = metrics.roc_curve(test, prob[:,1])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])
print (metrics.roc_auc_score(test, prob[:,1]))
plt.figure()
plt.plot(fpr[1], tpr[1])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.grid(True)
plt.savefig("27.png", transparent=True)
plt.show()
# 0.796212306794

#--------------------------------------------------------------------------------------------------------------------------------------------------------

#Now Considering whole dataset of "ORD" and Training the model again i.e.
#TrainX, TestX, TrainY, TestY = train_test_split(X_values, Y_values, test_size=0.50, random_state=0)

# RESULTS
# Confusion matrix
#        0      1
# 0  72650   6930
# 1  18613  13419

# precision = 0.66, recall = 0.42, F1 = 0.51, accuracy = 0.77

#           0         1
# 0  0.912918  0.216346
# 1  0.233890  0.418925

# this indicted 91% of the time our model correctly predicted non-delayed flights however delayed flights were predicted only 41% of the time

# Calculating cross-validation scores, using 10-fold CV and 70 trees.
# 0.765185450636 0.770443820577 0.774572170952

#This model has more uniform accuracy around 76-77%, with improved f1 score of 51%

#PLOTTING ROC Curve
test = np.where(TestY >= delay_threshold,1,0)
prob = clf_rf.predict_proba(TestX)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = metrics.roc_curve(test, prob[:,1])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])


print (metrics.roc_auc_score(test, prob[:,1]))
plt.figure()
plt.plot(fpr[1], tpr[1])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()

# 0.765394435492

#A lot of the time, when we predict a flight that is not delayed, it is actually delayed. As a result, there can be additional features related to the causes of flight delay that are not yet discovered using our existing data sources

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Applying SMOTE sampling to "ORD" dataset

sm = SMOTE(random_state=42)
temp1 = np.where(Y_values>delay_threshold,1,0)
Counter(temp1)
# Counter({0: 160809, 1: 62414})

X_sampled, Y_sampled = sm.fit_sample(X_values,temp1)
Counter(Y_sampled)
# Counter({0: 160809, 1: 160809})

TrainX, TestX, TrainY, TestY = train_test_split(X_sampled,Y_sampled, test_size=0.50, random_state=42)
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
plt.savefig("28.png", transparent=True)
plt.show()

# Confusion matrix
#        0      1
# 0  71393   9039
# 1  15805  64572

# precision = 0.88, recall = 0.80, F1 = 0.84, accuracy = 0.85

#           0         1
# 0  0.887619  0.112458
# 1  0.196501  0.803364

#Calculating cross validation score

RF_scores = cross_val_score(clf_rf, TrainX, TrainY, cv=10, scoring='accuracy')
print (RF_scores.min(), RF_scores.mean(), RF_scores.max())

# 0.839800995025 0.842272504064 0.845345438716