#----- implementing RandomForestClassifier with no of trees from 1 to 50 and checking "mean", "median" and "dispersion" of resulting accuracy measures

Ntrees = 100
Trees = np.arange(Ntrees)+1
m = np.sqrt(TrainX.shape[1]).astype(int)  # most common is sqrt(m) ..... according to Breiman log m / log 2
cv = 10
clf_scores = np.zeros((Ntrees,cv))

#----- Training data
for tree in Trees:
    cols = (tree - 1)
    clf = ensemble.RandomForestClassifier(n_estimators=tree, max_features=m, random_state=0,n_jobs=-1)
    clf_scores[cols,:] = cross_val_score(clf, TrainX, np.where(TrainY >= delay_threshold,1,0), cv=cv, scoring = 'accuracy' , n_jobs = -1)

#----- Plotting results
plt.subplots(figsize=(10,8))
score_means = np.mean(clf_scores, axis=1)
score_std = np.std(clf_scores, axis=1)
score_medians = np.median(clf_scores, axis=1)

#----- FIXERR: BOXPLOT DEBUG ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#----- Finding Accouracy by no of trees
plt.subplots(figsize=(20,12))
plt.hlines(np.max(score_means),0, 110, linestyle='--', color='red', linewidth=2, alpha=0.7,  zorder = 2, label= 'Maximum of means')
plt.hlines(np.max(score_medians),0, 110, linestyle='--',color='blue', linewidth=2, alpha=0.7,  zorder = 2, label= 'Maximum of medians')
plt.scatter((np.argmax(score_means)+1),np.max(score_means), s=50, c='red', marker='o', zorder=3)
plt.scatter((np.argmax(score_medians)+1),np.max(score_medians), s=50, c='blue', marker='o', zorder=3)
plt.plot(Trees,score_means, zorder=3, c= 'k', label= 'Mean of accuracy scores')
plt.errorbar(Trees, score_means, yerr = 2*score_std,color='#31a354', alpha =0.7, capsize=20, elinewidth=4, linestyle="None", zorder = 1, label= 'SE of accuracy scores')
plt.annotate((np.argmax(score_medians)+1), 
    xy = ((np.argmax(score_medians)+1), np.max(score_medians)), 
    xytext = (5, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'blue', alpha = 0.5))
plt.annotate((np.argmax(score_means)+1), 
    xy = ((np.argmax(score_means)+1), np.max(score_means)), 
    xytext = (5, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'red', alpha = 0.5))
plt.title('Accuracy by choice of the number of trees')
plt.legend(frameon=False, loc='lower right')
plt.ylabel('Mean accuracy scores')
plt.xlabel('Number of trees in the Random Forest')
plt.xlim(0, 101)
plt.xticks(Trees,rotation=90)
remove_border()
plt.grid(True)
plt.savefig("17.png", transparent=True)
plt.show()

#----- calculating OutOfBounds Score
# The RandomForestClassifier is trained using bootstrap aggregation, where each new tree is fit from a bootstrap sample of the training observations z_i = (x_i, y_i). The out-of-bag (OOB) error is the average error for each z_i calculated using predictions from the trees that do not contain z_i in their respective bootstrap sample. This allows the RandomForestClassifier to be fit and validated whilst being trained
Ntrees2 = 100
Trees2 = np.arange(Ntrees2)+1
clf_OOBscores = np.zeros((Ntrees2))
for tree in Trees2:
    cols = (tree - 1)
    clf = ensemble.RandomForestClassifier(n_estimators=tree, oob_score=True, max_features=m, random_state=0, n_jobs=-1)
    clf.fit(TrainX, np.where(TrainY >= delay_threshold,1,0))
    clf_OOBscores[cols] = clf.oob_score_

#----- Plotting OOB score
plt.subplots(figsize=(20,12))
plt.hlines(np.max(clf_OOBscores),0, 101, linestyle='--',color='blue', linewidth=2, alpha=0.7,  zorder = 2, label= 'Maximum of OOB scores')
plt.scatter((np.argmax(clf_OOBscores)+1),np.max(clf_OOBscores), s=50, c='blue', marker='o', zorder=3)
plt.plot(Trees2,clf_OOBscores, zorder=3, c= 'k', label= 'OOB scores')
plt.annotate((np.argmax(clf_OOBscores)+1),
    xy = ((np.argmax(clf_OOBscores)+1), np.max(clf_OOBscores)),
    xytext = (5, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'blue', alpha = 0.5))
plt.title('OOB score by choice of the number of trees')
plt.legend(frameon=False, loc='lower right')
plt.ylabel('OOB scores')
plt.xlabel('Number of trees in the Random Forest')
plt.xlim(0, 101)
remove_border()
plt.grid(True)
plt.savefig("18.png", transparent=True)
plt.show()


#---- CONCLUSION: Optimum No Of Trees = 70

#---- So based on the optimal choice of the number of trees, we classify again, saving the confusion matrix information. We see that the accuracy improves compared to the LR model, rising to 71%. While we are better classifying the flights that are not delayed, we are also classifying more delayed flights as not-delayed (our false positive rate went up). The F1 score goes down.

#----- Create Random Forest classifier with 70 trees
clf_rf = RandomForestClassifier(n_estimators=70, n_jobs=-1)
clf_rf.fit(TrainX, np.where(TrainY >= delay_threshold,1,0))

#---- Evaluate on test set
pred = clf_rf.predict(TestX)

# print results and confusion matrix
cm_rf = confusion_matrix(np.where(TestY >= delay_threshold,1,0), pred)
print("Confusion matrix")
print(pd.DataFrame(cm_rf))
report_rf = precision_recall_fscore_support(list(np.where(TestY >= delay_threshold,1,0)), list(pred), average='binary')
print("\nprecision = %0.2f, recall = %0.2f, F1 = %0.2f, accuracy = %0.2f\n" % \
        (report_rf[0], report_rf[1], report_rf[2], accuracy_score(list(np.where(TestY >= delay_threshold,1,0)), list(pred))))
print(pd.DataFrame(cm_rf.astype(np.float64) / cm_rf.sum(axis=1)))
show_confusion_matrix(cm_rf)
# plt.savefig("23.png", transparent=True)
# plt.plot()

# Confusion matrix
#       0     1
# 0  6170  1000
# 1  1934   896

# precision = 0.47, recall = 0.32, F1 = 0.38, accuracy = 0.71

#           0         1
# 0  0.860530  0.353357
# 1  0.269735  0.316608

#----- The left-hand side of the matrix is darker than the right, showing that the RF classifier guesses 'not delayed' more often than delayed. We are happy with the improvement in accuracy, but wondering how to improve the precision and the F1 score.

#----- Finding Feature Importance
importances = pd.Series(clf_rf.feature_importances_, index=Xcols)
importances.sort_values(inplace=True)
plt.barh(np.arange(len(importances)), importances, alpha=0.7)
plt.yticks(np.arange(.5,len(importances),1), importances.index)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature importance')
remove_border()
plt.savefig("20.png", transparent=True)
plt.show()

#----- Cross Validating current model
RF_scores = cross_val_score(clf_rf, TrainX, np.where(TrainY >= delay_threshold,1,0), cv=10, scoring='accuracy')
RF_scores.min(), RF_scores.mean(), RF_scores.max()
# (0.68968968968968969, 0.70230057890057895, 0.72499999999999998)

#Hence, We can expect a best accuracy of "72%" with this model and choice of number of trees = 70




















