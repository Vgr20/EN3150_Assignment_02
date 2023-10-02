from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score,f1_score
# Make predictions on the test set
y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
# Print the metrics
print('Accuracy:', accuracy)
print('Confusion Matrix:\n', confusion_matrix)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)

# Plot the confusion matrix
import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False)
plt.title('Confusion Matrix')
plt.show()