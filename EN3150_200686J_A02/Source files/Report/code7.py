lasso_logistic_pipeline = Pipeline([('scaler', StandardScaler()), ('lasso_logistic', LogisticRegression(penalty='l1', solver='liblinear', multi_class='auto'))])


param_grid = {'lasso_logistic__C': np.logspace(-2, 2, 9)}
#Initialize GridSearchCV
grid_search = GridSearchCV(lasso_logistic_pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best value of C
best_C = grid_search.best_params_['lasso_logistic__C']
print('Best C:', best_C)

# Predict on the test set
y_pred = grid_search.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)