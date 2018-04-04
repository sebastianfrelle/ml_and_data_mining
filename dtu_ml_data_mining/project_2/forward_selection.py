import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

def forward_selection_cv(X, y, cv_split=5):

	'''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	model = LinearRegression()
	model.fit(X_train,y_train)

	for i, _ in enumerate(X_test):
		print("predicted:", model.predict(X_test[i]), " . actual: ", y_test[i])

	scores = cross_val_score(model, X, y, cv=cv_split, scoring='neg_mean_squared_error')
	print("LinearRegression Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	'''
	for i in range(X.shape[1]):
		model = LinearRegression()

		scores = cross_val_score(model, X[:,i], y, cv=cv_split, scoring='neg_mean_squared_error')

		print("LinearRegression Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() ** 2))
		#for i, attribute in enumerate(X):
	#	print(attribute)