import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from project_2 import *

from toolbox.Tools.toolbox_02450 import feature_selector_lr, bmplot



def forward_selection_cv(X, y, cv_split=5):

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	forward_selected_indexes = []
	overall_highest_R2 = -1.0

	errors = []

	for j in range(X_train.shape[1]):
		best_index = -1
		highest_R2 = -1.0
		for i in range(X_train.shape[1]):
			if i not in forward_selected_indexes:
				tmp_index = forward_selected_indexes.copy()
				tmp_index.append(i)
				model = LinearRegression(normalize=True)
				model.fit(X_train[:,tmp_index],y_train)

				score = model.score(X_test[:,tmp_index], y_test)

				if j == 0:
					errors.append(mean_squared_error(y_test, model.predict(X_test[:,tmp_index])))

				if highest_R2 < score:
					highest_R2 = score
					best_index = i

		if best_index > -1 and highest_R2 > overall_highest_R2:
			overall_highest_R2 = highest_R2
			forward_selected_indexes.append(best_index)

	plt.bar([i for i in range(len(errors))], errors)
	plt.xticks(np.arange(len(errors)), (k_encoded_attr_names), rotation=45)
	plt.show()

	return forward_selected_indexes


X_train, X_test, y_train, y_test = train_test_split(X_k[:,forward_selection_cv(X_k, grades[:, 2])], grades[:, 2], test_size=0.2)
model = LinearRegression(normalize=True)
model.fit(X_train,y_train)
#print("predicted:", model.predict(X_test[i]), " actual: ", y_test[i] )
print(model.score(X_test,y_test))
print()