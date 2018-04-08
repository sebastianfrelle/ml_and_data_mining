import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from project_2 import *



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

	plt.scatter([i for i in range(2)], errors[0:2])
	plt.scatter([i for i in range(2,4)], errors[2:4])
	plt.scatter(4, errors[4])
	plt.scatter([i for i in range(5,7)], errors[5:7])
	plt.scatter([i for i in range(7,9)], errors[7:9])
	plt.scatter([i for i in range(9,11)], errors[9:11])
	plt.scatter(11, errors[11])
	plt.scatter(12, errors[12])
	plt.scatter([i for i in range(13,18)], errors[13:18])
	plt.scatter([i for i in range(18,23)], errors[18:23])
	plt.scatter([i for i in range(23,27)], errors[23:27])
	plt.scatter([i for i in range(27,30)], errors[27:30])
	plt.scatter(30, errors[30])
	plt.scatter(31, errors[31])
	plt.scatter(32, errors[32])
	plt.scatter([i for i in range(33,35)], errors[33:35])
	plt.scatter([i for i in range(35,37)], errors[35:37])
	plt.scatter([i for i in range(37,39)], errors[37:39])
	plt.scatter([i for i in range(39,41)], errors[39:41])
	plt.scatter([i for i in range(41,43)], errors[41:43])
	plt.scatter([i for i in range(43,45)], errors[43:45])
	plt.scatter([i for i in range(45,47)], errors[45:47])
	plt.scatter([i for i in range(47,49)], errors[47:49])
	plt.scatter(50, errors[50])
	plt.scatter(51, errors[51])
	plt.scatter(52, errors[52])
	plt.scatter(53, errors[53])
	plt.scatter(54, errors[54])
	plt.scatter(55, errors[55])
	plt.gcf().subplots_adjust(bottom=0.25)
	plt.gca().xaxis.grid(True)
	plt.xticks(np.arange(len(errors)), (k_encoded_attr_names), rotation=75)
	plt.show()

	return forward_selected_indexes


X_train, X_test, y_train, y_test = train_test_split(X_k[:,forward_selection_cv(X_k, grades[:, 2])], grades[:, 2], test_size=0.2)
model = LinearRegression(normalize=True)
model.fit(X_train,y_train)
#print("predicted:", model.predict(X_test[i]), " actual: ", y_test[i] )
print(model.score(X_test,y_test))
print()