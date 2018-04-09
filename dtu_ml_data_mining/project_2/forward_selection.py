import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from project_2 import *



def forward_selection_cv(X, y, cv_split=5):
	reduced_index = [0,2,4,5,7,9,11,12,13,14,15,16,18,19,20,21,23,24,25,27,28,30,31,32,33,35,37,39,41,43,45,47,49,50,51,52,53,54,55]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	forward_selected_indexes = []
	overall_highest_R2 = -1.0

	all_errors = []
	reduced_errors = []
	step_errors = []

	for j in range(X_train.shape[1]):
		best_index = -1
		highest_R2 = -1.0

		for i in range(X_train.shape[1]):
			if i not in forward_selected_indexes and i in reduced_index:
				tmp_index = forward_selected_indexes.copy()
				tmp_index.append(i)
				model = LinearRegression(normalize=True)

				model.fit(X_train[:,tmp_index],y_train)
				score = model.score(X_test[:,tmp_index], y_test)

				if j == 0:
					reduced_errors.append(score)
					all_errors.append(score)

				if highest_R2 < score:
					highest_R2 = score
					best_index = i
			else:
				if j == 0:
					model = LinearRegression(normalize=True)
					model.fit(X_train[:,i],y_train)
					all_errors.append(score)


		if best_index > -1 and highest_R2 > overall_highest_R2 and j in reduced_index:
			#print(best_index)
			step_errors.append(highest_R2)
			overall_highest_R2 = highest_R2
			forward_selected_indexes.append(best_index)


		
	return forward_selected_indexes, step_errors, reduced_errors, all_errors, model

reduced_index = [0,2,4,5,7,9,11,12,13,14,15,16,18,19,20,21,23,24,25,27,28,30,31,32,33,35,37,39,41,43,45,47,49,50,51,52,53,54,55]
X_k_reduced = X_k[:,reduced_index]

best_error = 0.0

forward_selected_indexes, step_errors, reduced_errors, all_errors, model = forward_selection_cv(X_k, grades[:, 2])
'''
for i in range(10):
	X_train, X_test, y_train, y_test = train_test_split(X_k[:,forward_selected_indexes], grades[:, 2], test_size=0.2)
	model = LinearRegression(normalize=True)
	model.fit(X_train,y_train)
	R2 = model.score(X_test, y_test)

	if R2 > best_error:
		best_error = R2
	#print("predicted:", model.predict(X_test[i]), " actual: ", y_test[i] )
		
'''
print(forward_selected_indexes)
print(step_errors)

f1 = plt.figure()
plt.scatter([i for i in range(len(reduced_errors))], reduced_errors)
plt.gcf().subplots_adjust(bottom=0.25)
plt.gca().xaxis.grid(True)
plt.xticks(np.arange(len(reduced_errors)), (np.array(k_encoded_attr_names)[reduced_index]), rotation=75)
plt.ylabel('Coefficient of determination, R2')
plt.show()
#plt.savefig('./reduced_forward_selection.eps', format='eps',dpi=1000, bbox_inches='tight')

plt.scatter([i for i in range(2)], all_errors[0:2])
plt.scatter([i for i in range(2,4)], all_errors[2:4])
plt.scatter(4, all_errors[4])
plt.scatter([i for i in range(5,7)], all_errors[5:7])
plt.scatter([i for i in range(7,9)], all_errors[7:9])
plt.scatter([i for i in range(9,11)], all_errors[9:11])
plt.scatter(11, all_errors[11])
plt.scatter(12, all_errors[12])
plt.scatter([i for i in range(13,18)], all_errors[13:18])
plt.scatter([i for i in range(18,23)], all_errors[18:23])
plt.scatter([i for i in range(23,27)], all_errors[23:27])
plt.scatter([i for i in range(27,30)], all_errors[27:30])
plt.scatter(30, all_errors[30])
plt.scatter(31, all_errors[31])
plt.scatter(32, all_errors[32])
plt.scatter([i for i in range(33,35)], all_errors[33:35])
plt.scatter([i for i in range(35,37)], all_errors[35:37])
plt.scatter([i for i in range(37,39)], all_errors[37:39])
plt.scatter([i for i in range(39,41)], all_errors[39:41])
plt.scatter([i for i in range(41,43)], all_errors[41:43])
plt.scatter([i for i in range(43,45)], all_errors[43:45])
plt.scatter([i for i in range(45,47)], all_errors[45:47])
plt.scatter([i for i in range(47,49)], all_errors[47:49])
plt.scatter(49, all_errors[49])
plt.scatter(50, all_errors[50])
plt.scatter(51, all_errors[51])
plt.scatter(52, all_errors[52])
plt.scatter(53, all_errors[53])
plt.scatter(54, all_errors[54])
plt.scatter(55, all_errors[55])
plt.gcf().subplots_adjust(bottom=0.25)
plt.gca().xaxis.grid(True)
plt.xticks(np.arange(len(all_errors)), (k_encoded_attr_names), rotation=75)
plt.ylabel('Coefficient of determination, R2')
plt.show()
	#print(model.score(X_test,y_test))


print()