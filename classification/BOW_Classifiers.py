import pickle
from sklearn.cross_validation import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


dataPath = "../tweets/";

def printToFile(modelName, scores, condition_one, condition_two, output_path):

	labels  = {'depression' : 'D', 'ptsd' : 'P','control' : 'C'}

	output = '\n' + modelName + '_' + labels[condition_one] + 'v' + labels[condition_two] + "," + str(scores.mean()) + ", "+ str(scores.std())

	print output

	fd = open(output_path,'a')
	fd.write(output)
	fd.close()

def runClassification(modelName, model, data, condition_one, condition_two, output_path):

	print "running Classification with " + modelName

	scores = cross_val_score(model, data[0], data[1], cv=10)

	printToFile(modelName, scores, condition_one, condition_two, output_path)


if __name__ == '__main__':

	names = ["Nearest_Neighbors", "Linear_SVM", "RBF_SVM", "Decision_Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
         "Quadratic Discriminant Analysis"]

	classifiers = [
		KNeighborsClassifier(20),
		SVC(kernel="linear", C=0.025),
		DecisionTreeClassifier(max_depth=5),
		RandomForestClassifier(max_depth=5, n_estimators=100),
		AdaBoostClassifier()]

	for (c1,c2) in [("control", "depression"), ("ptsd", "depression"), ("control", "ptsd")]:

		
		max_features = 5000
		data = pickle.load(open('../feature_vectors/bag_of_words/'+ c1 + "_" + c2 + "_" +"10000" + '.p', "rb" ))

		for name, model in zip(names, classifiers):

			runClassification(name, model, data, c1, c2,"../results/bow.csv")