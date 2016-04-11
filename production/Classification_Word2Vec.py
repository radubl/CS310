import datetime, os, simplejson as json
import FeatureExtraction_Word2Vec as extractor
from sklearn.cross_validation import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


dataPath = "../tweets/";


# prints scores to ./output_path/modelName-condition_one-condition-two.json
def printToFile(modelName, scores, condition_one, condition_two, output_path):

	output = { datetime.datetime.now().strftime("%H:%M:%S on %B %d, %Y") : {"scores" : ", ".join(str(item) for item in scores), "mean" : str(scores.mean()), "std": str(scores.std())}}

	outputFileName = modelName + '-' + condition_one + '-' + condition_two + '.json'

	outputFileNamePath = output_path + outputFileName

	if outputFileName in os.listdir(output_path):

		data = None

		with open(outputFileNamePath) as f:
			data = json.load(f)

		data.append(output)

		with open(outputFileNamePath, 'w') as f:
			json.dump(data, f)

		print 'appending results to ' + outputFileNamePath
	else:
		with open(outputFileNamePath, 'w') as f:
	   		json.dump([output], f)
		print 'printing results to ' + outputFileNamePath


def runClassification(modelName, model, data, condition_one, condition_two, output_path):

	print "running Classification with " + modelName

	scores = cross_val_score(model, data[0], data[1], cv=10)

	printToFile(modelName, scores, condition_one, condition_two, output_path)


if __name__ == '__main__':

	names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
         "Quadratic Discriminant Analysis"]

	classifiers = [
		KNeighborsClassifier(3),
		SVC(kernel="linear", C=0.025),
		SVC(gamma=2, C=1),
		DecisionTreeClassifier(max_depth=5),
		RandomForestClassifier(max_depth=5, n_estimators=100),
		AdaBoostClassifier(),
		GaussianNB(),
		LinearDiscriminantAnalysis(),
		QuadraticDiscriminantAnalysis()]

	for (c1,c2) in [("control", "depression"), ("ptsd", "depression"), ("control", "ptsd")]:

		
		data = extractor.Word2Vec(c1, c2)

		for name, model in zip(names, classifiers):

			runClassification(name, model, data, c1, c2,"../results_prod/Word2Vec/")