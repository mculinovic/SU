import sys

from bayes_classifier import *

def main():

	if (len(sys.argv) != 3):
		print("Wrong number of arguments!")
		print("Provide file names for train and test set!")
		sys.exit(-1)

	train_path = sys.argv[1]
	train_path = train_path[8:]
	test_path = sys.argv[2]
	test_path = test_path[8:]

	f_train_set = open(train_path,"r")
	f_test_set = open(test_path,"r")

	if f_train_set == None or f_test_set == None:
		print("File could not be opened")
		sys.exit(-1)


	train_set = readSetFromFile(f_train_set)

	test_set = readSetFromFile(f_test_set)

	bayes_classifier = BayesClassifier(train_set)

	bayes_classifier.classify(train_set, test_set)


	

def readSetFromFile(file):

	set_text = file.readlines()

	example_set = list()

	for line in set_text[1:]:
		data = line.split()

		input_example = data[:len(data) - 1]

		for i in range(len(input_example)):
			input_example[i] = float(input_example[i])

		output = data[len(data) - 1]
		example = (tuple(input_example), output)
		
		example_set.append(example)

	return example_set


if __name__ == "__main__": main()