# -*- coding: utf-8 -*-
import sys
import numpy
import math
import codecs

def main():

	if (len(sys.argv) != 6):
		print("Wrong number of arguments!")
		print("Provide file names!")
		sys.exit(-1)

	train_set_path = sys.argv[2]
	validation_set_path = sys.argv[3]
	test_set_path = sys.argv[4]
	dictionary_path = sys.argv[1]
	OUTPUT_FOLDER = sys.argv[5]


	f_train_set = open(train_set_path,"r")
	f_validation_set = open(validation_set_path, "r")
	f_test_set = open(test_set_path, "r")
	f_dictionary = open(dictionary_path, "r")

	dictionary = readDictFromFile(f_dictionary)

	N = len(dictionary)

	train_set = readSetFromFile(f_train_set, N)
	validation_set = readSetFromFile(f_validation_set, N)
	test_set = readSetFromFile(f_test_set, N)

	#GRADIENT DESCENT lambda = 0
	(w0, w) = batch_gradient_descent(train_set, 0.0, N)
	cee = cross_entropy_error(0.0, train_set, w0, w)
	ee = generalization_error(w0, w, train_set)

	#OUTPUT WEIGHTS 1
	fout = open(OUTPUT_FOLDER + "tezine1.dat", "w")

	output = ""
	output += str.format("{0:.2f}", w0)
	output += "\n"

	for i in xrange(N):
		output += str.format("{0:.2f}", w[i]) + "\n"
	output += "EE: " + str.format("{0:.2f}", ee) + "\n"
	output += "CEE: " + str.format("{0:.2f}", cee) + "\n"

	fout.writelines(output)
	fout.close()

	#CROSS VALIDATION
	optimization = cross_validation(train_set, validation_set, N)

	#OUTPUT OPTIMIZATION
	fout = codecs.open(OUTPUT_FOLDER + "optimizacija.dat", "w", "utf-8")
	output = ""
	optimal_l = optimization[0][0]
	min_ge = optimization[0][1]
	for (l, ge) in optimization:
		# output += u"\u03BB" + " = " + str.format("{0:.2f}", l) + ", " + str.format("{0:.2f}", ge) + "\n"
		output += u"\u03BB" + " = " + str(l) + ", " + str.format("{0:.2f}", ge) + "\n"
		if ge <= min_ge:
			min_ge = ge
			optimal_l = l
	# output += "optimalno: " + u"\u03BB" + " = " + str.format("{0:.2f}", optimal_l) + "\n"
	output += "optimalno: " + u"\u03BB" + " = " + str(optimal_l) + "\n"
	fout.writelines(output)
	fout.close()

	#GRADIENT DESCENT train_set + validation_set
	example_set = train_set + validation_set
	(w0, w) = batch_gradient_descent(example_set, optimal_l, N)
	cee = cross_entropy_error(0, example_set, w0, w)
	ee = generalization_error(w0, w, example_set)

	#OUTPUT WEIGHTS 2
	fout = open(OUTPUT_FOLDER + "tezine2.dat", "w")

	output = ""
	output += str.format("{0:.2f}", w0)
	output += "\n"

	for i in xrange(N):
		output += str.format("{0:.2f}", w[i]) + "\n"
	output += "EE: " + str.format("{0:.2f}", ee) + "\n"
	output += "CEE: " + str.format("{0:.2f}", cee) + "\n"

	fout.writelines(output)
	fout.close()

	#CLASSIFICATION PREDICTIONS TEST_SET
	(classification, ge) = classify(w0, w, test_set)

	#OUTPUT CLASSIFICATION
	fout = open(OUTPUT_FOLDER + "ispitni-predikcije.dat", "w")
	output = ""
	for h in classification:
		output += str(h) + "\n"
	output += "Greška: " + str.format("{0:.2f}", ge) + "\n"
	fout.writelines(output)
	fout.close()

	#DICTIONARY OUTPUT
	indexes = [i for i in xrange(N)]
	map_w = zip(w, indexes)
	map_w = sorted(map_w)
	map_w = map_w[::-1]

	words = [word for (value, word) in map_w[:20]]

	output = ""
	for i in words:
		output += dictionary[i] + "\n"
	fout = open(OUTPUT_FOLDER + "rijeci.txt", "w")
	fout.writelines(output)
	fout.close()


def classify(w0, w, example_set):

	classification = []
	error = 0
	for (x,y) in example_set:
		h = round(sigmoid(x, w0, w))
		classification.append(int(h))

		if h != y:
			error += 1.0

	return (classification, error / len(example_set))

def cross_validation(train_set, validation_set, N):

	optimization = []

	(w0, w) = batch_gradient_descent(train_set, 0.0, N)
	ge = generalization_error(w0, w, validation_set)
	optimization.append((0, ge))

	(w0, w) = batch_gradient_descent(train_set, 0.1, N)
	ge = generalization_error(w0, w, validation_set)
	optimization.append((0.1, ge))

	(w0, w) = batch_gradient_descent(train_set, 1.0, N)
	ge = generalization_error(w0, w, validation_set)
	optimization.append((1, ge))

	(w0, w) = batch_gradient_descent(train_set, 5, N)
	ge = generalization_error(w0, w, validation_set)
	optimization.append((5, ge))

	(w0, w) = batch_gradient_descent(train_set, 10, N)
	ge = generalization_error(w0, w, validation_set)
	optimization.append((10, ge))

	(w0, w) = batch_gradient_descent(train_set, 100, N)
	ge = generalization_error(w0, w, validation_set)
	optimization.append((100, ge))

	(w0, w) = batch_gradient_descent(train_set, 1000, N)
	ge = generalization_error(w0, w, validation_set)
	optimization.append((1000, ge))

	return optimization

def generalization_error(w0, w, example_set):

	error = 0
	for (x,y) in example_set:
		h = round(sigmoid(x, w0, w))

		if h != y:
			error += 1.0

	return error / len(example_set)

def batch_gradient_descent(example_set, reg_factor, N):

	w = numpy.zeros(N)
	w0 = 0.0
	cee = -1
	curr_error = -1

	while cee == -1 or math.fabs(curr_error - cee) > 10E-04:

		cee = curr_error

		delta_w0 = 0.0
		delta = numpy.zeros(N)

		for (x, y) in example_set:

			h = sigmoid(x, w0, w)

			delta_w0 += (h - y)
			delta += (h - y) * x

		learning_rate = line_search(reg_factor, example_set, delta_w0, delta, w0, w, N)
		w0 -= learning_rate * delta_w0
		w *= (1 - learning_rate * reg_factor)
		w -= learning_rate * delta

		curr_error = cross_entropy_error(reg_factor, example_set, w0, w)

	return (w0, w)

def sigmoid(x, w0, w):

	alfa = w0
	alfa += numpy.dot(x, w)

	if alfa > 30:
		return 1 - 1E-13
	elif alfa < - 30:
		return 1E-13
	else:
		return 1 / (1 + math.exp((-1) * alfa))

def line_search(reg_factor, example_set, delta_w0, delta, w0, w, N):

	learning_rate = 0.0
	delta_lr = 0.01

	if reg_factor >= 100:
		delta_lr /= 10
	if reg_factor >= 1000:
		delta_lr /= 10


	error = cross_entropy_error(reg_factor, example_set, w0, w)

	while learning_rate < 1.0:
		learning_rate += delta_lr

		w_i = numpy.zeros(N)
		w_i = w * (1 - learning_rate * reg_factor) - learning_rate * delta

		curr_error = cross_entropy_error(reg_factor, example_set, w0 - learning_rate * delta_w0, w_i)

		if curr_error > error:
			return learning_rate - delta_lr

		error = curr_error

	return learning_rate

def cross_entropy_error(reg_factor, example_set, w0, w):

	error = 0
	for (x, y) in example_set:
		h = sigmoid(x, w0, w)
		error -= y * math.log(h) + (1 - y) * math.log(1 - h)
	error += reg_factor / 2 * numpy.dot(w, w)
	return error

def readDictFromFile(file):
	
	dictionary = {}

	count = 0
	for line in file.readlines():
		data = line.split()
		word = data[0].strip()
		dictionary[count] = word
		count += 1
	return dictionary

def readSetFromFile(file, N):
	example_set = list()

	for line in file.readlines():
		data = line.split()

		output = int(data[0].strip())

		input_example = numpy.zeros(N)

		for feature in data[1:]:
			feature = feature.strip()
			feature = feature.split(":")
			feature_num = int(feature[0])
			feature_val = float(feature[1])
			input_example[feature_num] = feature_val

		example = (input_example, output)
		
		example_set.append(example)

	return example_set

if __name__ == "__main__": main()
