import sys
import numpy
import math


def main():

	if (len(sys.argv) != 4):
		print("Wrong number of arguments!")
		print("Provide file names!")
		sys.exit(-1)


	OUTPUT_FOLDER = sys.argv[3]
	configurations_path = sys.argv[2]
	data_path = sys.argv[1]

	kmeans_path = OUTPUT_FOLDER + "kmeans-all.dat"
	kmeans_k4_path = OUTPUT_FOLDER + "kmeans-k4.dat"
	em_path = OUTPUT_FOLDER + "em-all.dat"
	em_k4_path = OUTPUT_FOLDER + "em-k4.dat"
	em_conf_path = OUTPUT_FOLDER + "em-konf.dat"
	em_kmeans_path = OUTPUT_FOLDER + "em-kmeans.dat"

	f_data_set = open(data_path, "r")
	f_kmeans = open(kmeans_path, "w")
	f_kmeans_k4 = open(kmeans_k4_path, "w")
	f_em = open(em_path, "w")
	f_em_k4 = open(em_k4_path, "w")
	f_configurations = open(configurations_path, "r")
	f_em_conf = open(em_conf_path, "w")
	f_em_kmeans = open(em_kmeans_path, "w")

	data_set, classes, n = readSetFromFile(f_data_set)

	print "ALGORITAM K-SREDNIH VRIJEDNOSTI"
	print "###############################\n"

	kmeans_output = []

	K = 2
	means, b, J, iter_num = k_means_algorithm(K, data_set, n)
	kmeans_output.extend(logger(means, b, K, J, iter_num))
	kmeans_output.append("--")

	K = 3
	means, b, J, iter_num = k_means_algorithm(K, data_set, n)
	kmeans_output.extend(logger(means, b, K, J, iter_num))
	kmeans_output.append("--")

	K = 4
	means, b, J, iter_num = k_means_algorithm(K, data_set, n, f_kmeans_k4)
	start_means = means
	kmeans_output.extend(logger(means, b, K, J, iter_num))
	kmeans_output.append("--")

	K = 5
	means, b, J, iter_num = k_means_algorithm(K, data_set, n)
	kmeans_output.extend(logger(means, b, K, J, iter_num))

	f_kmeans.write("\n".join(kmeans_output))


	print "\nALGORITAM MAKSIMIZACIJE OCEKIVANJA"
	print "##################################\n"

	em_output = []

	K = 2
	means_indices = getIndicesFromDataSet(K, data_set)
	init_means = []
	for index in means_indices:
		init_means.append((data_set[index][0], data_set[index][1]))
	means, h, likelihood, iter_num = em_algorithm(K, data_set, n, init_means)
	b = getBfromH(h)
	em_output.extend(logger(means, b, K, likelihood, iter_num, True))
	em_output.append("--")

	K = 3
	means_indices = getIndicesFromDataSet(K, data_set)
	init_means = []
	for index in means_indices:
		init_means.append((data_set[index][0], data_set[index][1]))
	means, h, likelihood, iter_num = em_algorithm(K, data_set, n, init_means)
	b = getBfromH(h)
	em_output.extend(logger(means, b, K, likelihood, iter_num, True))
	em_output.append("--")

	K = 4
	means_indices = getIndicesFromDataSet(K, data_set)
	init_means = []
	for index in means_indices:
		init_means.append((data_set[index][0], data_set[index][1]))
	means, h, likelihood, iter_num = em_algorithm(K, data_set, n, init_means)
	b = getBfromH(h)
	output_groups(h, b, data_set, f_em_k4)
	em_output.extend(logger(means, b, K, likelihood, iter_num, True))
	em_output.append("--")

	K = 5
	means_indices = getIndicesFromDataSet(K, data_set)
	init_means = []
	for index in means_indices:
		init_means.append((data_set[index][0], data_set[index][1]))
	means, h, likelihood, iter_num = em_algorithm(K, data_set, n, init_means)
	b = getBfromH(h)
	em_output.extend(logger(means, b, K, likelihood, iter_num, True))

	f_em.write("\n".join(em_output))

	#KONFIGURACIJE
	print "\nKONFIGURACIJE"
	print "###############"
	output_complete = []

	K = 4
	configurations = getConfigFromFile(f_configurations, K)
	init_means = configurations[0]
	means, h, likelihood, iter_num = em_algorithm(K, data_set, n, init_means)
	output_complete.extend(log_configuration(1, likelihood, iter_num))
	print "--"
	output_complete.append("--")

	init_means = configurations[1]
	means, h, likelihood, iter_num = em_algorithm(K, data_set, n, init_means)
	output_complete.extend(log_configuration(2, likelihood, iter_num))
	print "--"
	output_complete.append("--")

	# init_means = configurations[2]
	# means, h, likelihood, iter_num = em_algorithm(K, data_set, n, init_means)
	# output_complete.extend(log_configuration(3, likelihood, iter_num))
	# print "--"
	# output_complete.append("--")

	init_means = configurations[3]
	means, h, likelihood, iter_num = em_algorithm(K, data_set, n, init_means)
	output_complete.extend(log_configuration(4, likelihood, iter_num))
	print "--"
	# output_complete.append("--")

	# init_means = configurations[4]
	# means, h, likelihood, iter_num = em_algorithm(K, data_set, n, init_means)
	# output_complete.extend(log_configuration(5, likelihood, iter_num))

	f_em_conf.write("\n".join(output_complete))


	#START EM WITH K-MEANS
	# K = 4
	# means, h, likelihood, iter_num = em_algorithm(K, data_set, n, start_means, f_em_kmeans)


def log_configuration(i, likelihood, iter_num):
	output_complete = []
	output = "Konfiguracija %d:" % (i)
	print output
	output_complete.append(output)
	output = "log-izglednost: %.2f" % (likelihood)
	print output
	output_complete.append(output)
	output = "#iteracija: %d" % (iter_num)
	print output
	output_complete.append(output)
	return output_complete
	


def getConfigFromFile(config, K):

	lines = config.readlines()
	i = 0
	configurations = []
	while i < len(lines):
		conf = []
		for line in lines[i + 1: i + K + 1]:
			data = line.split()
			clazz = data[-1]
			data = data[:-1]
			n = len(data)
			example_input = numpy.zeros(n)

			for j, feature in enumerate(data):
				feature = feature.strip()
				example_input[j] = float(feature)

			example = (example_input, clazz)
			conf.append(example)

		configurations.append(conf)
		i += (K + 2) 

	return configurations


def output_groups(h, b, data_set, f_output):

	K = len(h[0])
	groups = []
	for x in xrange(K):
		groups.append([])
	for i, (x, clazz) in enumerate(data_set):
		for k, group in enumerate(groups):
			if b[i][k] == 1:
				group.append((clazz, h[i][k]))
				break
	output_complete = []
	for k, group in enumerate(groups):
		output = "Grupa %s:" % (str(k + 1))
		output_complete.append(output)
		for clazz, p in sorted(group, reverse = True, key = lambda x: x[1]):
			output = "%s %.2f" % (clazz, p)
			output_complete.append(output)
		if k < len(groups) - 1:
			output_complete.append("--")
	f_output.write("\n".join(output_complete))


def getBfromH(h):
	b = []
	for i, hi in enumerate(h):
		K = len(hi)
		bi = numpy.zeros(K)
		max_val = None
		for k in xrange(K):
			if max_val is None or max_val[1] < hi[k]:
				max_val = (k, hi[k])
		bi[max_val[0]] = 1
		b.append(bi)
	return b


def em_algorithm(K, data_set, n, means, f_em = None):

	print "\nEM algorithm with K = %s" % (K)
	print "------------------------------------"

	#inicijalizacija parametara pi, sigma
	N = len(data_set)
	covariance_matrices = []
	pi = []
	det_list = []
	inv_list = []
	for k in xrange(K):
		covariance_matrices.append(numpy.identity(n))
		det_list.append(numpy.linalg.det(covariance_matrices[k]))
		inv_list.append(numpy.linalg.inv(covariance_matrices[k]))
		pi.append(1 / float(K))

	iter_num = 0
	likelihood = None
	curr_likelihood = None


	output_complete = []
	if f_em is not None:
		output = "#iteracije: log-izglednost\n--"
		output_complete.append(output)
		curr_likelihood = 0.0
		for i, (x, _) in enumerate(data_set):
			value = 0.0
			for k, (mi, _) in enumerate(means):
				det = det_list[k]
				inv = inv_list[k]
				temp_value = pi[k] * 1 / (math.pow(2 * math.pi, n / 2) * math.sqrt(det))
				m = numpy.matrix(x - mi)
				temp_value *= math.exp((-1.0) / 2 * m * inv * m.T)
				value += temp_value
			curr_likelihood += math.log(value)
		output = "#%d: %.2f" % (iter_num, likelihood)
		output_complete.append(output)

	#UVJET ZAUSTAVLJANJA DODATI
	while likelihood is None or math.fabs(curr_likelihood - likelihood) > 1E-5:

		likelihood = curr_likelihood

		#E KORAK
		h = []
		for i, (x, _) in enumerate(data_set):
			hi = numpy.zeros(K)
			di = 0.0
			for k, (mi, _) in enumerate(means):
				cov = covariance_matrices[k]
				# det = numpy.linalg.det(cov)
				det = det_list[k]
				inv = inv_list[k]
				hi[k] = pi[k] * 1 / (math.pow(2 * math.pi, n / 2) * math.sqrt(det))
				m = numpy.matrix(x - mi)
				hi[k] *= math.exp((-1.0) / 2 * m * inv * m.T)
				di += hi[k]
			hi /= di
			h.append(hi)

		#M KORAK
		det_list = []
		inv_list = []
		for k, (_, clazz) in enumerate(means):
			k_freq = 0
			
			#srednje vrijednosti
			numerator = numpy.zeros(n)
			for i, (x, _) in enumerate(data_set):
				numerator += h[i][k] * x
				k_freq += h[i][k]
			mi = numerator / k_freq
			means[k] = (mi, clazz)

			#kovarijacijske matrice
			cov = numpy.zeros((n, n))
			for i, (x, _) in enumerate(data_set):
				matrix = numpy.matrix(x - mi)
				cov += h[i][k] * (matrix.T * matrix) 
			cov /= k_freq
			covariance_matrices[k] = cov
			det_list.append(numpy.linalg.det(cov))
			inv_list.append(numpy.linalg.inv(cov))

			#komponent mjesavine
			pi[k] = 1.0 / N * k_freq
			

		# #TRENUTNA VRIJEDNOST IZGLEDNOSTI
		curr_likelihood = 0.0
		for i, (x, _) in enumerate(data_set):
			value = 0.0
			for k, (mi, _) in enumerate(means):
				det = det_list[k]
				inv = inv_list[k]
				temp_value = pi[k] * 1 / (math.pow(2 * math.pi, n / 2) * math.sqrt(det))
				m = numpy.matrix(x - mi)
				temp_value *= math.exp((-1.0) / 2 * m * inv * m.T)
				value += temp_value
			curr_likelihood += math.log(value)

		if f_em is not None:
			output = "#%d: %.2f" % (iter_num + 1, likelihood)
			output_complete.append(output)

		iter_num += 1

	#ispis ako treba
	if f_em is not None:
		b = getBfromH(h)
		output_complete.append("--")
		for k, (mi, _) in enumerate(means):
			grouping = {}
			for i, (_, clazz) in enumerate(data_set):
				if b[i][k] == 1:
					grouping[clazz] = grouping.get(clazz, 0) + 1
			output = "Grupa %d: " % (k + 1) + ", ".join("%s %s" % (k, v) for (k, v) in sorted(grouping.items(), key=lambda x: x[1], reverse = True))
			output_complete.append(output)
		f_em.write("\n".join(output_complete))

	likelihood = curr_likelihood

	return means, h, likelihood, iter_num


def logger(means, b, K, J, iter_num, em = False):
	count_examples = [0] * K

	output = "K = %s" % (K)
	output_complete = [output]
	print output
	for i, (mi, label) in enumerate(means):
		output = "c%s: " % (i + 1) + " ".join(map(lambda x: "%.2f" % x, mi))
		print output

		output_complete.append(output)

		for bi in b:
			if bi[i] == 1:
				count_examples[i] += 1
		output = "grupa %s: %s primjera" % (i + 1, str(count_examples[i]))
		print output

		output_complete.append(output) 

	output = "#iter: %d" % (iter_num)
	output_complete.append(output)
	print output

	if em:
		output = "log-izglednost: %.2f" % (J)
	else:
		output = "J: %.2f" % (J)
	output_complete.append(output)
	print output

	return output_complete


def k_means_algorithm(K, data_set, n, f_kmeans = None):

	print "\nK - MEANS algorithm with K = %s" % (K)
	print "------------------------------------"

	output_complete = []
	if f_kmeans is not None:
		output = "#iteracije: J\n--"
		output_complete.append(output)

	means_indices = getIndicesFromDataSet(K, data_set)
	means = []
	for index in means_indices:
		means.append((data_set[index][0], data_set[index][1]))

	b = []
	iter_num = 0
	changed = True
	J = 0

	while changed:

		b_temp = b[:]
		b = []
		changed = False

		for i, (x, _) in enumerate(data_set):

			min_dist = None
			for k, (mi, _) in enumerate(means):
				dist = numpy.linalg.norm(x - mi)
				if min_dist is None or dist < min_dist[1]:
					min_dist = (k, dist)

			bi = numpy.zeros(K)
			if iter_num == 0 or b_temp[i][min_dist[0]] == 0:
				changed = True
			bi[min_dist[0]] = 1
			b.append(bi)
	
		J = criterion_function(data_set, means, b)

		if f_kmeans is not None:
			output = "#%d: %.2f" % (iter_num, J)
			output_complete.append(output)
		iter_num += 1

		for k, (mi, label) in enumerate(means[:]):
			mi = numpy.zeros(n)
			k_freq = 0
			for i, (x, _) in enumerate(data_set):
				if b[i][k] == 1:
					mi += x
					k_freq += 1
			mi /= k_freq
			means[k] = (mi, label)

	if f_kmeans is not None:
		output_complete.append("--")
		for k, (mi, _) in enumerate(means):
			grouping = {}
			for i, (_, clazz) in enumerate(data_set):
				if b[i][k] == 1:
					grouping[clazz] = grouping.get(clazz, 0) + 1
			output = "Grupa %d: " % (k + 1) + ", ".join("%s %s" % (k, v) for (k, v) in sorted(grouping.items(), key=lambda x: x[1], reverse = True)) #grouping.iteritems()
			output_complete.append(output)
		f_kmeans.write("\n".join(output_complete))

	return means, b, J, iter_num


def criterion_function(data_set, means, b):
	value = 0.0
	for k, (mi, _) in enumerate(means):
		for i, (x, _) in enumerate(data_set):
			dist = numpy.linalg.norm(x - mi)
			value += b[i][k] * dist * dist
	return value


def getIndicesFromDataSet(K, data_set):
	
	indices = [None] * K

	if K >= 2:
		for i, (_, clazz) in enumerate(data_set):
			
			if indices[0] is None and clazz == "opel":
				indices[0] = i
			if indices[1] is None and clazz == "bus":
				indices[1] = i

			if K >= 3:
				if indices[2] is None and clazz == "van":
					indices[2] = i
				
				if K >= 4:
					if indices[3] is None and clazz == "saab":
						indices[3] = i

					if K >= 5 and indices[3] is not None and i != indices[3]:
						if indices[4] is None and clazz == "saab":
							indices[4] = i

	return tuple(indices)


def readSetFromFile(file):

	data_set = list()
	classes = list()
	n = 0

	for line in file.readlines():
		data = line.split()

		clazz = data[-1]
		if clazz not in classes:
			classes.append(clazz)

		data = data[:-1]
		n = len(data)
		example_input = numpy.zeros(n)

		for i, feature in enumerate(data):
			feature = feature.strip()
			example_input[i] = float(feature)

		example = (example_input, clazz)

		data_set.append(example)

	return data_set, classes, n


if __name__ == "__main__": main()