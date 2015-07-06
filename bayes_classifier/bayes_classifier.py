import math

#izračunati pogrešku učenja za svaki od 4 modela
#izračunati generalizacijsku pogrešku

class BayesClassifier(object):

	def __init__(self, example_set):

		self.output_header = ["Narančasta", "Žuta", "Zelena", "Plava", "Tirkizna", "Indigo", "Modra", "Magenta"]
		self.OUTPUT_FOLDER = "../output/"

		self.class_occurences = {}
		self.class_aprior = {}
		self.class_means = {}
		self.class_covariance_matrices = {}
		self.class_covariance_matrices_determninants = {}
		self.class_examples = {}
		self.train_set = example_set
		self.N = len(self.train_set)
		self.shared_covariance_matrix = {}
		self.diagonal_covariance_matrix = {}
		self.isotrop_covariance_matrix = {}


		if self.N < 1:
			print("Train set doesn't contain any examples")
			sys.exit(-1)

		self.n = len(example_set[0][0])


		self.initializeClassExamples()
		self.calculateApriorProbabilities()
		self.calculateMeans()
		self.calculateCovarianceMatrices()
		self.calculateDeterminants()
		self.calculateSharedAndDiagonalCovarianceMatrix()


	def initializeClassExamples(self):
		for example in self.train_set:
			Cj = example[1]
			example_input = example[0]
			example_list = self.class_examples.get(Cj, [])
			example_list.append(example_input)
			self.class_examples[Cj] = example_list

	def calculateApriorProbabilities(self):

		for example in self.train_set:
			Cj = example[1]
			self.class_occurences[Cj] = self.class_occurences.get(Cj, 0) + 1

		for Cj in self.class_occurences:
			self.class_aprior[Cj] = self.class_occurences[Cj] / self.N


	def mean(self, index, Cj):
		
		sum = 0.
		for example in self.class_examples[Cj]:
			sum += example[index]

		return float(sum) / self.class_occurences[Cj]

	def meanTwoVariables(self, i, j, Cj):
		
		sum = 0
		for example in self.class_examples[Cj]:
			sum += example[i] * example[j]

		return sum / self.class_occurences[Cj]


	def calculateMeans(self):

		for Cj in self.class_occurences:
			mean_vector = []
			for i in range(self.n):
				mean_vector.append(self.mean(i, Cj))
			self.class_means[Cj] = tuple(mean_vector)


	def calculateCovarianceMatrices(self):
		
		for Cj in self.class_occurences:
			self.class_covariance_matrices[Cj] = self.getCovariance_matrix(Cj)

			matrica = self.class_covariance_matrices[Cj]

			# print(Cj)
			# print(self.class_means[Cj])
			# print("matrica")
			# for i in range(self.n):
			#  	ispis = []
			#  	for j in range(self.n):
			#  		ispis.append(matrica[(i,j)])
			#  	print(ispis)
			# print()


	def getCovariance_matrix(self, Cj):

		covariance_matrix = dict()

		for i in range(self.n):
			covariance_matrix[(i,i)] = self.covariance(i, i, Cj)
			for j in range(i+1, self.n):
				covariance = self.covariance(i, j, Cj) 
				covariance_matrix[(i,j)] = covariance
				covariance_matrix[(j,i)] = covariance


		return covariance_matrix

	def covariance(self, i, j, Cj):
		
		covariance = self.meanTwoVariables(i, j, Cj) - self.class_means[Cj][i] * self.class_means[Cj][j]

		return covariance

	def classify(self, train_set, test_set):

		lines = []
		gen_empirial = self.classifySet(train_set, None, "general")
		gen_generalization = self.classifySet(test_set, "opceniti.dat", "general")
		gen = "opceniti\t" + "{:.3f}".format(gen_empirial) + "\t" + "{:.3f}".format(gen_generalization) + "\n"
		lines.append(gen)

		shared_empirial = self.classifySet(train_set, None, "shared")
		shared_generalization = self.classifySet(test_set, "dijeljena.dat", "shared")
		shared = "dijeljena\t" + "{:.3f}".format(shared_empirial) + "\t" + "{:.3f}".format(shared_generalization) + "\n"
		lines.append(shared)

		diagonal_empirial = self.classifySet(train_set, None, "diagonal")
		diagonal_generalization = self.classifySet(test_set, "dijagonalna.dat", "diagonal")
		diagonal = "dijagonalna\t" + "{:.3f}".format(diagonal_empirial) + "\t" + "{:.3f}".format(diagonal_generalization) + "\n"
		lines.append(diagonal)

		isotrop_empirial = self.classifySet(train_set, None, "isotrop")
		isotrop_generalization = self.classifySet(test_set, "izotropna.dat", "isotrop")
		isotrop = "izotropna\t" + "{:.3f}".format(isotrop_empirial) + "\t" + "{:.3f}".format(isotrop_generalization) + "\n"
		lines.append(isotrop)

		output = open(self.OUTPUT_FOLDER + "greske.dat", "w")
		output.writelines(lines)



	def classifySet(self, example_set, filename, flag):

		negative = 0
		matrix = {}
		determinant = 0.0

		output_list = []

		ambivalent_indices = {}

		output = ""

		for header in self.output_header:
			output += header + "\t"

		for Cj in sorted(self.class_occurences):
			if Cj not in self.output_header:
				output += Cj + "\t"
		output += "Klasa\n"

		output_list.append(output)

		if flag == "shared":
			matrix = self.shared_covariance_matrix
			determinant = self.determinant(matrix)
		elif flag == "diagonal":
			matrix = self.diagonal_covariance_matrix
			determinant = self.determinant(matrix)
		elif flag == "isotrop":
			matrix = self.isotrop_covariance_matrix
			determinant = self.determinant(matrix)

		index = 0

		for example in example_set:
			x = example[0]
			class_likelihoods = {}
			class_aposteriori = {}
			
			P_x = 0.0

			for Cj in self.class_occurences:
				if flag == "general":
					matrix = self.class_covariance_matrices[Cj]
					determinant = self.class_covariance_matrices_determninants[Cj]
				class_likelihoods[Cj] = self.classLikelihood(x, Cj, matrix, determinant)
				P_x += class_likelihoods[Cj] * self.class_aprior[Cj]

			for Cj in self.class_occurences:
				class_aposteriori[Cj] = class_likelihoods[Cj] * self.class_aprior[Cj] / P_x

			classification = max(class_aposteriori, key=class_aposteriori.get)
											
			output_list.append(self.output_line(class_aposteriori, classification))

			if classification != example[1]:
				negative += 1

			#nejednoznacne
			if flag == "general":

				new_ambivalent = False
				if len(ambivalent_indices) < 5:
					ambivalent_indices[index] = class_aposteriori[classification]
				else:
					for key in ambivalent_indices:
						if class_aposteriori[classification] < ambivalent_indices[key]:
							new_ambivalent = True
							break

					if new_ambivalent == True:
						ambivalent_indices[index] = class_aposteriori[classification]
						remove_index = max(ambivalent_indices, key = ambivalent_indices.get)
						del ambivalent_indices[remove_index]
			
			index +=1

		if filename != None:
			fout = open(self.OUTPUT_FOLDER + filename, "w")
			fout.writelines(output_list)
			
			if flag == "general":
				fambivalent = open(self.OUTPUT_FOLDER + "nejednoznacne.dat", "w")
				ambivalent_output = []


				ambivalent_header = ""
				for header in self.output_header:
					ambivalent_header += header + "\t"
				for Cj in sorted(self.class_occurences):
					if Cj not in self.output_header:
						ambivalent_header += Cj + "\t"
				ambivalent_header += "Klasa\n"

				ambivalent_output.append(ambivalent_header)
				for key in sorted(ambivalent_indices, key = ambivalent_indices.get):
					ambivalent_output.append(output_list[key + 1])
				fambivalent.writelines(ambivalent_output)


		return negative / len(example_set)

	def classLikelihood(self, x, Cj, matrix, determinant):


		likelihood = 1 / (math.pow(2 * math.pi, self.n / 2) * math.sqrt(determinant))

		x_minus_mean = tuple(x - y for x,y in zip(x, self.class_means[Cj]))

		likelihood *= math.exp(-0.5 * self.quadraticForm(x_minus_mean, matrix, x_minus_mean))

		return likelihood

	def output_line(self, class_aposteriori, classification):

		output = ""
		for header in self.output_header:
			output += "{:.3f}".format(class_aposteriori[header]) + "\t"

		for Cj in sorted(self.class_occurences):
			if Cj not in self.output_header:
				output += "{:.3f}".format(class_aposteriori[Cj]) + "\t"
		output += classification + "\n"

		return output


	def quadraticForm(self, x, matrix, y):
		
		inverse_matrix = self.invert(matrix)

		x_times_matrix = [0] * self.n

		x_transposed = list(x)
		x_vector = list(y)

		for i in range(self.n):
			for j in range(self.n):
				x_times_matrix[i] += x_transposed[j] * inverse_matrix[(j,i)]

		result = 0.0
		for i in range(self.n):
			result += x_times_matrix[i] * x_vector[i]
		
		return result

	def invert(self, matrix):

		#cofactor
		invert_matrix = self.cofactorMatrix(matrix)
		#transpose
		invert_matrix = self.transpose(invert_matrix)

		determinant = self.determinant(matrix)
		for i in range(self.n):
			for j in range(self.n):
				invert_matrix[(i,j)] *= 1.0 / determinant
		return invert_matrix

	def cofactorMatrix(self, matrix):

		dimension = int(math.sqrt(len(matrix)))

		cofactor = {}
		for i in range(dimension):
			for j in range(dimension):
				cofactor[(i,j)] = self.isEven(i) * self.isEven(j) * self.determinant(self.subMatrix(matrix, i, j))
		return cofactor

	def transpose(self, matrix):


		dimension = int(math.sqrt(len(matrix)))

		transposed_matrix = {}
		for i in range(dimension):
			for j in range(dimension):
				transposed_matrix[(j,i)] = matrix[(i,j)]
		return transposed_matrix

	def determinant(self, matrix):

		dimension = int(math.sqrt(len(matrix)))

		if dimension == 1:
			return matrix[(0,0)]
		if dimension == 2:
			return matrix[(0,0)] * matrix[(1,1)] - matrix[(0,1)] * matrix[(1,0)]

		det = 0.0
		for i in range(dimension):
			det += self.isEven(i) * matrix[(0,i)] * self.determinant(self.subMatrix(matrix, 0, i))
		return det

	def isEven(self, i):
		if i % 2 == 0:
			return 1
		else:
			return -1

	def subMatrix(self, matrix, row, column):

		dimension = int(math.sqrt(len(matrix)))

		sub_matrix = {}
		subRow = -1
		for i in range(dimension):
			if i == row:
				continue
			subRow += 1
			subCol = -1
			for j in range(dimension):
				if j == column:
					continue
				subCol += 1
				sub_matrix[(subRow, subCol)] = matrix[(i,j)]

		return sub_matrix

	def calculateDeterminants(self):

		for Cj in self.class_occurences:
			self.class_covariance_matrices_determninants[Cj] = self.determinant(self.class_covariance_matrices[Cj])

	def calculateSharedAndDiagonalCovarianceMatrix(self):

		for Cj in self.class_occurences:

			apriori = self.class_aprior[Cj]
			cov_matrix = self.class_covariance_matrices[Cj]

			for i in range(self.n):
				for j in range(self.n):
					self.shared_covariance_matrix[(i,j)] = self.shared_covariance_matrix.get((i,j), 0) + apriori * cov_matrix[(i,j)]

		sum = 0.0
		for i in range(self.n):
			for j in range(self.n):
				if i == j:
					self.diagonal_covariance_matrix[(i,j)] = self.shared_covariance_matrix[(i,j)]
					sum += self.diagonal_covariance_matrix[(i,j)]
				else:
					self.diagonal_covariance_matrix[(i,j)] = 0.0
		
		sum /= self.n
		for i in range (self.n):
			for j in range (self.n):
				if i == j:
					self.isotrop_covariance_matrix[(i,j)] = sum
				else:
					self.isotrop_covariance_matrix[(i,j)] = 0.0
