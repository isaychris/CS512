# =====================================================================================================================
# Group 8: Chris Banci / Esmond Liang
# CS512 - Data Mining
# 3/28/18
#
# Assignment 4
# =====================================================================================================================
import time
import csv
import mlr
import numpy as np
import FromDataFileMLR
import FromFinessFileMLR


class DifferentialEvolution:
    def __init__(self, model, fff, pop, fea, gen):
        self.numOfPop = pop
        self.numOfFea = fea
        self.numOfGenerations = gen
        self.population = np.zeros((self.numOfPop, self.numOfFea))
        self.fitness = np.zeros(self.numOfPop)
        self.model = model
        self.fff = fff
        self.fileW = self.createAnOutputFile()


    # Purpose: creates an initial population.
    def create_initial_population(self):
        np.random.seed()

        for i in range(self.numOfPop):
            row = self.getAValidrow()
            self.population[i] = row


    # Purpose: generators a unique set of numbers and checks if set already exists.
    def generate_unique_set(self, used_random_sets):
        random_set = set(np.random.choice(range(50), 3, replace=False))

        # keep generating until random set doesnt exist in the used sets list
        while random_set in used_random_sets:
            random_set = set(np.random.choice(range(50), 3, replace=False))

        used_random_sets.append(random_set)
        return random_set


    # calculates the vector for population at index [i]
    def calculate_vector(self, V, U, x1, x2, x3, F=0.5, CV=0.7):

        # calculate the bits for V[i]
        for j in range(self.numOfFea):
            V[j] = np.floor(x3[j] + F * (x2[j] - x1[j]))

            # if the bit becomes negative, change to 0
            if V[j] < 0:
                V[j] = 0

            # do cross validation, set the bit j in vector to the bit j in old population.
            if np.random.uniform(0, 1) > CV:
                V[j] = U[j]

        return V


    # Purpose: creates a new population based off the current population and its fitness
    def create_new_population(self):
        # create the new population
        new_population = np.zeros((self.numOfPop, self.numOfFea))

        # sort old population based off fitness values
        idx = self.fitness.argsort(axis=0)
        self.population = self.population[idx]

        # move best row of old population to first row of new population.
        new_population[0] = self.population[0]

        # list of used unique sets
        used_sets = []

        # initialize the vector
        V = np.zeros((self.numOfPop,self.numOfFea))

        # for the next 49 rows in new population
        for i in range(1, self.numOfPop):
            # get three random rows
            unique_set = self.generate_unique_set(used_sets)
            x1, x2, x3 = self.population[list(unique_set)]

            # calculate vector[i] using three random rows, then assign to new_population[i]
            new_population[i] = self.calculate_vector(V[i], self.population[i], x1, x2, x3)

        self.population = new_population


    # Purpose: evolutionary process
    def evolve_population(self, trainX, trainY, validateX, validateY, testX, testY):
        print('gen #0 \t\t min fit: ' + str(self.fitness.min()) + '\t\t avg fit: ' + str(np.average(self.fitness)))

        for i in range(1, self.numOfGenerations):
            self.create_new_population()
            self.evaluate_population(trainX, trainY, validateX, validateY, testX, testY)
            print('gen #' + str(i) + '\t\t min fit: ' + str(self.fitness.min()) + '\t\t avg fit: ' + str(np.average(self.fitness)))


    # Purpose: evaluates a population and returns its fitness and status
    def evaluate_population(self, trainX, trainY, validateX, validateY, testX, testY):
        fittingStatus, self.fitness = self.fff.validate_model(self.model, self.fileW, self.population, trainX,
                                                              trainY, validateX, validateY, testX, testY)

    # Purpose: randomly selects 1.5% of features in a row
    def getAValidrow(self, eps=0.015):
        sum = 0
        while sum < 3:
            row = np.zeros(self.numOfFea)
            for j in range(self.numOfFea):
                if np.random.uniform(0, 1) < eps:
                    row[j] = 1
            sum = row.sum()
        return row


    # Purpose: The following creates an output file.
    def createAnOutputFile(self):
        file_name = None
        algorithm = None

        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if file_name is None and algorithm is not None:
            file_name = "{}_{}_gen{}_{}.csv".format(algorithm.__class__.__name__, algorithm.model.__class__.__name__,
                                                    algorithm.gen_max, timestamp)
        elif file_name is None:
            file_name = "{}.csv".format(timestamp)

        fileW = csv.writer(file(file_name, 'wb'))
        fileW.writerow(['Descriptor ID', 'Fitness', 'Model','R2', 'Q2', 'R2Pred_Validation', 'R2Pred_Test'])

        return fileW


# --------------------------------------------------------------------------------------------------------------------

def main():
    # DE parameters
    num_pop = 50
    num_feat = 385
    num_gens = 100

    # initialize objects
    model = mlr.MLR()
    FDF = FromDataFileMLR.FromDataFileMLR()
    FFF = FromFinessFileMLR.FromFinessFileMR(FDF)
    DE = DifferentialEvolution(model, FFF, num_pop, num_feat, num_gens)

    # load in data from files
    trainX, trainY, validateX, validateY, testX, testY = FDF.getAllOfTheData()
    trainX, validateX, testX = FDF.rescaleTheData(trainX, validateX, testX)

    # differential evolution algorithm
    DE.create_initial_population()
    DE.evaluate_population(trainX, trainY, validateX, validateY, testX, testY)
    DE.evolve_population(trainX, trainY, validateX, validateY, testX, testY)

if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------------------------------------

