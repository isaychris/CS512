# =====================================================================================================================
# Group 8: Chris Banci / Esmond Liang
# CS512 - Data Mining
# 4/2/18
#
# Assignment 3 - REDO
# =====================================================================================================================
import time
import numpy as np
import csv
import mlr
import FromDataFileMLR
import FromFinessFileMLR


class GeneticAlgorithm:
    def __init__(self, model, fff, pop, fea, gen):
        self.numOfPop = pop
        self.numOfFea = fea
        self.numOfGenerations = gen
        self.population = np.zeros((self.numOfPop, self.numOfFea))
        self.fitness = np.zeros(self.numOfPop)
        self.model = model
        self.fff = fff
        self.fileW = self.createAnOutputFile()

    # -------------------------------------------------------------------------------------------------

    # Purpose: creates an initial population.
    def create_initial_population(self):
        np.random.seed()

        for i in range(self.numOfPop):
            row = self.getAValidrow()
            self.population[i] = row

    # -------------------------------------------------------------------------------------------------

    # Purpose: creates a new population based off the current population and its fitness
    def create_new_population(self):
        newPopulation = np.zeros((self.numOfPop, self.numOfFea))

        # select the two best fit models by sorting fitness
        newPopulation[0], newPopulation[1] = self.selection()

        # do a crossover between them to create children
        child1, child2 = self.crossover(newPopulation[0], newPopulation[1])

        # mutate the children
        child1 = self.mutate(child1)
        child2 = self.mutate(child2)

        # put created children into new population
        newPopulation[2] = child1
        newPopulation[3] = child2

        # for the next 46, fill in the rest of new population just like when creating the initial population.
        for i in range(4, self.numOfPop):
            row = self.getAValidrow()
            newPopulation[i] = row

        self.population = newPopulation

    # -------------------------------------------------------------------------------------------------

    # Purpose: evolves the population n times
    def evolve_population(self, trainX, trainY, validateX, validateY, testX, testY):
        print('gen #' + str(0) + '\t\t min fit: ' + str(self.fitness.min()))

        for i in range(1, self.numOfGenerations):
            self.create_new_population()
            self.evaluate_population(trainX, trainY, validateX, validateY, testX, testY)
            print('gen #' + str(i) + '\t\t min fit: ' + str(self.fitness.min()))


    # -------------------------------------------------------------------------------------------------

    # Purpose: evaluates the population and returns its fitnesses
    def evaluate_population(self, trainX, trainY, validateX, validateY, testX, testY):
        fittingStatus, self.fitness = self.fff.validate_model(self.model, self.fileW, self.population, trainX,
                                                                  trainY, validateX, validateY, testX, testY)

    # -------------------------------------------------------------------------------------------------

    # Purpose: selects the two best fit models from sorting population by its fitnesses.
    def selection(self):
        idx = self.fitness.argsort()
        sortedPopulation = self.population[idx]

        return sortedPopulation[0], sortedPopulation[1]

    # -------------------------------------------------------------------------------------------------

    # Purpose: creates two children by crossing over the features of mom and dad
    def crossover(self, mom, dad):
        pos = np.random.randint(1, len(mom) - 1)
        child1 = np.append(dad[pos:], mom[:pos])
        child2 = np.append(dad[:pos], mom[pos:])

        return child1, child2

    # -------------------------------------------------------------------------------------------------

    # Purpose: mutates the features of a child by flipping its bits
    def mutate(self, child, eps=0.0005):
        for i in range(len(child)):
            if np.random.uniform(0, 1) < eps:
                child[i] = 1 - child[i]

        return child

    # -------------------------------------------------------------------------------------------------

    # Purpose: randomly selects 1.5% of features in a row
    def getAValidrow(self, eps=0.015):
        sum = 0
        while (sum < 3):
            row = np.zeros(self.numOfFea)
            for j in range(self.numOfFea):
                if (np.random.uniform(0, 1) < eps):
                    row[j] = 1
            sum = row.sum()
        return row

    # -------------------------------------------------------------------------------------------------

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
    # GA parameters
    num_pop = 50
    num_feat = 385
    num_gens = 1000


    # initialize objects
    model = mlr.MLR()
    fdf = FromDataFileMLR.FromDataFileMLR()
    fff = FromFinessFileMLR.FromFinessFileMR(fdf)
    GA = GeneticAlgorithm(model, fff, num_pop, num_feat, num_gens)

    # load in data from files
    trainX, trainY, validateX, validateY, testX, testY = fdf.getAllOfTheData()
    trainX, validateX, testX = fdf.rescaleTheData(trainX, validateX, testX)

    # genetic algorithm
    GA.create_initial_population()
    GA.evaluate_population(trainX, trainY, validateX, validateY, testX, testY)
    GA.evolve_population(trainX, trainY, validateX, validateY, testX, testY)

if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------------------------------------