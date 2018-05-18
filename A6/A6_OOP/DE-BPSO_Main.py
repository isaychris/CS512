# =====================================================================================================================
# Chris Banci
# CS512 - Data Mining
# 4/26/18
#
# Assignment 6 - OOP
# =====================================================================================================================
import time
import numpy as np
import csv
import mlr
import FromDataFileMLR
import FromFinessFileMLR

class DE_BinaryParticleSwarmOptimization:
    def __init__(self, model, fff, pop, fea, gen):
        self.numOfPop = pop
        self.numOfFea = fea
        self.numOfGens = gen
        self.model = model
        self.population = np.zeros((self.numOfPop, self.numOfFea))
        self.fitness = np.zeros(self.numOfPop)

        self.alpha = 0.5
        self.local_best_matrix = np.zeros((self.numOfPop, self.numOfFea))
        self.local_fitness = np.zeros(self.numOfPop)
        self.global_best_row = np.zeros(self.numOfFea)
        self.global_best_row_fitness = 0.0
        self.velocity = np.zeros((self.numOfPop, self.numOfFea))

        self.fff = fff
        self.fileW = self.createAnOutputFile()

    # -------------------------------------------------------------------------------------------------

    # creates an initial population by randomly selecting 1.5% features for each row.
    def create_initial_population(self, Lambda=0.01):
        for i in range(self.numOfPop):
            for j in range(self.numOfFea):
                if self.velocity[i][j] <= Lambda:
                    self.population[i][j] = 1
                else:
                    self.population[i][j] = 0

    # -------------------------------------------------------------------------------------------------

    # creates the initial velocity matrix by setting each element in matrix to a random number between 0 and 1
    def create_initial_velocity(self):
        for i in range(self.numOfPop):
            for j in range(self.numOfFea):
                self.velocity[i][j] = np.random.random()

    # -------------------------------------------------------------------------------------------------

    # creates the initial best matrix by copying the initial population.
    def create_initial_local_best_matrix(self):
        self.local_best_matrix = self.population
        self.local_fitness = self.fitness

    # -------------------------------------------------------------------------------------------------

    # creates the initial global best row by sorting initial population by its fitness and copying its best row.
    def create_initial_global_best_row(self):
        idx = self.local_fitness.argmin(axis=0)
        self.global_best_row = self.local_best_matrix[idx]
        self.global_best_row_fitness = self.local_fitness[idx]

    # -------------------------------------------------------------------------------------------------

    # creates a new population by comparing velocity conditional statements
    def create_new_population(self, current_gen):
        new_population = np.zeros((self.numOfPop, self.numOfFea))

        p = 0.5 * (1 + self.alpha)
        beta = 0.004
        a,b,c,d = 0,0,0,0

        for i in range(self.numOfPop):
            for j in range(self.numOfFea):
                if (self.alpha < self.velocity[i][j]) and (self.velocity[i][j] <= p):
                    new_population[i][j] = self.local_best_matrix[i][j]
                    a += 1

                elif (p < self.velocity[i][j]) and (self.velocity[i][j] <= (1 - beta)):
                    new_population[i][j] = self.global_best_row[j]
                    b += 1

                elif ((1 - beta) < self.velocity[i][j]) and (self.velocity[i][j] <= 1):
                    new_population[i][j] = 1 - self.population[i][j]
                    c += 1

                else:
                    new_population[i][j] = self.population[i][j]
                    d += 1

        if current_gen == (self.numOfGens / 2):
            new_population = self.optimize(new_population)

        self.population = new_population
        print(" - cases = [" + str(a) + ", " + str(b) + ", " + str(c) + ", " + str(d) + "]")

    # -------------------------------------------------------------------------------------------------

    # optimizes the new population at gen 500 by making half of the population random.
    def optimize(self, population):
        print(" -- optimizing population")
        for i in range((self.numOfPop / 2), self.numOfPop):
            row = self.getAValidrow()
            population[i] = row

        return population

    # -------------------------------------------------------------------------------------------------

    # best row from local best matrix becomes global best row
    def update_global_best_row(self):
        idx = self.local_fitness.argmin(axis=0)

        if self.local_fitness[idx] < self.global_best_row_fitness:
            self.global_best_row = self.local_best_matrix[idx]
            self.global_best_row_fitness = self.local_fitness[idx]
            print(' - found new global best row')

    # -------------------------------------------------------------------------------------------------

    # local best matrix updated from comparing local fitness and fitness
    def update_local_best_matrix(self):
        for i in range(self.numOfPop):
            if self.fitness[i] < self.local_fitness[i]:
                self.local_best_matrix[i] = self.population[i]

                # update the local fitness since local best matrix was changed
                self.local_fitness[i] = self.fitness[i]

    # -------------------------------------------------------------------------------------------------

    # velocity matrix is updated through formula.
    def update_velocity(self, F=0.7, CR=0.7):
        new_velocity = np.zeros((self.numOfFea, self.numOfFea))

        for i in range(self.numOfPop):
            random_set = np.random.choice(range(50), 3, replace=False)
            r1, r2, r3 = self.population[random_set]
            r = r3 + F * (r2 - r1)

            for j in range(self.numOfFea):
                # do cross validation between new_velocity and r
                if np.random.random() < CR:
                    new_velocity[i][j] = r[j]
                else:
                    new_velocity[i][j] = self.velocity[i][j]

        self.velocity = new_velocity

    # -------------------------------------------------------------------------------------------------

    # evolves the population a set number of times
    def evolve_population(self, trainX, trainY, validateX, validateY, testX, testY):
        print('gen #' + str(0) + '\t\t min fit: ' + str(self.fitness.min()) + '\t\t avg fit: ' + str(np.average(self.fitness)))

        for i in range(1, self.numOfGens):
            self.update_velocity()
            self.create_new_population(i)
            self.evaluate_population(trainX, trainY, validateX, validateY, testX, testY)
            self.update_local_best_matrix()
            self.update_global_best_row()

            self.alpha = self.alpha - (0.17 / self.numOfGens)

            print('gen #' + str(i) + '\t\t min fit: ' + str(self.fitness.min()) + '\t\t avg fit: ' + str(np.average(self.fitness)))

    # -------------------------------------------------------------------------------------------------

    # evaluates the population and returns its fitnesses
    def evaluate_population(self, trainX, trainY, validateX, validateY, testX, testY):
        fittingStatus, self.fitness = self.fff.validate_model(self.model, self.fileW, self.population, trainX,
                                                                  trainY, validateX, validateY, testX, testY)

    # -------------------------------------------------------------------------------------------------

    # randomly selects 1.5% of features in a row
    def getAValidrow(self, eps=0.015):
        sum = 0
        while (sum < 3):
            row = np.zeros(self.numOfFea)
            for j in range(self.numOfFea):
                if (np.random.random() < eps):
                    row[j] = 1
            sum = row.sum()
        return row

    # -------------------------------------------------------------------------------------------------

    # the following creates an output file.
    def createAnOutputFile(self):
        file_name = None
        algorithm = self

        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if file_name is None and algorithm is not None:
            file_name = "{}_{}_gen{}_{}.csv".format(algorithm.__class__.__name__, algorithm.model.__class__.__name__,
                                                    algorithm.numOfGens, timestamp)
        elif file_name is None:
            file_name = "{}.csv".format(timestamp)

        fileW = csv.writer(file(file_name, 'wb'))
        fileW.writerow(['Descriptor ID', 'Fitness', 'Model','R2', 'Q2', 'R2Pred_Validation', 'R2Pred_Test'])

        return fileW

# --------------------------------------------------------------------------------------------------------------------

def main():
    np.random.seed()

    # BPSO parameters
    num_pop = 50
    num_feat = 385
    num_gens = 1000

    # initialize objects
    model = mlr.MLR()
    fdf = FromDataFileMLR.FromDataFileMLR()
    fff = FromFinessFileMLR.FromFinessFileMR(fdf)
    de_bpso = DE_BinaryParticleSwarmOptimization(model, fff, num_pop, num_feat, num_gens)

    # load in data from files
    trainX, trainY, validateX, validateY, testX, testY = fdf.getAllOfTheData()
    trainX, validateX, testX = fdf.rescaleTheData(trainX, validateX, testX)

    # DE_BPSO algorithm
    de_bpso.create_initial_velocity()
    de_bpso.create_initial_population()
    de_bpso.evaluate_population(trainX, trainY, validateX, validateY, testX, testY)
    de_bpso.create_initial_local_best_matrix()
    de_bpso.create_initial_global_best_row()
    de_bpso.evolve_population(trainX, trainY, validateX, validateY, testX, testY)

if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------------------------------------