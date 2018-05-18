# =====================================================================================================================
# Chris Banci
# CS512 - Data Mining
# 4/29/18
#
# Assignment 5
# =====================================================================================================================
import time
import numpy as np
import csv
import mlr
import FromDataFileMLR
import FromFinessFileMLR


class BinaryParticleSwarmOptimization:
    def __init__(self, model, fff, pop, fea, gen):
        self.numOfPop = pop
        self.numOfFea = fea
        self.numOfGens = gen

        self.model = model
        self.population = np.zeros((self.numOfPop, self.numOfFea))
        self.fitness = np.zeros(self.numOfPop)
        self.alpha = 0.5
        self.global_best_row = np.zeros(self.numOfFea)
        self.global_best_row_fitness = 0.00
        self.velocity = np.zeros((self.numOfPop, self.numOfFea))
        self.fff = fff
        self.fileW = self.createAnOutputFile()

    # -------------------------------------------------------------------------------------------------

    # creates an initial population by randomly selecting 1.5% features for each row.
    def create_initial_population(self):
        np.random.seed()

        for i in range(self.numOfPop):
            row = self.getAValidrow()
            self.population[i] = row

    # -------------------------------------------------------------------------------------------------

    # creates the initial velocity matrix by setting each element in matrix to a random number between 0 and 1
    def create_initial_velocity(self):
        for i in range(self.numOfPop):
            for j in range(self.numOfFea):
                self.velocity[i][j] = np.random.random()
    # -------------------------------------------------------------------------------------------------

    # creates the initial best matrix by copying the initial population.
    def create_initial_local_best_matrix(self):
        local_best_matrix = self.population
        local_fitness = self.fitness
        return local_best_matrix, local_fitness
    # -------------------------------------------------------------------------------------------------

    # creates the initial global best row by sorting initial population by its fitness and copying its best row.
    def create_initial_global_best_row(self):
        idx = self.fitness.argmin(axis=0)
        self.global_best_row = self.population[idx]
        self.global_best_row_fitness = self.fitness[idx]

    # -------------------------------------------------------------------------------------------------

    # creates a new population by comparing velocity conditional statements
    def create_new_population(self, local_best_matrix, current_gen):
        new_population = np.zeros((self.numOfPop, self.numOfFea))

        p = 0.5 * (1 + self.alpha)
        a,b,c,d = 0,0,0,0

        for i in range(self.numOfPop):
            for j in range(self.numOfFea):
                if self.velocity[i][j] <= self.alpha:
                    new_population[i][j] = self.population[i][j]
                    a += 1
                elif self.alpha < self.velocity[i][j] <= p:
                    new_population[i][j] = local_best_matrix[i][j]
                    b += 1
                elif p < self.velocity[i][j] <= 1:
                    new_population[i][j] = self.global_best_row[j]
                    c += 1
                else:
                    new_population[i][j] = self.population[i][j]
                    d += 1

        # optimize new_population at gen 500 by making half random.
        if current_gen == (self.numOfGens / 2):
            new_population = self.optimize(new_population)

        self.population = new_population
        print(" - choices = [" + str(a) + ", " + str(b) + ", " + str(c) + ", " + str(d) + "]")

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
    def update_global_best_row(self, local_best_matrix, local_fitness):
        idx = local_fitness.argmin(axis=0)

        if local_fitness[idx] < self.global_best_row_fitness:
            print(" -- found new global best row")
            self.global_best_row = local_best_matrix[idx]
            self.global_best_row_fitness = local_fitness[idx]

    # -------------------------------------------------------------------------------------------------

    # local best matrix updated from comparing local fitness and fitness
    def update_local_best_matrix(self, local_best_matrix, local_fitness):
        for i in range(self.numOfPop):
            if self.fitness[i] < local_fitness[i]:
                local_best_matrix[i] = self.population[i]

                # update the local fitness since local best matrix was changed
                local_fitness[i] = self.fitness[i]

        return local_best_matrix, local_fitness

    # -------------------------------------------------------------------------------------------------

    # velocity matrix is updated through formula.
    def update_velocity(self, local_best_matrix, c1=2, c2=2, inertia=0.9):
        for i in range(self.numOfPop):
            for j in range(self.numOfFea):
                term1 = c1 * np.random.random() * (local_best_matrix[i][j] - self.population[i][j])
                term2 = c2 * np.random.random() * (self.global_best_row[j] - self.population[i][j])
                self.velocity[i][j] = (inertia * self.velocity[i][j]) + term1 + term2

    # -------------------------------------------------------------------------------------------------

    # evolves the population a set number of times
    def evolve_population(self, local_best_matrix, local_fitness, trainX, trainY, validateX, validateY, testX, testY):
        print('gen #' + str(0) + '\t\t min fit: ' + str(self.fitness.min()) + '\t\t avg fit: ' + str(np.average(self.fitness)))

        for i in range(1, self.numOfGens):

            self.create_new_population(local_best_matrix, i)
            self.evaluate_population(trainX, trainY, validateX, validateY, testX, testY)
            local_best_matrix, local_fitness = self.update_local_best_matrix(local_best_matrix, local_fitness)
            self.update_global_best_row(local_best_matrix, local_fitness)
            self.update_velocity(local_best_matrix)

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
    # BPSO parameters
    num_pop = 50
    num_feat = 385
    num_gens = 1000

    # initialize objects
    model = mlr.MLR()
    fdf = FromDataFileMLR.FromDataFileMLR()
    fff = FromFinessFileMLR.FromFinessFileMR(fdf)
    bpso = BinaryParticleSwarmOptimization(model, fff, num_pop, num_feat, num_gens)

    # load in data from files
    trainX, trainY, validateX, validateY, testX, testY = fdf.getAllOfTheData()
    trainX, validateX, testX = fdf.rescaleTheData(trainX, validateX, testX)

    # BPSO algorithm
    bpso.create_initial_population()
    bpso.evaluate_population(trainX, trainY, validateX, validateY, testX, testY)
    bpso.create_initial_velocity()
    initial_local_best_matrix, initial_local_fitness = bpso.create_initial_local_best_matrix()
    bpso.create_initial_global_best_row()
    bpso.evolve_population(initial_local_best_matrix, initial_local_fitness, trainX, trainY, validateX, validateY, testX, testY)

if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------------------------------------