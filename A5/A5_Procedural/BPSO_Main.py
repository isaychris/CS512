# =====================================================================================================================
# Chris Banci
# CS512 - Data Mining
# 4/26/18
#
# Assignment 5 - OOP
# =====================================================================================================================
import time
import numpy as np
import csv
import mlr
import FromDataFileMLR
import FromFinessFileMLR

# GLOBALS
numOfPop = 50
numOfFea = 385
numOfGens = 1000
global_best_row = np.zeros(numOfFea)
global_best_row_fitness = 0.00

# --------------------------------------------------------------------------------------------------------------------

# creates an initial population by randomly selecting 1.5% features for each row.
def create_initial_population():
    population = np.zeros((numOfPop, numOfFea))

    for i in range(numOfPop):
        row = getAValidrow()
        population[i] = row

    return population

# --------------------------------------------------------------------------------------------------------------------

# creates the initial velocity matrix by setting each element in matrix to a random number between 0 and 1
def create_initial_velocity():
    velocity = np.zeros((numOfPop, numOfFea))
    for i in range(numOfPop):
        for j in range(numOfFea):
            velocity[i][j] = np.random.random()

    return velocity

# --------------------------------------------------------------------------------------------------------------------

# creates the initial best matrix by copying the initial population.
def create_initial_local_best_matrix(population, fitness):
    local_best_matrix = population
    local_fitness = fitness

    return local_best_matrix, local_fitness

# --------------------------------------------------------------------------------------------------------------------

# creates the initial global best row by sorting initial population by its fitness and copying its best row.
def create_initial_global_best_row(local_best_matrix, local_fitness):
    global global_best_row
    global global_best_row_fitness

    idx = local_fitness.argmin(axis=0)
    global_best_row = local_best_matrix[idx]
    global_best_row_fitness = local_fitness[idx]

# --------------------------------------------------------------------------------------------------------------------

# creates a new population by comparing velocity conditional statements
def create_new_population(numOfPop, numOfFea, oldPopulation, velocity, local_best_matrix, alpha, current_gen):
    new_population = np.zeros((numOfPop, numOfFea))

    p = 0.5 * (1 + alpha)
    a,b,c,d = 0,0,0,0

    for i in range(numOfPop):
        for j in range(numOfFea):
            if velocity[i][j] <= alpha:
                new_population[i][j] = oldPopulation[i][j]
                a += 1
            elif alpha < velocity[i][j] <= p:
                new_population[i][j] = local_best_matrix[i][j]
                b += 1
            elif p < velocity[i][j] <= 1:
                new_population[i][j] = global_best_row[j]
                c += 1
            else:
                new_population[i][j] = oldPopulation[i][j]
                d += 1

    # optimize new_population at gen 500 by making half random.
    if current_gen == (numOfGens / 2):
        new_population = optimize(new_population)

    print(" - choices = [" + str(a) + ", " + str(b) + ", " + str(c) + ", " + str(d) + "]")

    return new_population

# --------------------------------------------------------------------------------------------------------------------

# optimizes the new population at gen 500 by making half of the population random.
def optimize(population):
    print(" -- optimizing population")
    for i in range((numOfPop / 2), numOfPop):
        row = getAValidrow()
        population[i] = row

    return population

# --------------------------------------------------------------------------------------------------------------------

# best row from local best matrix becomes global best row
def update_global_best_row(local_best_matrix, local_fitness):
    global global_best_row
    global global_best_row_fitness

    idx = local_fitness.argmin(axis=0)

    if local_fitness[idx] < global_best_row_fitness:
        global_best_row = local_best_matrix[idx]
        global_best_row_fitness = local_fitness[idx]
        print(' - found new global best row')

# --------------------------------------------------------------------------------------------------------------------

# local best matrix updated from comparing local fitness and fitness
def update_local_best_matrix(population, fitness, local_best_matrix, local_fitness):
    for i in range(numOfPop):
        if fitness[i] < local_fitness[i]:
            local_best_matrix[i] = population[i]

            # update the local fitness since local best matrix was changed
            local_fitness[i] = fitness[i]

    return local_best_matrix, local_fitness

# --------------------------------------------------------------------------------------------------------------------

# velocity matrix is updated through formula.
def update_velocity(velocity, population, local_best_matrix, global_best_row, c1=2, c2=2, inertia=0.9):
    new_velocity = np.zeros((numOfFea, numOfFea))

    for i in range(numOfPop):
        for j in range(numOfFea):
            term1 = c1 * np.random.random() * (local_best_matrix[i][j] - population[i][j])
            term2 = c2 * np.random.random() * (global_best_row[j] - population[i][j])
            new_velocity[i][j] = (inertia * velocity[i][j]) + term1 + term2

    return new_velocity

# --------------------------------------------------------------------------------------------------------------------

# evolves the population a set number of times
def evolve_population(init_population, init_fitness, init_velocity, init_local_best_matrix, init_local_fitness, model, fileW, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):
    print('gen #' + str(0) + '\t\t min fit: ' + str(init_fitness.min()) + '\t\t avg fit: ' + str(np.average(init_fitness)))
    alpha = 0.5

    for i in range(1, numOfGens):
        population = create_new_population(numOfPop, numOfFea, init_population, init_velocity, init_local_best_matrix, alpha, i)
        fitness = evaluate_population(model, fileW, population, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
        local_best_matrix, local_fitness = update_local_best_matrix(population, fitness, init_local_best_matrix, init_local_fitness)
        update_global_best_row(local_best_matrix, local_fitness)
        init_velocity = update_velocity(init_velocity, init_population, local_best_matrix, global_best_row)

        alpha = alpha - (0.17 / numOfGens)

        print('gen #' + str(i) + '\t\t min fit: ' + str(fitness.min()) + '\t\t avg fit: ' + str(np.average(fitness)))

# --------------------------------------------------------------------------------------------------------------------

# evaluates the population and returns its fitnesses
def evaluate_population(model, fileW, population, trainX, trainY, validateX, validateY, testX, testY):
    fittingStatus, fitness = FromFinessFileMLR.validate_model(model, fileW, population, trainX,
                                                              trainY, validateX, validateY, testX, testY)
    return fitness
# --------------------------------------------------------------------------------------------------------------------

# randomly selects 1.5% of features in a row
def getAValidrow(eps=0.015):
    sum = 0
    while (sum < 3):
        row = np.zeros(numOfFea)
        for j in range(numOfFea):
            if (np.random.random() < eps):
                row[j] = 1
        sum = row.sum()
    return row

# --------------------------------------------------------------------------------------------------------------------

# the following creates an output file.
def createAnOutputFile():
    file_name = None
    algorithm = None

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if file_name is None and algorithm is not None:
        file_name = "{}_{}_gen{}_{}.csv".format(algorithm.__class__.__name__, algorithm.model.__class__.__name__,
                                                algorithm.numOfGens, timestamp)
    elif file_name is None:
        file_name = "{}.csv".format(timestamp)

    fileW = csv.writer(file(file_name, 'wb'))
    fileW.writerow(['Descriptor ID', 'Fitness', 'Model', 'R2', 'Q2', 'R2Pred_Validation', 'R2Pred_Test'])

    return fileW

# --------------------------------------------------------------------------------------------------------------------

def main():
    np.random.seed()

    # initialize objects
    fileW = createAnOutputFile()
    model = mlr.MLR()

    # load in data from files
    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileMLR.getAllOfTheData()
    TrainX, ValidateX, TestX = FromDataFileMLR.rescaleTheData(TrainX, ValidateX, TestX)

    # BPSO algorithm
    init_population = create_initial_population()
    init_fitness = evaluate_population(model, fileW, init_population, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
    init_velocity = create_initial_velocity()
    init_local_best_matrix, init_local_fitness = create_initial_local_best_matrix(init_population, init_fitness)
    create_initial_global_best_row(init_local_best_matrix, init_local_fitness)
    evolve_population(init_population, init_fitness, init_velocity, init_local_best_matrix, init_local_fitness, \
                                        model, fileW, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------------------------------------
