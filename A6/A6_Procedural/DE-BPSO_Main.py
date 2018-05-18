# =====================================================================================================================
# Chris Banci
# CS512 - Data Mining
# 5/7/18
#
# Assignment 6 - Procedural
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
def create_initial_population(velocity, Lambda=0.01):
    population = np.zeros((numOfPop, numOfFea))

    for i in range(numOfPop):
        for j in range(numOfFea):
            if velocity[i][j] <= Lambda:
                population[i][j] = 1
            else:
                population[i][j] = 0

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
    beta = 0.004
    a,b,c,d = 0,0,0,0

    for i in range(numOfPop):
        for j in range(numOfFea):
            if (alpha < velocity[i][j]) and (velocity[i][j] <= p):
                new_population[i][j] = local_best_matrix[i][j]
                a += 1

            elif (p < velocity[i][j]) and (velocity[i][j] <= (1 - beta)):
                new_population[i][j] = global_best_row[j]
                b += 1

            elif ((1 - beta) < velocity[i][j]) and (velocity[i][j] <= 1):
                new_population[i][j] = 1 - oldPopulation[i][j]
                c += 1

            else:
                new_population[i][j] = oldPopulation[i][j]
                d += 1

    if current_gen == (numOfGens / 2):
        new_population = optimize(new_population)

    print(" - cases = [" + str(a) + ", " + str(b) + ", " + str(c) + ", " + str(d) + "]")

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
def update_velocity(velocity, population, F=0.7, CR=0.7):
    new_velocity = np.zeros((numOfFea, numOfFea))

    for i in range(numOfPop):
        random_set = np.random.choice(range(50), 3, replace=False)
        r1, r2, r3 = population[random_set]
        r = r3 + F * (r2 - r1)

        for j in range(numOfFea):
            # do cross validation between new_velocity and r
            if np.random.random() < CR:
                new_velocity[i][j] = r[j]
            else:
                new_velocity[i][j] = velocity[i][j]

    return new_velocity

# --------------------------------------------------------------------------------------------------------------------

# evolves the population a set number of times
def evolve_population(initial_population, initial_fitness, initial_velocity, initial_local_best_matrix, initial_local_fitness, model, fileW, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):
    print('gen #' + str(0) + '\t\t min fit: ' + str(initial_fitness.min()) + '\t\t avg fit: ' + str(np.average(initial_fitness)))
    alpha = 0.5

    for i in range(1, numOfGens):
        velocity = update_velocity(initial_velocity, initial_population)
        population = create_new_population(numOfPop, numOfFea, initial_population, velocity, initial_local_best_matrix, alpha, i)
        x, fitness = FromFinessFileMLR.validate_model(model, fileW, population, \
                                                    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
        local_best_matrix, local_fitness = update_local_best_matrix(population, fitness, initial_local_best_matrix, initial_local_fitness)
        update_global_best_row(local_best_matrix, local_fitness)

        alpha = alpha - (0.17 / numOfGens)

        print('gen #' + str(i) + '\t\t min fit: ' + str(fitness.min()) + '\t\t avg fit: ' + str(np.average(fitness)))

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

    # DE_BPSO algorithm
    velocity = create_initial_velocity()
    population = create_initial_population(velocity, Lambda=0.01)
    x, fitness = FromFinessFileMLR.validate_model(model,fileW, population, \
                                        TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    local_best_matrix, local_fitness = create_initial_local_best_matrix(population, fitness)
    create_initial_global_best_row(local_best_matrix, local_fitness)

    evolve_population(population, fitness, velocity, local_best_matrix, local_fitness, \
                                        model, fileW, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------------------------------------
