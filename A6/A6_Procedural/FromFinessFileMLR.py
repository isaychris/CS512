from numpy import *
import hashlib
import FromDataFileMLR

# ------------------------------------------------------------------------------


def r2(y, yHat):
    # Coefficient of determination
    numer = ((y - yHat) ** 2).sum()  # Residual Sum of Squares
    denom = ((y - y.mean()) ** 2).sum()
    r2 = 1 - numer / denom
    return r2

# ------------------------------------------------------------------------------

def r2Pred(yTrain, yTest, yHatTest):
    numer = ((yHatTest - yTest) ** 2).sum()
    denom = ((yTest - yTrain.mean()) ** 2).sum()
    r2Pred = 1 - numer / denom
    return r2Pred


# ------------------------------------------------------------------------------

def cv_predict(model, set_x, set_y):
    # Predict using cross validation.
    yhat = empty_like(set_y)
    for idx in range(0, yhat.shape[0]):
        train_x = delete(set_x, idx, axis=0)
        train_y = delete(set_y, idx, axis=0)
        modelName = model.fit(train_x, train_y)
        yhat[idx] = model.predict(set_x[idx])
    return yhat


# ------------------------------------------------------------------------------
# Ahmad Hadaegh: Modified  on: July 16, 2013

def calc_fitness(xi, Y, Yhat, c=2):
    """
    Calculate fitness of a prediction.
    xi : array_like -- Mask of features to measure fitness of. Must be of dtype bool.
    c : float       -- Adjustment parameter.
    """

    p = sum(xi)  # Number of selected parameters
    n = len(Y)  # Sample size
    numer = ((Y - Yhat) ** 2).sum() / n  # Mean square error
    pcn = p * (c / n)
    if pcn >= 1:
        return 1000
    denom = (1 - pcn) ** 2
    theFitness = numer / denom
    return theFitness


# ------------------------------------------------------------------------------
# Ahmad Hadaegh: Modified  on: July 16, 2013
def InitializeTracks():
    trackDesc = {}
    trackFitness = {}
    trackModel = {}
    trackR2 = {}
    trackQ2 = {}
    trackR2PredValidation = {}
    trackR2PredTest = {}

    return trackDesc, trackFitness, trackModel, trackR2, trackQ2, \
           trackR2PredValidation, trackR2PredTest


# ------------------------------------------------------------------------------
# Ahmad Hadaegh: Modified  on: July 16, 2013
def initializeYDimension():
    yTrain = {}
    yHatTrain = {}
    yHatCV = {}
    yValidation = {}
    yHatValidation = {}
    yTest = {}
    yHatTest = {}
    return yTrain, yHatTrain, yHatCV, yValidation, yHatValidation, yTest, yHatTest


# ------------------------------------------------------------------------------
def OnlySelectTheOnesColumns(popI):
    numOfFea = popI.shape[0]
    xi = zeros(numOfFea)
    for j in range(numOfFea):
        xi[j] = popI[j]

    xi = xi.nonzero()[0]
    xi = xi.tolist()
    return xi


# ------------------------------------------------------------------------------

def validate_model(model, fileW, population, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):
    numOfPop = population.shape[0]
    fitness = zeros(numOfPop)
    c = 2
    false = 0
    true = 1
    predictive = false

    trackDesc, trackFitness, trackModel, trackR2, trackQ2, \
    trackR2PredValidation, trackR2PredTest = InitializeTracks()

    yTrain, yHatTrain, yHatCV, yValidation, \
    yHatValidation, yTest, yHatTest = initializeYDimension()

    unfit = 1000
    itFits = 1
    for i in range(numOfPop):
        xi = OnlySelectTheOnesColumns(population[i])

        idx = hashlib.sha1(array(xi)).digest()

        X_train_masked = TrainX.T[xi].T

        X_validation_masked = ValidateX.T[xi].T
        X_test_masked = TestX.T[xi].T

        try:
            model_desc = model.fit(X_train_masked, TrainY)
        except:
            return unfit, fitness

        # Computed predicted values
        Yhat_cv = cv_predict(model, X_train_masked, TrainY)  # Cross Validation
        Yhat_validation = model.predict(X_validation_masked)
        Yhat_test = model.predict(X_test_masked)

        # Compute R2 statistics (Prediction for Valiation and gpx-streetview set)
        q2_loo = r2(TrainY, Yhat_cv)
        q2_loo = FromDataFileMLR.getTwoDecPoint(q2_loo)

        r2pred_validation = r2Pred(TrainY, ValidateY, Yhat_validation)
        r2pred_validation = FromDataFileMLR.getTwoDecPoint(r2pred_validation)

        r2pred_test = r2Pred(TrainY, TestY, Yhat_test)
        r2pred_test = FromDataFileMLR.getTwoDecPoint(r2pred_test)

        Y_fitness = append(TrainY, ValidateY)
        Yhat_fitness = append(Yhat_cv, Yhat_validation)

        fitness[i] = calc_fitness(xi, Y_fitness, Yhat_fitness, c)

        if predictive and ((q2_loo < 0.5) or (r2pred_validation < 0.5) or (r2pred_test < 0.5)):
            # if it's not worth recording, just return the fitness
            print("ending the program because of predictive is: ", predictive)
            continue

        # Compute predicted Y_hat for training set.
        Yhat_train = model.predict(X_train_masked)
        r2_train = r2(TrainY, Yhat_train)

        idxLength = len(xi)

        # store stats
        trackDesc[idx] = str(xi)

        trackFitness[idx] = FromDataFileMLR.getTwoDecPoint(fitness[i])

        trackModel[idx] = model_desc

        trackR2[idx] = FromDataFileMLR.getTwoDecPoint(r2_train)
        trackQ2[idx] = FromDataFileMLR.getTwoDecPoint(q2_loo)
        trackR2PredValidation[idx] = FromDataFileMLR.getTwoDecPoint(r2pred_validation)
        trackR2PredTest[idx] = FromDataFileMLR.getTwoDecPoint(r2pred_test)

        yTrain[idx] = TrainY.tolist()

        yHatTrain[idx] = Yhat_train.tolist()
        for i in range(len(yHatTrain[idx])):
            yHatTrain[idx][i] = FromDataFileMLR.getTwoDecPoint(yHatTrain[idx][i])

        yHatCV[idx] = Yhat_cv.tolist()
        for i in range(len(yHatCV[idx])):
            yHatCV[idx][i] = FromDataFileMLR.getTwoDecPoint(yHatCV[idx][i])

        yValidation[idx] = ValidateY.tolist()

        yHatValidation[idx] = Yhat_validation.tolist()
        for i in range(len(yHatValidation[idx])):
            yHatValidation[idx][i] = FromDataFileMLR.getTwoDecPoint(yHatValidation[idx][i])

        yTest[idx] = TestY.tolist()

        yHatTest[idx] = Yhat_test.tolist()
        for i in range(len(yHatTest[idx])):
            yHatTest[idx][i] = FromDataFileMLR.getTwoDecPoint(yHatTest[idx][i])

    write(model, fileW, trackDesc, trackFitness, trackModel, trackR2, \
          trackQ2, trackR2PredValidation, trackR2PredTest)

    return itFits, fitness


# ------------------------------------------------------------------------------

def write(model, fileW, trackDesc, trackFitness, trackModel, trackR2, \
          trackQ2, trackR2PredValidation, trackR2PredTest):
    for key in trackFitness.keys():
        fileW.writerow([trackDesc[key], trackFitness[key], trackModel[key], \
                        trackR2[key], trackQ2[key], trackR2PredValidation[key], trackR2PredTest[key]])

    # fileOut.close()

# ------------------------------------------------------------------------------

