import numpy as np

class MLR:
    """Multiple Linear Regression"""
    
    def __init__(self):
        """Initialization"""
        self.coef = None
    
    def fit(self, x_set, y_set):
        """Fit to training X and Y arrays"""
        # Add a column of 1's for the intercept
        x_set = np.append(np.ones((x_set.shape[0], 1)), x_set, axis=1)
        self.coef = np.linalg.lstsq(x_set, y_set, rcond=None)[0]
        return 'MLR'
    
    def predict(self, x_set):
        """Predict a Y from an X, object must already be fitted."""
        # matrix multiplication of X appended with a column of 1's (for
        # intercepts) and the coeficients
        if len(x_set.shape) == 1:
            x_set = np.reshape(x_set, (1, x_set.shape[0]))
        x_set = np.append(np.ones((x_set.shape[0], 1)), x_set, axis=1)
        return np.dot(x_set, self.coef)

    def printing(self):
        """Predict a Y from an X, object must already be fitted."""
        print('How are you doing?')
