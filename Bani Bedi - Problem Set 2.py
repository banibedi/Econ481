# Exercise 0

def github() -> str:
    """
    This function will return Bani Bedi's GitHub page for Problem Set 2.
    """

    return "https://github.com/banibedi/Econ481.git"

# Exercise 1

import numpy as np

def simulate_data(seed: int) -> tuple: # Defining the function 'simulate_data' that contains integers and will return a tuple
    """
    This function returns 1000 simulated observations using this process: y = 5 + 3x_{i1} + 2x_{i2} + 6x_{i3} +  y_i epsilon_i.
    The independent variables are drawn from a normal distribution with a mean of 0 and a standard deviation of 2.
    The error term is drawn from a normal distribution with a mean of 0 and a standard deviation of 1. 
    """
    
    # Set the seed to regenerate results for random processes

    np.random.seed(seed = 481) # Set the 481 like stated in the assignment

    # Generate the independent variables (x_{i1}. x_{i2}. x_{i3})

    x = np.random.normal(0, 2, size = (1000, 3)) # The equation is standard normal; the mean is 0 and the standard deviation is 2; this array is 1000 x 3

    # Generate the error term

    E = np.random.normal(0, 1, size = (1000, 1)) # This equation is standard normal; the mean is 0 and standard deviation is 1; the array is 1000 x 1

    # Calculate 'y' value of the equation (for all 1000 observations)

    y = 5 + 3*x[:, 0] + 2*x[:,1] + 6*x[:, 2] + E # Rewrite equation in python; this takes all 1000 values for the corresponding column and multiply it by the corresponding value (i.e. takes all 1000 values of the first column of x_{i1} and mutiplies it by 3); add error term in model too

    # Reshape y to be a 1000 x 1 array

    y = y.reshape(-1, 1) # Change shape of array using reshape from 1D to 2D; ; since we do not know / want to explictly tell the dimension of that axis, we reshape with (-1, 1) so the array only has 1 column

    return y, x # Return a tuple containing the array of y values and a matrix of x values

# Call the function simulate_data

simulate_data(seed = 481) # Seed is set to 481 like stated in the assignment

# Exercise 2

import numpy as np 
# Since this is a closed-form function, scipy.optimize isn't necessarily needed

def estimate_mle(y: np.array, X: np.array) -> np.array: # Defines function 'estimate_mle' and states that 'y' and 'X' are arrays and the result will also be an array
    """
    This function estimates the MLE parameters (B_hat{MLE}) where the assumed model is y{1} = B{0} + B{1}x_{i1} + B{2}x_{i2} + B{3}x_{i3} + E{i}.
    The error term E{i} is normally distributed with a mean of 0 and a standard error of 1.
    It takes a 1000 x 1 'y' array and 1000 x 3 'x' array and returns a 4 x 1 array with coefficients B{0}, B{1}, B{2}, and B{3} in that order.
    """

    # Note: If error term is normally distributed, the MLE approach will obtain the same estimates as those obtained from OLS 

    # Add the intercept function B{0}

    X_int = np.hstack((np.ones((X.shape[0], 1)), X)) # Defining the intercept B{0}; np.ones creates a vector of 1000 x 1 (aka the number of rows in the matrix), and every element in this column is 1; hstack puts the two matrices and sticks them horizontally 
        # Each row is an observation with the first number being '1'

    # Compute the OLS estimates for regression coefficients

    beta_hat = np.linalg.inv(X_int.T @ X_int) @ X_int.T @ y # Transposes X_int and multiply both X_int and X_int.T; calculate the inverse of the matrix with np.linalg.inv; multiply again by X_int.T; multiply again by y

    # Return the parameter estimates    
        
    return beta_hat.reshape(-1, 1) # Like the previous question, reshape the array to calculate the number of rows in beta_hat (hence the -1) with one column

# Test

# Setting the seed

np.random.seed(481) # Set default seed to ensure the same random values for consistency

# Simulate the independent variables (X)

X = np.random.normal(0, 1, (1000, 3)) # Set the X variable to a normal distribution with a mean of 0 and a standard deviation of 1; the array will be 1000 x 3

# Define all beta values

beta_values = np.array([6, 3, 2, 4]) # Create sample beta values

# Simulate error term

E = np.random.normal(0, 1, (1000, 1)) # Set the error term to a normal distrubution with a mean of 0 and a standard deviation of 1; the array will be 1000 x 1; this is based on given information from the assignment

# Calculate 'y' variable

y = np.hstack((np.ones((1000, 1)), X)) @ beta_values[:, None] + E # Calculates 'y' similar to prior problem; horizontally stacks the 1000 x 1 columns with matrix X; then multiplies with beta values; adds error term after

# Estimate MLE Parameters

estimated_parameters = estimate_mle(y, X) # This will return the estimate of the parameters of the model based on simulated 'y' and 'X'
print(estimated_parameters) # Return estiamted paramters in 4 x 1 format like assignment stated

# Exercise 3

import numpy as np 
from scipy.optimize import minimize # Minimize RSS using scipy.optimize.minimize
    
def estimate_ols(y: np.array, X: np.array) -> np.array: # Defines the 'estimate_ols' function, stating that both the 'X' and 'y' variables are an array and the result will also be an array
    """
    This function estinates the OLS coefficients for the simulated data without using the closed form solution.
    It takes a 1000 x 1 'y' array and 1000 x 3 'x' array and returns a 4 x 1 array with coefficients B{0}, B{1}, B{2}, and B{3} in that order.
    """

# Use an optimization technique to minimize the residual sum of squares (sum of squared of observed values vs. values predicted by the model)
    
# Define cost function that calculates residual sum of squares
    # The goal is to find a set of beta coefficients that makes our model's predictions as close to the observed data as possible (RSS)

    def cost_function(beta, X, y): # Defines the 'cost function' that includes three arguments: beta, X, and y
        y_estimated = X @ beta # Calculates the predicted value of 'y' by  multiplying 'X' by 'beta'; 
        residuals = y - y_estimated # After obtaining the predicted values, calculate the difference between the observed values and values generated by the model
        sum_sq = np.sum(residuals ** 2) # Penalizes larger errors more severely than smaller ones; they are then summed up to represent the total error of the model's predictions
        return sum_sq
        
# Use scipy.optimize functions to find parameters that minimize this cost function
    
    X_int = np.hstack((np.ones((X.shape[0], 1)), X)) # Defines X-intercept; takes 1000 x 1 matrix, includes 1 as the first number, and stacks it up horizontally; then makes the array as big as 'X"
    beta_i = np.zeros(X_int.shape[1]) # Initializes starting point with vector of zeros; includes intercept and slope of each variable
    result = minimize(fun=cost_function, x0 = beta_i, args=(X_int, y)) # Minimizes cost function; sets initial guess of coefficients; provides both variable arguments
    return result.x.reshape(-1, 1)    

# Test

# Setting the seed

np.random.seed(481) # Set default seed to ensure the same random values for consistency

# Simulate the independent variables (X)

X = np.random.normal(0, 1, (1000, 3)) # Set the X variable to a normal distribution with a mean of 0 and a standard deviation of 1; the array will be 1000 x 3

# Sample values

sample_coef = np.array([1, 2, 3, 4])  # Sample coefficient
intercept_column = np.ones((1000, 1))  # Intercept term
X_with_intercept = np.hstack((intercept_column, X))  # Add intercept term to the matrix of independent variables
y = X_with_intercept @ sample_coef + np.random.normal(0, 1, (1000, 1))  # 'y' variable

# Result

estimated_parameters = estimate_ols(y, X)
print(estimated_parameters) # Result is very close to actual values (success!)
