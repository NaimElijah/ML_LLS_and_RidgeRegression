import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

##############################################    Actual ML regression functions    ##############################################

def least_squares_pinv(X, y):  # n
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def ridge_regression(X, y, lambdaa):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lambdaa * np.eye(d), X.T @ y)

##############################################   ML Regression Function's Experiments   ##############################################

# Linear Least Squares

def LeastSquaresExperiment100to500(X, y):
    ms = range(100, 501, 10)
    train_losses = []
    test_losses = []

    np.random.seed(0)

    for m in ms:
        idx = np.random.choice(len(X), m, replace=False)
        X_m = X[idx]
        y_m = y[idx]

        w_ls = least_squares_pinv(X_m, y_m)

        train_losses.append(mse(X_m, y_m, w_ls))
        test_losses.append(mse(X_test, Y_test, w_ls))

    # Plotting
    plt.figure()
    plt.plot(ms, test_losses)
    plt.xlabel("Training size m")
    plt.ylabel("Test MSE")
    plt.title("Least Squares: Test Loss vs Training Size")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(ms, train_losses)
    plt.xlabel("Training size m")
    plt.ylabel("Training MSE")
    plt.title("Least Squares: Training Loss vs Training Size")
    plt.grid(True)
    plt.show()






# Ridge Regression

def RidgeRegressionExperiment60(X, y, m, lambdas):
    np.random.seed(0)

    idx = np.random.choice(len(X), m, replace=False)
    X_m = X[idx]
    y_m = y[idx]

    test_losses_60 = []

    for lam in lambdas:
        if lam == 0:
            w = least_squares_pinv(X_m, y_m)
        else:
            w = ridge_regression(X_m, y_m, lam)

        test_losses_60.append(mse(X_test, Y_test, w))

    # Plotting
    plt.figure()
    plt.plot(lambdas, test_losses_60, marker='o', label="Ridge")
    plt.axhline(test_losses_60[0], linestyle='--', label="Least Squares")
    plt.xlabel("λ")
    plt.ylabel("Test MSE")
    plt.title("Ridge Regression: Test Loss vs λ (m = 60)")
    plt.legend()
    plt.grid(True)
    plt.show()




def RidgeRegressionExperiment500(X, y, m, lambdas):
    idx = np.random.choice(len(X), m, replace=False)
    X_m = X[idx]
    y_m = y[idx]

    test_losses_500 = []

    for lam in lambdas:
        if lam == 0:
            w = least_squares_pinv(X_m, y_m)
        else:
            w = ridge_regression(X_m, y_m, lam)

        test_losses_500.append(mse(X_test, Y_test, w))

    # Plotting
    plt.figure()
    plt.plot(lambdas, test_losses_500, marker='o', label="Ridge")
    plt.axhline(test_losses_500[0], linestyle='--', label="Least Squares")
    plt.xlabel("λ")
    plt.ylabel("Test MSE")
    plt.title("Ridge Regression: Test Loss vs λ (m = 500)")
    plt.legend()
    plt.grid(True)
    plt.show()




# Helper functions
def mse(X, y, w):
    return np.mean((X @ w - y) ** 2)

# main function
if __name__ == '__main__':
    data = sio.loadmat('lsdata.mat')
    X, Y = data['X'], data['Y']
    X_test, Y_test = data['Xtest'], data['Ytest']

    lambdas = [0, 0.01, 0.02, 0.05, 0.1, 1, 10, 15]

    # data = sio.loadmat('lsdata.mat')
    # X = data['X']  # (n_train, d)
    # Y = data['Y'].ravel()
    # X_test = data['Xtest']
    # Y_test = data['Ytest'].ravel()

    LeastSquaresExperiment100to500(X, Y)
    RidgeRegressionExperiment60(X, Y, 60, lambdas)
    RidgeRegressionExperiment500(X, Y, 500, lambdas)



