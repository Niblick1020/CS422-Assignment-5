# ~ in order to get this to work in linix ubuntu I hade to do the following.

# ~ sudo apt install python3-venv
# ~ python3 -m venv env
# ~ source env/bin/activate
# ~ pip install sklearn
# ~ Note: I dont know if this step is needed
# ~ pip3 install -U scikit-learn
# ~ deactivate

import random
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

def runData(dataFrame):

     # Create an empty list to store predicted labels
    y_pred = []

    # Iterate through test data to make predictions
    for i in range(len(dataFrame)):
        # Predict the label for each test instance and append to y_pred
        y_pred.insert(len(y_pred), clf.predict([[dataFrame.iloc[i, 0], dataFrame.iloc[i, 1], dataFrame.iloc[i, 2],
                                                 dataFrame.iloc[i, 3], dataFrame.iloc[i, 4], dataFrame.iloc[i, 5],
                                                  dataFrame.iloc[i, 6], dataFrame.iloc[i, 7], dataFrame.iloc[i, 8],
                                                  dataFrame.iloc[i, 9], dataFrame.iloc[i, 10]]])[0])
        
        # vv For when we remove the binary input features (for testing) vv
            #y_pred.insert(len(y_pred), clf.predict([[dataFrame.iloc[i, 0], dataFrame.iloc[i, 1], dataFrame.iloc[i, 2],
		    # dataFrame.iloc[i, 3], dataFrame.iloc[i, 4], dataFrame.iloc[i, 5],
		    # dataFrame.iloc[i, 6], dataFrame.iloc[i, 7], dataFrame.iloc[i, 8],
		    # dataFrame.iloc[i, 9]]])[0])

    # Prepare the test data for model evaluation
    temp = dataFrame.drop(output_name, axis=1)
    X_test = np.array(temp.values.tolist())
    y_true = dataFrame[output_name].tolist()

    # Display relevant evaluation metrics
    print(f'MSE:    {mean_squared_error(y_true, y_pred)}')
    print(f'MAE:    {mean_absolute_error(y_true, y_pred)}')
    print(f'r2:    {r2_score(y_true, y_pred)}')


if __name__ == '__main__':
    # Read the dataset from a CSV file
    df = pd.read_csv('ParisHousing.csv')

    # Generate a random number for data splitting
    random_seed = random.randint(0, 9999)

    # Remove rows with any empty column
    df = df.dropna()
    
    # vv For when removing the binary input features (for testing) vv
        #df = df.drop('hasYard', axis=1)
        #df = df.drop('hasPool', axis=1)
        #df = df.drop('isNewBuilt', axis=1)
        #df = df.drop('hasStormProtector', axis=1)
        #df = df.drop('hasStorageRoom', axis=1)
        #df = df.drop('hasGuestRoom', axis=1)
    
    # Store the name of the output column
    output_name = 'price'

    # Display the entire dataframe
    print(f'\n\nEntire dataframe: \n{df}')

    # Randomly choose 20% of the rows for the test data
    testData = df.sample(frac=0.2, random_state=random_seed)

    # Display the test data
    print(f'\n\nTest Data: \n{testData}')

    # Remove the test data to create the training data (80%)
    trainData = df.drop(testData.index)
    
    # Create a deep copy of the test and training data for OLS
    OLS_testData = testData.copy(deep=True)
    OLS_trainData = trainData.copy(deep=True)
    
    # Insert an input column at the beginning of the data with a constant value 1
    OLS_testData.insert(0, 'x0', 1)
    OLS_trainData.insert(0, 'x0', 1)
    
    # Separating the test data into inputs x and outputs y, and converting them into matrices
    OLS_testData_y = np.array(OLS_testData[output_name].tolist())
    OLS_testData_x = np.array((OLS_testData.drop(output_name, axis=1)).values.tolist())
    
    # Separating the training data into inputs x and outputs y, and converting them into matrices
    OLS_trainData_y = np.array(OLS_trainData[output_name].tolist())
    OLS_trainData_x = np.array((OLS_trainData.drop(output_name, axis=1)).values.tolist())
    
    # Obtain the w vector by multiplying the matrices obtained above following the formula provided in the class notes
    OLS_w_vector = np.matmul(np.matmul(np.linalg.inv(np.matmul(OLS_trainData_x.transpose(), OLS_trainData_x)), OLS_trainData_x.transpose()), OLS_trainData_y)
    
    # Get the predicted output y values for both the training data nad test data, respectively
    OLS_trainData_y_pred = np.matmul(OLS_trainData_x, OLS_w_vector)
    OLS_testData_y_pred = np.matmul(OLS_testData_x, OLS_w_vector)
    
    # Display relevant evaluation metrics for OLS training data
    print(f'\n\nOLS Training Data: \n{OLS_trainData}')
    
    # Display the OLS w Vector
    print(f'[OLS w Vector] (including bias term):    {OLS_w_vector}')
    print(f"\nRunning Train Data w/ OLS Model:")
    
    print(f'MSE:    {mean_squared_error(OLS_trainData_y, OLS_trainData_y_pred)}')
    print(f'MAE:    {mean_absolute_error(OLS_trainData_y, OLS_trainData_y_pred)}')
    print(f'r2:    {r2_score(OLS_trainData_y, OLS_trainData_y_pred)}')
    
    print(f"\nRunning Test Data w/ OSL Model:")
    
    print(f'MSE:    {mean_squared_error(OLS_testData_y, OLS_testData_y_pred)}')
    print(f'MAE:    {mean_absolute_error(OLS_testData_y, OLS_testData_y_pred)}')
    print(f'r2:    {r2_score(OLS_testData_y, OLS_testData_y_pred)}')

    # Display the training data
    print(f'\n\nLinear Regression Training Data: \n{trainData}')

    # Check that the test and training data combined match the original dataset size
    assert len(testData) + len(trainData) == len(df)

    # Extract the output labels from the training data
    output = trainData[output_name].tolist()

    # Remove the 'Potability' column from the training data
    trainDataTemp = trainData.drop(output_name, axis=1)

    # Convert training data to feature vectors
    features = trainDataTemp.values.tolist()
    X = np.array(features)
    Y = np.array(output)

	# Create a SGDClassifier, and put it through a Pipeline to scale the input features to allow for a more accurately fitted model
    clf = Pipeline([('mmc', StandardScaler()), ('clfRaw', SGDRegressor())])
    clf.fit(X, Y)
    
    # Obtain the bias term and input paramaters of the model to obtain and display the w vector
    w_vector = np.concatenate((clf.named_steps['clfRaw'].intercept_, clf.named_steps['clfRaw'].coef_), axis=None)
    print(f'[Linear Regression w Vector] (including bias term):    {w_vector}')
    
    print(f"\nRunning Train Data w/ Linear Regression Model:")
    runData(trainData)
    
    print(f"\nRunning Test Data w/ Linear Regression Model:")
    runData(testData)
