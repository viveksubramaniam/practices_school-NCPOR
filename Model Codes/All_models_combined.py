# Importing Libraries
from keras.engine import sequential
from keras.layers.core import Dropout
import numpy as np
from numpy.core.fromnumeric import shape 
import pandas as pd 
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import sklearn 

# Perfomance Metrics 
from sklearn.metrics import confusion_matrix, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#Train test split 
from sklearn.model_selection import train_test_split

#scaler 
from sklearn.preprocessing import MinMaxScaler

# Regressors 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import svm
from keras.models import Sequential 
from keras.layers import Flatten
from keras.layers import Dense 
from keras.layers import LSTM 
from keras.layers import Bidirectional
from keras.layers import SimpleRNN

sc = MinMaxScaler(feature_range = (0,1))



# Variables 
fileLocation = 'e:\Documents\College Stuff\PS\Code\Dataset\data_trial.csv'
dropColumn = 'blizzard'
index = 'date'


n_in = 90
n_out = 14
n_days = 7
trainingWindow = 0.7699
numberOfMonths = 6
numberOfDays = 30*numberOfMonths
mode = 'D'


start_date = '2019-01-01'
end_date = '2019-09-01'


models = ['extratrees', 'svm', 'adaboost', 'randomforest', 'ANN', 'RNN', 'CNN', 'LSTM', 'RNN+ANN', 'ANN+LSTM']
RNNModels = ['RNN', 'LSTM', 'RNN+ANN', 'ANN+LSTM']
#change to change the regressor
modelNumber = -2
regressor = models[modelNumber]
attributes = ['tempr', 'ap', 'ws', 'rh']
predicted_variable = attributes[0] + '_t'

windowSize = 90 
futurePredictionSize = 14
step = futurePredictionSize

def dataPreparation(fileLocation, dropColumn, index):
    # Data pre-preocessing
    data = pd.read_csv(fileLocation)
    df = pd.DataFrame(data)

    #Drop time stamps and convert the date column to the index
    df[index] = pd.to_datetime(df[index])
    df.index = df[index]
    df.set_index(index, inplace = True)

    #plt.plot(df.index, df['tempr'])
    #plt.show()

    #Drop the unwanted columns for now
    df.drop(columns = [dropColumn], inplace = True)

    df = df.resample(mode).mean()
    print(df)
    # plt.figure(figsize = (16,6))
    # plt.plot(df['tempr'])
    # plt.show()

    return data, df


def colNameChange(agg):
      #Rename the columns for better readbility 
    cols = ['tempr_t','ap_t','ws_t', 'rh_t']
    cols_out = ['tempr_t','ap_t','ws_t', 'rh_t']
    for j in range(1, n_in+1):
        for i in range(1, 5):
            cols.insert(0,cols[-i] + '-'+ str(j))

    for j in range(1, n_out):
        for i in range(0, 4):
            cols.insert(len(cols), cols_out[i] + '+'+ str(j))
    i = 0
    for colName in cols:
        for name_change in cols_out:
            if name_change == colName:
                cols[i] = name_change + '+0'
        i+=1

    agg.columns = cols

    return agg


#Function to convert the given time series data
def series_to_supervised(data, df, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	#df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)

	return agg


def attributeDataFrame(predicted_variable, sup_data):
    
    attributeDf = pd.DataFrame()

    for i in range(windowSize, 0, -1):
        attributeDf[predicted_variable +'-'+str(i)] = sup_data[predicted_variable + '-' + str(i)]

    for i in range(futurePredictionSize):
        attributeDf[predicted_variable + '+' + str(i)] = sup_data[predicted_variable + '+' + str(i)]

    return attributeDf


def shiftData(df, trainingWindow, n_in, n_out, step):
    trainSize = numberOfMonths*30
    # trainSize = int(trainingWindow*len(df))
    X_train = []
    y_train =[]
    X_test = []
    y_test = []

    for i in range(0, trainSize+1, 1):
        X_train.append(attributeDf.iloc[i, 0:n_in])
        y_train.append(attributeDf.iloc[i, n_in:n_out+n_in])

    # for i in range(trainSize+1, len(df), step):
    #     X_test.append(attributeDf.iloc[i, 0:n_in])
    #     y_test.append(attributeDf.iloc[i, n_in:n_out+n_in])

    X_test.append(attributeDf.iloc[trainSize+1, 0:n_in])
    y_test.append(attributeDf.iloc[trainSize+1,n_in:n_in+n_out])

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, X_test, y_train, y_test



def NormalizeData(x_train, x_test, y_train, y_test, regressor):

    # x_train = np.reshape(x_train, (1,-1))
    # y_train = np.reshape(y_train, (1,-1))
    # print(x_train)
    print(np.shape(y_train))
    # x_test = np.reshape(x_test, (-1,1))
    # y_test = np.reshape(y_test, (1,-1))
    x_train = sc.fit_transform(x_train)
    y_train = sc.fit_transform(y_train)
    x_test = sc.fit_transform(x_test)
    y_test = sc.fit_transform(y_test)

    if regressor in RNNModels:
        x_train = x_train.reshape(np.shape(x_train)[0], np.shape(x_train)[1], 1)
        x_test = x_test.reshape(np.shape(x_test)[0], np.shape(x_train)[1], 1)


    return x_train, x_test, y_train, y_test



def trainModel(x_train, y_train, regressor):

    epochs = 200 
    batchSize = 30  

    if regressor == 'randomforest':
        model = RandomForestRegressor(n_estimators=1000, verbose = 1, n_jobs=-1, random_state= 0)
        model.fit(x_train, y_train)
    elif regressor == 'adaboost':
        model = AdaBoostRegressor()
        model.fit(x_train, y_train)
    elif regressor == 'extratrees':
        model = ExtraTreesRegressor(n_estimators=1000, n_jobs=4, min_samples_split=25, min_samples_leaf=35)
        model.fit(x_train, y_train)
    elif regressor == 'svm':
        model = svm.SVR(kernel = 'poly', C = 200, degree = 10)
        model.fit(x_train, y_train)
    elif regressor == 'ANN':

        model = Sequential()
        model.add(Dense(units = windowSize, input_dim = windowSize, activation = 'relu'))
        model.add(Dense(units = windowSize, activation = 'relu'))
        model.add(Dense(units = windowSize, activation = 'relu'))
        model.add(Dense(units = windowSize, activation = 'relu'))
        model.add(Dense(units = windowSize, activation = 'relu'))
        model.add(Dense(units = windowSize, activation = 'relu'))
        model.add(Dense(units = windowSize, activation = 'relu'))
        model.add(Dense(units = futurePredictionSize, activation = 'relu'))
        model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
        model.fit(x_train, y_train, epochs = epochs, batch_size = batchSize)
    elif regressor == "RNN+ANN":
        model = Sequential()
        model.add(SimpleRNN(units = windowSize, input_shape = (windowSize,1), activation = 'relu', return_sequences = "True"))
        model.add(SimpleRNN(units = windowSize, activation = 'tanh', return_sequences = "True"))
        model.add(Dense(units = windowSize, activation = 'relu'))
        model.add(SimpleRNN(units = windowSize, activation = 'tanh', return_sequences = "True"))
        model.add(Dense(units = windowSize, activation = 'relu'))
        model.add(SimpleRNN(units = windowSize, activation = 'tanh', return_sequences = "True"))
        model.add(Dense(units = windowSize, activation = 'relu'))
        model.add(SimpleRNN(units = windowSize, activation = 'tanh', return_sequences = "True"))
        model.add(Dense(units = windowSize, activation = 'relu'))
        model.add(SimpleRNN(units = windowSize, activation = 'tanh', return_sequences = "True"))
        model.add(Dense(units = windowSize, activation = 'relu'))
        model.add(Flatten())
        model.add(Dense(units = futurePredictionSize, activation = "relu"))
        model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
        model.fit(x_train, y_train, epochs = epochs, batch_size = batchSize)
    elif regressor == "RNN":
        model = Sequential()
        model.add(SimpleRNN(units = windowSize, input_shape = (windowSize, 1), activation = "relu", return_sequences= True))
        model.add(SimpleRNN(units = windowSize, activation = 'relu', return_sequences = "True"))
        model.add(SimpleRNN(units = windowSize, activation = 'relu', return_sequences = "True"))
        model.add(SimpleRNN(units = windowSize, activation = 'relu', return_sequences = "True"))
        model.add(SimpleRNN(units = windowSize, activation = 'relu', return_sequences = "True"))
        model.add(Flatten())
        model.add(Dense(units = futurePredictionSize))
        model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
        model.fit(x_train, y_train, epochs = epochs, batch_size = batchSize)
    elif regressor == "ANN+LSTM":
        model = Sequential()
        model.add(Bidirectional(LSTM(units = windowSize*2, input_shape = (windowSize, 1), activation = "relu")))
        model.add(Dropout(0.5))
        model.add(Dense(units = windowSize*2, activation = "relu"))
        model.add(Dropout(0.1))
        model.add(Dense(units = windowSize*2, activation = "relu"))
        model.add(Dropout(0.1))
        model.add(Dense(units = windowSize*2, activation = "relu"))
        model.add(Dropout(0.1))
        model.add(Dense(units = windowSize*2, activation = "relu"))
        model.add(Dropout(0.1))
        model.add(Dense(units = windowSize*2, activation = "relu"))
        model.add(Dropout(0.1))
        model.add(Dense(units = futurePredictionSize))
        model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
        model.fit(x_train, y_train, epochs = epochs, batch_size = batchSize)
    

    return model

def ModelPredictions(x_test, y_test, model):
    
    y_pred = model.predict(x_test)

    # y_pred = np.reshape(y_pred, (-1,1))
    # y_test = np.reshape(y_test, (-1,1))

    y_pred = sc.inverse_transform(y_pred)
    y_test = sc.inverse_transform(y_test)
    # x_train = sc.inverse_transform(x_train)

    return y_pred, y_test


def perfMetrics(y_pred, y_test):
    #print performance metrics 
    errors = abs(y_pred - y_test)
    mape = 100 * np.mean(abs(errors / y_test))
    accuracy = 100 - mape 
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared = False)
    mse = mean_squared_error(y_test, y_pred, squared = True)
    avgError = np.mean(errors)
    print('Root Mean Squared Error:{:0.4f} degrees.'.format(mean_squared_error(y_test, y_pred, squared = False)))
    print('Mean Squared Error: {:0.4f}'.format(mse))
    print('MAPE: {:0.4f} degrees.'.format(mape))
    print('Accuracy = {:0.4f}%.'.format(accuracy))
    print('mae:{:0.4f}'.format(mae))

    return rmse, avgError, accuracy, mae

def printGraphs(df, y_train, y_test, y_pred):

    #evaluate the model
    rmse, avgError, accuracy, mae = perfMetrics(y_pred, y_test)

    y_train = y_train.flatten()
    y_pred = y_pred.flatten()
    y_test = y_test.flatten()

    GraphLabelX = 'Date'
    GraphLabelY = ['Temperature (degree celsius)', 'Atmoshperic Pressure', 'Relative Humidity', 'Wind Speed']

    graphStep = 1

    # endTrain = len(df) - len(y_test)
    # startTrain = int(endTrain - len(y_test)*3.5)

    endTrain = 181 + 90 
    startTrain = 181 + 90 - 90

    plt.figure(figsize=(16,6))
    plt.plot(df.index[startTrain:endTrain+1:graphStep], df.iloc[startTrain:endTrain+1:graphStep,0:1], label = 'Original data', c = 'g')
    plt.plot(df.index[endTrain:endTrain+14:graphStep], y_test[::graphStep], label = 'Validation', c = 'k')
    plt.plot(df.index[endTrain:endTrain+14:graphStep], y_pred[::graphStep], label = 'Prediction', c = 'r')
    plt.title('Parameter: ' + GraphLabelY[0] + '\nTraining Duration:' + str(df.index[0]) + '-' + str(df.index[90]) + '\nModel: ' + models[modelNumber] + ', Forecast Duration:' + str(futurePredictionSize) + ' days')
    plt.text(mdates.date2num(df.index[3]), -5 ,'RMSE: ' + str(rmse) + '\nAverage Error: ' + str(avgError) +'\nAccuracy: ' + str(accuracy) + '\nMean Average Error: ' + str(mae))
    plt.xlabel(GraphLabelX)
    plt.ylabel(GraphLabelY[0])
    plt.gca().grid(True)
    plt.gca().set_facecolor((1, 1, 0.8))
    plt.savefig(models[modelNumber]+'.png', dpi=300)
    plt.legend()
    plt.show()
    

#supervised learning dataset 
data, df = dataPreparation(fileLocation, dropColumn, index)
sup_data = series_to_supervised(data, df, n_in, n_out, dropnan = True)
sup_data = colNameChange(sup_data)
attributeDf = attributeDataFrame(predicted_variable, sup_data)

# Split the data intro train and test 
labels = np.array(attributeDf.iloc[:, n_in:n_out+n_in])
#Split for daily 
x_train, x_test, y_train, y_test = shiftData(sup_data, trainingWindow, n_in, n_out, step)
x_train, x_test, y_train, y_test = NormalizeData(x_train, x_test, y_train, y_test, regressor)

#train the model
model = trainModel(x_train, y_train, regressor)

#Predictions 
y_pred, y_test = ModelPredictions(x_test, y_test, model)

#print the graphs
printGraphs(df, y_train, y_test, y_pred)
