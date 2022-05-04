# Importing Libraries
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import sklearn 

# Perfomance Metrics 
from sklearn.metrics import confusion_matrix, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#Train test split 
from sklearn.model_selection import train_test_split

# Regressors 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import svm



# Variables 
n_in = 20
mode = 'D'
start_date = '2019-01-01'
end_date = '2019-05-01'
predicted_variable = 'tempr' + '_t'


# Data pre-preocessing
data = pd.read_csv('.\Dataset\data_trial.csv')
df = pd.DataFrame(data)


#Drop time stamps and convert the date column to the index
df_temp = df.date.str.split(" ", expand = True)
df['date'] = df_temp
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace = True)


#Drop the unwanted columns for now
df.drop(columns = ['blizzard'], inplace = True)


#Function to convert the given time series data
def series_to_supervised(df, n_in=1, n_out=1, dropnan=True):
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


#Group the data according to date 
sup_data = df.resample(mode).mean()


#supervised learning dataset 
sup_data = series_to_supervised(df, n_in, 1, dropnan = True)


#Rename the columns for better readbility 
cols = ['tempr_t','ap_t','ws_t', 'rh_t']
for j in range(1, n_in+1):
    for i in range(1, 5):
          cols.insert(0,cols[-i] + '-'+ str(j))
sup_data.columns = cols



# Gather the data required to train the model 
df_train = sup_data.loc[start_date : end_date]


#define all the models 
model = RandomForestRegressor(n_estimators=1000, verbose = 1, n_jobs=-1, random_state= 0)
    #model = AdaBoostRegressor()
    #model = ExtraTreesRegressor()
    #model = svm.SVR(kernel = 'poly', C = 200, degree = 10)


# Split the data intro train and test 
labels = np.array(df_train[predicted_variable])
train_set = df_train.iloc[:, 0:(n_in*4)]
x_train, x_test, y_train, y_test = train_test_split(train_set, labels, test_size = 0.2, shuffle = False) 


#train teh model 
model.fit(x_train, y_train)

#predict outcomes 
y_pred = model.predict(x_test)

print(mean_squared_error(y_pred, y_test))

# Plotting the outputs
plt.figure(figsize=(16,6))
plt.plot(df_train.index, labels, label = 'original data')
plt.plot(x_test.index, y_test, label = 'expected')
plt.plot(x_test.index, y_pred, label = 'predicted')
plt.title(predicted_variable + " predictions for 3 months using Random Forest Regressor")
plt.xlabel("Date")
plt.ylabel(predicted_variable)
plt.legend()
plt.show()