if False:
    import numpy as np

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM

    # Input sequence
    wholeSequence = [[0,0,0,0,0,0,0,0,0,2,1],
                     [0,0,0,0,0,0,0,0,2,1,0],
                     [0,0,0,0,0,0,0,2,1,0,0],
                     [0,0,0,0,0,0,2,1,0,0,0],
                     [0,0,0,0,0,2,1,0,0,0,0],
                     [0,0,0,0,2,1,0,0,0,0,0],
                     [0,0,0,2,1,0,0,0,0,0,0],
                     [0,0,2,1,0,0,0,0,0,0,0],
                     [0,2,1,0,0,0,0,0,0,0,0],
                     [2,1,0,0,0,0,0,0,0,0,0]]

    # Preprocess Data:
    wholeSequence = np.array(wholeSequence, dtype=float) # Convert to NP array.
    data = wholeSequence[:-1] # all but last

    target = wholeSequence[1:] # all but first


    # Reshape training data for Keras LSTM model
    # The training data needs to be (batchIndex, timeStepIndex, dimentionIndex)
    # Single batch, 9 time steps, 11 dimentions
    print(data.shape)
    print(data)
    data = data.reshape((1, 9, 11))
    target = target.reshape((1, 9, 11))
    print(data.shape)
    print(data)
    print(target.shape)
    print(target)
    what = input('what')
    # Build Model
    model = Sequential()  
    model.add(LSTM(11, input_shape=(9, 11), unroll=True, return_sequences=True))
    model.add(Dense(11))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    model.fit(data, target, nb_epoch=2000, batch_size=1, verbose=2)

#https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
if False: #train on one timestep
    from pandas import read_csv
    from datetime import datetime
    # load data
    def parse(x):
        return datetime.strptime(x, '%Y %m %d %H')
    dataset = read_csv('Data/raw.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
    dataset.drop('No', axis=1, inplace=True)
    # manually specify column names
    dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'
    # mark all NA values with 0
    dataset['pollution'].fillna(0, inplace=True)
    # drop the first 24 hours
    dataset = dataset[24:]
    # summarize first 5 rows
    print(dataset.head(5))
    # save to file
    dataset.to_csv('Data/pollution.csv')

    from matplotlib import pyplot
    # load dataset
    dataset = read_csv('Data/pollution.csv', header=0, index_col=0)
    values = dataset.values #(43800, 8)43800 rows, 8 features
    # specify columns to plot
    groups = [0, 1, 2, 3, 5, 6, 7]
    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(dataset.columns[group], y=0.5, loc='right')
        i += 1
    pyplot.show()

    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    import pandas as pd
    
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
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
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    # load dataset
    dataset = read_csv('Data/pollution.csv', header=0, index_col=0)
    values = dataset.values
    # integer encode direction
    encoder = LabelEncoder()
    values[:,4] = encoder.fit_transform(values[:,4])
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values) #the shape is still (43800, 8)

    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1) #(43799, 16)

    # drop columns we don't want to predict
    # this is such a werid way of doing this
    reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
    print(reframed.head())

    # split into train and test sets
    values = reframed.values
    n_train_hours = 365 * 24
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

if True: #train on multiple timesteps
    from math import sqrt
    from numpy import concatenate
    from matplotlib import pyplot
    from pandas import read_csv
    from pandas import DataFrame
    from pandas import concat
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_squared_error
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM

    # convert series to supervised learning
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
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
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    # load dataset
    dataset = read_csv('Data/pollution.csv', header=0, index_col=0)
    values = dataset.values
    # integer encode direction
    encoder = LabelEncoder()
    values[:,4] = encoder.fit_transform(values[:,4])
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # specify the number of lag hours
    n_hours = 3
    n_features = 8
    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_hours, 1) #(43797, 32)

    # split into train and test sets
    values = reframed.values
    n_train_hours = 365 * 24 #train test split, expecting new dataset to be 8760 long
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    n_obs = n_hours * n_features #24
    #now we end up with something of original length, and repeated features by timestep
    train_X, train_y = train[:, :n_obs], train[:, -n_features] 
    test_X, test_y = test[:, :n_obs], test[:, -n_features] 
    print(train_y)
    print(train_y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    #(8760, 3, 8) (8760,) (35037, 3, 8) (35037,)
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    print(train_X[:10])
    wait = input('bookmark')
    # design network
    model = Sequential()
    #3, 8? why
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, -7:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, -7:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    
    
