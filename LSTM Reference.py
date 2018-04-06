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

if False: #train on multiple timesteps
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
    
#https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/    
if False:
    '''
    # load and plot dataset
    from pandas import read_csv
    from pandas import datetime
    from matplotlib import pyplot
    # load dataset
    def parser(x):
        return datetime.strptime('190'+x, '%Y-%m')
    series = read_csv('Data/shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    # summarize first few rows
    print(series.head())
    # line plot
    series.plot()
    pyplot.show()
    '''
    from pandas import DataFrame
    from pandas import Series
    from pandas import concat
    from pandas import read_csv
    from pandas import datetime
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from math import sqrt
    import matplotlib
    import time

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy
     
    # date-time parsing function for loading the dataset
    def parser(x):
        return datetime.strptime('190'+x, '%Y-%m')
     
    # frame a sequence as a supervised learning problem
    def timeseries_to_supervised(data, lag=1):
        df = DataFrame(data)
        columns = [df.shift(i) for i in range(1, lag+1)]
        columns.append(df)
        df = concat(columns, axis=1)
        df = df.drop(0)
        return df
     
    # create a differenced series
    def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return Series(diff)
     
    # scale train and test data to [-1, 1]
    def scale(train, test):
        # fit scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(train)
        # transform train
        train = train.reshape(train.shape[0], train.shape[1])
        train_scaled = scaler.transform(train)
        # transform test
        test = test.reshape(test.shape[0], test.shape[1])
        test_scaled = scaler.transform(test)
        return scaler, train_scaled, test_scaled
     
    # inverse scaling for a forecasted value
    def invert_scale(scaler, X, yhat):
        new_row = [x for x in X] + [yhat]
        array = numpy.array(new_row)
        array = array.reshape(1, len(array))
        inverted = scaler.inverse_transform(array)
        return inverted[0, -1]
     
    # evaluate the model on a dataset, returns RMSE in transformed units
    def evaluate(model, raw_data, scaled_dataset, scaler, offset, batch_size):
        # separate
        X, y = scaled_dataset[:,0:-1], scaled_dataset[:,-1]
        # reshape
        reshaped = X.reshape(len(X), 1, 1)
        # forecast dataset
        output = model.predict(reshaped, batch_size=batch_size)
        # invert data transforms on forecast
        predictions = list()
        for i in range(len(output)):
            yhat = output[i,0]
            # invert scaling
            yhat = invert_scale(scaler, X[i], yhat)
            # invert differencing
            yhat = yhat + raw_data[i]
            # store forecast
            predictions.append(yhat)
        # report performance
        rmse = sqrt(mean_squared_error(raw_data[1:], predictions))
        return rmse
     
    # fit an LSTM network to training data
    def fit_lstm(train, test, raw, scaler, batch_size, nb_epoch, neurons):
        X, y = train[:, 0:-1], train[:, -1]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        # prepare model
        model = Sequential()
        model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        # fit model
        train_rmse, test_rmse = list(), list()
        for i in range(nb_epoch):
            model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
            model.reset_states()
            # evaluate model on train data
            raw_train = raw[-(len(train)+len(test)+1):-len(test)]
            train_rmse.append(evaluate(model, raw_train, train, scaler, 0, batch_size))
            model.reset_states()
            # evaluate model on test data
            raw_test = raw[-(len(test)+1):]
            test_rmse.append(evaluate(model, raw_test, test, scaler, 0, batch_size))
            model.reset_states()
        history = DataFrame()
        history['train'], history['test'] = train_rmse, test_rmse
        return history
     
    # run diagnostic experiments
    def run():
        # load dataset
        series = read_csv('Data/shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
        # transform data to be stationary
        raw_values = series.values
        diff_values = difference(raw_values, 1)
        
        # transform data to be supervised learning
        supervised = timeseries_to_supervised(diff_values, 1)
        
        supervised_values = supervised.values

        # split data into train and test-sets
        train, test = supervised_values[0:-12], supervised_values[-12:]
        # transform the scale of the data
        scaler, train_scaled, test_scaled = scale(train, test)

        # fit and evaluate model
        train_trimmed = train_scaled[2:, :]
        # config
        repeats = 10
        n_batch = 4
        n_epochs = 2000
        n_neurons = 1
        # run diagnostic tests
        for i in range(repeats):
            t0 = time.time()
            history = fit_lstm(train_trimmed, test_scaled, raw_values, scaler, n_batch, n_epochs, n_neurons)
            plt.plot(history['train'], color='blue')
            plt.plot(history['test'], color='orange')
            print('time elapsed: {} secs'.format(time.time() - t0))
            print('%d) TrainRMSE=%f, TestRMSE=%f' % (i+1, history['train'].iloc[-1], history['test'].iloc[-1]))
        plt.show()
     
    # entry point
    run()

if True:
    from pandas import DataFrame
    from pandas import Series
    from pandas import concat
    from pandas import read_csv
    from pandas import datetime
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from math import sqrt
    import matplotlib
    # be able to save images on server
    from matplotlib import pyplot as plt
    import numpy

    # date-time parsing function for loading the dataset
    def parser(x):
        return datetime.strptime('190'+x, '%Y-%m')

    # frame a sequence as a supervised learning problem
    def timeseries_to_supervised(data, lag=1):
        df = DataFrame(data)
        columns = [df.shift(i) for i in range(1, lag+1)]
        columns.append(df)
        df = concat(columns, axis=1)
        df = df.drop(0)
        return df

    # create a differenced series
    def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return Series(diff)

    # invert differenced value
    def inverse_difference(history, yhat, interval=1):
        return yhat + history[-interval]

    # scale train and test data to [-1, 1]
    def scale(train, test):
        # fit scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(train)
        # transform train
        train = train.reshape(train.shape[0], train.shape[1])
        train_scaled = scaler.transform(train)
        # transform test
        test = test.reshape(test.shape[0], test.shape[1])
        test_scaled = scaler.transform(test)
        return scaler, train_scaled, test_scaled

    # inverse scaling for a forecasted value
    def invert_scale(scaler, X, yhat):
        new_row = [x for x in X] + [yhat]
        array = numpy.array(new_row)
        array = array.reshape(1, len(array))
        inverted = scaler.inverse_transform(array)
        return inverted[0, -1]

    # fit an LSTM network to training data
    def fit_lstm(train, batch_size, nb_epoch, neurons):
        X, y = train[:, 0:-1], train[:, -1]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        model = Sequential()
        model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        for i in range(nb_epoch):
            model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
            model.reset_states()
        return model

    # run a repeated experiment
    def experiment(repeats, series, epochs):
        # transform data to be stationary
        raw_values = series.values
        diff_values = difference(raw_values, 1)
        # transform data to be supervised learning
        supervised = timeseries_to_supervised(diff_values, 1)
        supervised_values = supervised.values
        # split data into train and test-sets
        train, test = supervised_values[0:-12], supervised_values[-12:]
        # transform the scale of the data
        scaler, train_scaled, test_scaled = scale(train, test)
        # run experiment
        error_scores = list()
        for r in range(repeats):
            # fit the model
            batch_size = 4
            train_trimmed = train_scaled[2:, :]
            lstm_model = fit_lstm(train_trimmed, batch_size, epochs, 1)
            # forecast the entire training dataset to build up state for forecasting
            train_reshaped = train_trimmed[:, 0].reshape(len(train_trimmed), 1, 1)
            lstm_model.predict(train_reshaped, batch_size=batch_size)
            # forecast test dataset
            test_reshaped = test_scaled[:,0:-1]
            test_reshaped = test_reshaped.reshape(len(test_reshaped), 1, 1)
            output = lstm_model.predict(test_reshaped, batch_size=batch_size)
            predictions = list()
            for i in range(len(output)):
                yhat = output[i,0]
                X = test_scaled[i, 0:-1]
                # invert scaling
                yhat = invert_scale(scaler, X, yhat)
                # invert differencing
                yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
                # store forecast
                predictions.append(yhat)
            # report performance
            rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
            print('%d) Test RMSE: %.3f' % (r+1, rmse))
            error_scores.append(rmse)
    return error_scores


    # load dataset
    series = read_csv('Data/shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    # experiment
    repeats = 30
    results = DataFrame()
    # vary training epochs
    epochs = [500, 1000, 2000, 4000, 6000]
    for e in epochs:
        results[str(e)] = experiment(repeats, series, e)
    # summarize results
    print(results.describe())
    # save boxplot
    results.boxplot()
    plt.show()
