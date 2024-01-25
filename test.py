import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

Microsoft = yf.Ticker("MSFT")
interest = yf.Ticker("IRX")

# create a plot of the closing prices and name the file "MSFT.png"
stock = Microsoft.history(period="max")
interest_rate = interest.history(period="max")

# print(interest_rate)
#print(stock)
# creating 2 new columns for the tomorrow price and increase yes or no
stock["Tomorrow"] = stock["Close"].shift(-1)
stock["increase"] = (stock["Tomorrow"] > stock["Close"]).astype(int)
stock["T_Bills"] = interest_rate["Open"]



# making a copy of the data history from 2000 onwards
stock = stock.loc["2001-01-01":].copy()

stock = pd.DataFrame(stock)
print(stock)

# turn stock into a pandas datframe
# stock = pd.DataFrame(stock)

# initializing the model
model = RandomForestClassifier(n_estimators = 100, min_samples_split = 50, random_state = 1) 

# feed the training set all data but last 100 days
training = stock.iloc[:-100]
# test on the most recent 100 days
test = stock.iloc[-100:]

# initialize the predictors 
predictors = ["Close", "Volume", "Open", "High", "Low"]
# train the model for the Target 
def predict(training, test, predictors, model):
    model.fit(training[predictors], training["increase"])


    # make the predictions
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    # make into a pandas dataframe
    preds = pd.Series(preds, index=test.index, name="predictions")
    
    # test the prediction
   # score = precision_score(test["increase"], predictions)

    combine = pd.concat([test["increase"], preds], axis=1)
    return combine

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):

        training = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(training, test, predictors, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)

predictions = backtest(stock, model, predictors)

horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_average = stock.rolling(horizon).mean()

    ratio_column = f"Close_Ratio{horizon}"

    stock[ratio_column] = stock["Close"] / rolling_average["Close"]

    trend_column = f"Trend{horizon}"
    stock[trend_column] = stock.shift(1).rolling(horizon).sum()["increase"]

    new_predictors += [ratio_column, trend_column]


#model = RandomForestClassifier(n_estimators = 100, min_samples_split = 50, random_state = 1)


predictions = backtest(stock, model, new_predictors)

# print(precision_score(predictions["increase"], predictions["predictions"]))
count = 0
for _, row in predictions.iterrows():
    if (row["increase"] == 1) & (row["predictions"] == 1):
        count += 1

print ((count)/predictions["predictions"].value_counts())