import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
# Pré processing

credit_card_data = pd.read_csv('credit_card_clients.csv',
                               header=1)
pd.options.display.max_columns = None

# Creating a new column with the sums of the last 6 months of credit
# card expenses
credit_card_data['BILL_Total'] = (credit_card_data['BILL_AMT1'] +
                                  credit_card_data['BILL_AMT2'] +
                                  credit_card_data['BILL_AMT3'] +
                                  credit_card_data['BILL_AMT4'] +
                                  credit_card_data['BILL_AMT5'] +
                                  credit_card_data['BILL_AMT6'])
X_credit_card = credit_card_data.iloc[:, [1, 2, 3, 4, 5, 25]].values
# Staggering
X_scaler = StandardScaler()
X_credit_card_staggered = X_scaler.fit_transform(X_credit_card)

with open('credit_card2_data.pkl', mode='wb') as f:
    pickle.dump([X_credit_card_staggered, X_scaler], f)
