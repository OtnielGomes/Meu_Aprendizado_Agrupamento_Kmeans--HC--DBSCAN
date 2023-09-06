import pickle
from sklearn.cluster import DBSCAN
import plotly.express as px
import numpy as np
# Pre-processing
with open('credit_card_data.pkl', mode='rb') as f:
    X_credit_card, X_scaler = pickle.load(f)

# Training
dbscan_credit_card = DBSCAN(eps=0.37, min_samples=5)
dbscan_credit_card.fit(X_credit_card)
labels = dbscan_credit_card.labels_

# Pos-processing
print(f'\nLabels: {labels}')
print(np.unique(labels, return_counts=True))

fig = px.scatter(x=X_credit_card[:, 0],
                 y=X_credit_card[:, 1],
                 color=labels)
fig.show()

