import pickle
from sklearn.cluster import KMeans
import plotly.express as px
# Pre-processing
with open('credit_card_data.pkl', mode='rb') as f:
    X_credit_card, X_scaler = pickle.load(f)

# Training
# Found the number of clusters
wcss = []
for i in range(1, 11):
    kmeans_test = KMeans(n_clusters=i, n_init=10, random_state=0)
    kmeans_test.fit(X_credit_card)
    wcss.append(kmeans_test.inertia_)

# Results
print(f'WCSS: {wcss}')
fig_test = px.line(x=range(1, 11), y=wcss)
fig_test.show()
# n_clusters = 4
kmeans_credit_card = KMeans(n_clusters=4, n_init=10, random_state=0)
labels = kmeans_credit_card.fit_predict(X_credit_card)

# Pos-processing
fig = px.scatter(x=X_credit_card[:, 0],
                 y=X_credit_card[:, 1],
                 color=labels)
fig.show()

