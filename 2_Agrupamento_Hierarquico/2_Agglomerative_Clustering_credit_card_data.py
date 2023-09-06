import pickle
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px

# Pre-processing
with open('credit_card_data.pkl', mode='rb') as f:
    X_credit_card, X_scaler = pickle.load(f)

# Training
hierarchy_clusters = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = hierarchy_clusters.fit_predict(X_credit_card)



# Pos-processing

# Dendrogram
dendrogram_credit_card = dendrogram(linkage(X_credit_card, method='ward'))
plt.show()

print(f'\nLabels: {labels}')

fig = px.scatter(x=X_credit_card[:, 0],
                 y=X_credit_card[:, 1],
                 color=labels)
fig.show()

