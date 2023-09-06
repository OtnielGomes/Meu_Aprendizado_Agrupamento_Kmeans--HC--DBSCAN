import pickle
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
# Pre-processing
with open('salary_data.pkl', mode='rb') as f:
    salary_data, salary_scaler = pickle.load(f)
# Training
kmeans_salary = KMeans(n_clusters=3, n_init=10)
kmeans_salary.fit(salary_data)

centroids = kmeans_salary.cluster_centers_
centroids_inverse = salary_scaler.inverse_transform(centroids)
labels = kmeans_salary.labels_
salary_data_inverse = salary_scaler.inverse_transform(salary_data)

# Pos-processing
print(f'Centroids: \n{centroids_inverse}')
print(f'\nLabels: {labels}')

fig1 = px.scatter(x=salary_data_inverse[:, 0],
                  y=salary_data_inverse[:, 1],
                  color=labels)
fig2 = px.scatter(x=centroids_inverse[:, 0],
                  y=centroids_inverse[:, 1],
                  size=[10, 10, 10])
fig3 = go.Figure(data=fig1.data+fig2.data)
fig3.show()
