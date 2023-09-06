import pickle
from sklearn.cluster import DBSCAN
import plotly.express as px

# Pro-processing
with open('salary_data.pkl', mode='rb') as f:
    salary_data, salary_scaler = pickle.load(f)

# Training
dbscan_salary = DBSCAN(eps=0.95, min_samples=2)
dbscan_salary.fit(salary_data)
labels = dbscan_salary.labels_

# Pos-processing
print(f'\nLabels: {labels}')
fig = px.scatter(x=salary_data[:, 0],
                 y=salary_data[:, 1],
                 color=labels)
fig.show()