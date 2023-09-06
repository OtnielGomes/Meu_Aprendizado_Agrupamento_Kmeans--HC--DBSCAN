import pickle
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px
import matplotlib.pyplot as plt
# Pre-processing
with open('salary_data.pkl', mode='rb') as f:
    salary_data, salary_scaler = pickle.load(f)

# Training
hierarchy_clusters_salary = AgglomerativeClustering(n_clusters=3,
                                                    linkage='ward')
labels = hierarchy_clusters_salary.fit_predict(salary_data)

# Pos-processing
# Dendrogram
dendrogram_salary = dendrogram(linkage(salary_data, method='ward'))
plt.title('Dendrogram Salary x People')
plt.xlabel('People')
plt.ylabel('Salary')
plt.show()
print(f'\nLabels: {labels}')

fig = px.scatter(x=salary_data[:, 0],
                 y=salary_data[:, 1],
                 color=labels)
fig.show()

