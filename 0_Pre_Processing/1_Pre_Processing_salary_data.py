import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# Creating Database
age = [20,  27,  21,  37,  46, 53, 55,  47,  52,  32,  39,  41,  39,  48,  48]
salary = [1000, 1200, 2900, 1850, 900, 950, 2000, 2100, 3000, 5900, 4100, 5100,
          7000, 5000, 6500]
salary_data = np.array([[20, 1000], [27, 1200], [21, 2900], [37, 1850],
                        [46, 900], [53, 950], [55, 2000], [47, 2100],
                        [52, 3000], [32, 5900], [39, 4100], [41, 5100],
                        [39, 7000], [48, 5000], [48, 6500]])
# Pré-processing
salary_scaler = StandardScaler()
salary_staggered = salary_scaler.fit_transform(salary_data)
with open('salary_data.pkl', mode='wb') as f:
    pickle.dump([salary_staggered, salary_scaler], f)
