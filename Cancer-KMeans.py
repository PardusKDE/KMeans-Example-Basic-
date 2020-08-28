from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn import datasets
cancer = datasets.load_breast_cancer()
cancerd = cancer.data
uns = KMeans(n_clusters=2)
uns.fit(cancerd)
cancert= uns.predict(cancerd)
data = {"Kanser sınıfı": cancert, "Kanser Veri1": cancerd[:, 0], "Kanser Veri2": cancerd[:, 1], "Kanser Veri3": cancerd[:, 2]}
cancert = pd.DataFrame(data=data)
print(cancert)