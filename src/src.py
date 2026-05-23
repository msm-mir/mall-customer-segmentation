import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# read dataset
df = pd.read_csv('Mall_Customers.csv')

# drop CustomerID column
df = df.drop('CustomerID', axis=1)

# pairplot including gender
sns.pairplot(df, hue='Gender')
plt.suptitle("Pairplot segmented by Gender", y=1.02)
plt.show()

# drop Gender & Age column because it has no influence on clustering
df = df.drop('Gender', axis=1)
df = df.drop('Age', axis=1)