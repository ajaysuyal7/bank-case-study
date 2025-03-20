import pandas as pd

# Load the dataset
dataset_path = "Bankloans.csv"  # Update this path
data = pd.read_csv(dataset_path)

# Display the first few rows of the dataset
print(data.head())

# Display the shape of the dataset
print(data.shape)

# Display the columns of the dataset
print(data.columns)

# Display the data types of the columns
print(data.dtypes)

# Display the summary statistics of the dataset
print(data.describe())

# Display the missing values in the dataset
print(data.isnull().sum())

# Display the unique values in the dataset
print(data.nunique())

# Display the correlation matrix of the dataset
print(data.corr())

# Display the value counts of a specific column
print(data.default.value_counts(normalize=True))