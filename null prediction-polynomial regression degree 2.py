# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load your dataset with NaN values
df = pd.read_excel('C:\\Users\\homa.behmardi\\Downloads\\h3.xlsx')

# Separate the rows with NaN values and those without NaN values
df_with_nan = df[df.isna().any(axis=1)]
df_without_nan = df[~df.isna().any(axis=1)]

# Create features (X) and target (y) for training the model
X = df_without_nan[['Total_Traffic(UL+DL)(GB)(Ericsson_LTE_Sector)','DL_Traffic(GB)(Ericsson_LTE_Sector)', 'DL_Spectral_efficiency(Ericsson_LTE_Sector)', 'Average_Reported_CQI(Ericsson_LTE_Sector)','DL_PRB_Utilization_Rate(Ericsson_LTE_Sector)','Sector_Bandwidth(MHz)(Ericsson_LTE_Sector)','Cell_Availability_Rate_Include_Blocking(Ericsson_LTE_Sector)']]
y = df_without_nan[['Average_UE_DL_Throughput(Mbps)(Ericsson_LTE_Sector)','Average_RRC_Connected_Users(Ericsson_LTE_Sector)']]

# Create polynomial features of degree 2
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train a linear regression model on polynomial features
model = LinearRegression()
model.fit(X_poly, y)

# Calculate the R-squared value for the combined target variables
r_squared = model.score(X_poly, y)

# Use the trained model to predict NaN values
X_nan = df_with_nan[['Total_Traffic(UL+DL)(GB)(Ericsson_LTE_Sector)','DL_Traffic(GB)(Ericsson_LTE_Sector)', 'DL_Spectral_efficiency(Ericsson_LTE_Sector)', 'Average_Reported_CQI(Ericsson_LTE_Sector)','DL_PRB_Utilization_Rate(Ericsson_LTE_Sector)','Sector_Bandwidth(MHz)(Ericsson_LTE_Sector)','Cell_Availability_Rate_Include_Blocking(Ericsson_LTE_Sector)']]
X_nan_poly = poly.transform(X_nan)
predicted_values = model.predict(X_nan_poly)

# Replace NaN values with predicted values in the original dataframe
df.loc[df.isna().any(axis=1), ['Average_UE_DL_Throughput(Mbps)(Ericsson_LTE_Sector)', 'Average_RRC_Connected_Users(Ericsson_LTE_Sector)']] = predicted_values

# Print the R-squared value
print(f"R-squared (Coefficient of Determination): {r_squared}")
