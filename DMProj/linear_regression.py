### Loading Data
#-------------------------------------------------------------------------------
# import packages
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

# Load data from files as numpy arrays 
# Load data from files as numpy arrays 
dollars = np.genfromtxt("US_Corn_Futures.csv", delimiter = ",", skip_header=1)
# Discard the date and index so we have only the values also "vertically stack"
# the array so it is in the correct orientation for matrix multiplications
dollars = np.vstack(dollars[:,2])
# Do the same for the other files
interest_rates = np.genfromtxt("FEDFUNDS.csv", delimiter = ",", skip_header=1)
interest_rates = np.vstack(interest_rates[:,1])
inflation = np.genfromtxt("inflation.csv", delimiter = ",", skip_header=1)
inflation = np.vstack(inflation[:,3])

### Linear Regression based solely off precipitation
#-------------------------------------------------------------------------------
# Add a column of ones to account for the bias in the data
bias = np.ones(shape = (len(interest_rates),1))
X = np.concatenate((bias, interest_rates), 1)
# Matrix form of linear regression
coeffs = la.inv(X.T @ X ) @ X.T @ dollars
bias = coeffs[0][0]
slope = coeffs[1][0]
print("Linearly Predicted Prices From Precipitation")
print("Bias: " + str(bias))
print("Slope: " + str(slope))

residuals = np.zeros(len(dollars))
predictions = np.zeros(len(dollars))
for i in range(len(dollars)):
  predictions[i] = interest_rates[i]*slope + bias
  residuals[i] = dollars[i] - predictions[i]

MSE = np.mean(residuals**2)
print("Mean Squared Error = " + str(MSE) +"\n\n")

plt.plot(predictions)
plt.plot(dollars)
plt.xlabel("Date")
plt.ylabel("Price per Ton ($USD)")
plt.title("Linearly Predicted Prices From Precipitation")
plt.show()
#-------------------------------------------------------------------------------

### Linear prediction from temperature
#-------------------------------------------------------------------------------
# Add a column of ones to account for the bias in the data
bias = np.ones(shape = (len(inflation),1))
X = np.concatenate((bias, inflation), 1)

# Matrix form of linear regression
coeffs = la.inv(X.T @ X) @ X.T @ dollars
bias = coeffs[0][0]
slope = coeffs[1][0]
print("Linearly Predicted Prices From Temperature")
print("Bias: " + str(bias))
print("Slope: " + str(slope))

residuals = np.zeros(len(dollars))
predictions = np.zeros(len(dollars))
for i in range(len(dollars)):
  predictions[i] = inflation[i]*slope + bias
  residuals[i] = dollars[i] - predictions[i]

MSE = np.mean(residuals**2)
print("Mean Squared Error = " + str(MSE) +"\n\n")

plt.plot(predictions)
plt.plot(dollars)
plt.xlabel("Date")
plt.ylabel("Price per Ton ($USD)")
plt.title("Linearly Predicted Prices From Temperature")
plt.show()
#-------------------------------------------------------------------------------

### Multilinear regression based off both variables
#-------------------------------------------------------------------------------
# Clearly neither of them predicts it very well on their own, lets 
# try them together and see if that's any better

# Add a column of ones to account for the bias in the data
bias = np.ones(shape = (len(inflation),1))
X = np.concatenate((bias, inflation, interest_rates), 1)

# Matrix form of linear regression
coeffs = la.inv(X.T @ X) @ X.T @ dollars
bias = coeffs[0][0]
tempSlope = coeffs[1][0]
precSlope = coeffs[2][0]
print("Multilinear Regression Predicted Prices")
print("Bias: " + str(bias))
print("Temperature Slope: " + str(tempSlope))
print("Precipitation Slope: " + str(precSlope))

residuals = np.zeros(len(dollars))
predictions = np.zeros(len(dollars))
for i in range(len(dollars)):
  predictions[i] = inflation[i]*tempSlope + interest_rates[i]*precSlope + bias
  residuals[i] = dollars[i] - predictions[i]

MSE = np.mean(residuals**2)
print("Mean Squared Error = " + str(MSE) +"\n\n")

plt.plot(predictions)
plt.plot(dollars)
plt.xlabel("Months after Jan 1 1990")
plt.ylabel("Price per Ton ($USD)")
plt.title("Multilinear Regression Predicted Prices")
plt.show()

# ### Ridge Regression based solely off precipitation
# #-------------------------------------------------------------------------------
# # Add a column of ones to account for the bias in the data
# bias = np.ones(shape = (len(precipitation),1))
# X = np.concatenate((bias, precipitation), 1)
# I = np.identity(2)

# #regularization parameter
# s = 50
# # Matrix form of linear regression
# coeffs = la.inv((X.T @ X) + s * I) @ X.T @ dollars
# bias = coeffs[0][0]
# slope = coeffs[1][0]
# print("Ridge Regression Predicted Prices From Precipitation S="+str(s))
# print("Bias: " + str(bias))
# print("Slope: " + str(slope))

# residuals = np.zeros(len(dollars))
# predictions = np.zeros(len(dollars))
# for i in range(len(dollars)):
#   predictions[i] = precipitation[i]*slope + bias
#   residuals[i] = dollars[i] - predictions[i]

# MSE = np.mean(residuals**2)
# print("Mean Squared Error = " + str(MSE) +"\n\n")

# plt.plot(predictions)
# plt.plot(dollars)
# plt.xlabel("Date")
# plt.ylabel("Price per Ton ($USD)")
# plt.title("Ridge Regression Predicted Prices From Precipitation S="+str(s))
# plt.show()


# ### Ridge Regression based solely off temperature
# #-------------------------------------------------------------------------------
# # Add a column of ones to account for the bias in the data
# bias = np.ones(shape = (len(temperature),1))
# X = np.concatenate((bias, temperature), 1)
# I = np.identity(2)
# #regularization parameter
# s = 50

# # Matrix form of linear regression
# coeffs = la.inv((X.T @ X) + s * I) @ X.T @ dollars
# bias = coeffs[0][0]
# slope = coeffs[1][0]
# print("Ridge Regression Predicted Prices From Temperature S="+str(s))
# print("Bias: " + str(bias))
# print("Slope: " + str(slope))

# residuals = np.zeros(len(dollars))
# predictions = np.zeros(len(dollars))
# for i in range(len(dollars)):
#   predictions[i] = temperature[i]*slope + bias
#   residuals[i] = dollars[i] - temperature[i]

# MSE = np.mean(residuals**2)
# print("Mean Squared Error = " + str(MSE) +"\n\n")

# plt.plot(predictions)
# plt.plot(dollars)
# plt.xlabel("Date")
# plt.ylabel("Price per Ton ($USD)")
# plt.title("Ridge Regression Predicted Prices From Temperature S="+str(s))
# plt.show()


### Ridge Regression based solely off temperature
#-------------------------------------------------------------------------------
# Add a column of ones to account for the bias in the data
bias = np.ones(shape = (len(inflation),1))
X = np.concatenate((bias, inflation, interest_rates), 1)
I = np.identity(3)
#regularization parameter
s = 4
# Matrix form of linear regression
coeffs = la.inv((X.T @ X) + s * I) @ X.T @ dollars
bias = coeffs[0][0]
tempSlope = coeffs[1][0]
precSlope = coeffs[2][0]

print("Multilinear Ridge Regression Predicted Prices S="+str(s))
print("Bias: " + str(bias))
print("Temperature Slope: " + str(tempSlope))
print("Precipitation Slope: " + str(precSlope))

residuals = np.zeros(len(dollars))
predictions = np.zeros(len(dollars))
for i in range(len(dollars)):
  predictions[i] = inflation[i]*tempSlope + interest_rates[i]*precSlope + bias
  residuals[i] = dollars[i] - predictions[i]

MSE = np.mean(residuals**2)
print("Mean Squared Error = " + str(MSE) +"\n\n")

plt.plot(predictions)
plt.plot(dollars)
plt.xlabel("Date")
plt.ylabel("Price per Ton ($USD)")
plt.title("Multilinear Ridge Regression Predicted Prices S=" + str(s))
plt.show()

# Slightly better mean errors...
#-------------------------------------------------------------------------------