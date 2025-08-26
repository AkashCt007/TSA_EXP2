## Developed by: AKASH CT
## Register no: 212224240007

# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
## Date:26/8/25
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```
import numpy as np
import matplotlib.pyplot as plt

df_yearly = df.groupby("year")["rating"].mean().reset_index()

years = df_yearly['year'].tolist()
ratings = df_yearly['rating'].tolist()

X = [i - years[len(years)//2] for i in years]   
x2 = [i**2 for i in X]
xy = [i*j for i, j in zip(X, ratings)]

n = len(years)
b = (n*sum(xy) - sum(ratings)*sum(X)) / (n*sum(x2) - (sum(X)**2))
a = (sum(ratings) - b*sum(X)) / n
linear_trend = [a + b*X[i] for i in range(n)]

x3 = [i**3 for i in X]
x4 = [i**4 for i in X]
x2y = [i*j for i, j in zip(x2, ratings)]

coeff = [
    [len(X), sum(X), sum(x2)],
    [sum(X), sum(x2), sum(x3)],
    [sum(x2), sum(x3), sum(x4)]
]
Y = [sum(ratings), sum(xy), sum(x2y)]

solution = np.linalg.solve(np.array(coeff), np.array(Y))
a_poly, b_poly, c_poly = solution
poly_trend = [a_poly + b_poly*X[i] + c_poly*(X[i]**2) for i in range(n)]

plt.figure(figsize=(12,6))
plt.plot(years, ratings, 'o-', color='blue', label='Avg Rating per Year')
plt.plot(years, linear_trend, '--', color='red', label='Linear Trend')
plt.plot(years, poly_trend, '-', color='green', label='Polynomial Trend')
plt.title("IMDb Average Ratings Trend over Years")
plt.xlabel("Year")
plt.ylabel("Average Rating")
plt.legend()
plt.show()

```

### OUTPUT
<img width="1430" height="653" alt="image" src="https://github.com/user-attachments/assets/3470c18a-9d74-4f4c-9a0f-419cb879afdf" />



### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
