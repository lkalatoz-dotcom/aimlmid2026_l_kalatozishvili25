import numpy as np
import matplotlib.pyplot as plt

# Correct extracted data from blue dots
x = np.array([-8.9, -6.7, -4, -2, 1, 2.6, 4.5, 6, 8.9])
y = np.array([-7, -5, -2, 0.8, 1, 3, 4, 6.5, 8])

# Pearson correlation coefficient
r = np.corrcoef(x, y)[0, 1]
print("Pearson correlation coefficient:", r)

# Scatter plot
plt.scatter(x, y)
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Scatter plot of extracted data points")
plt.show()
