import sys
import matplotlib.pyplot as plt
from numpy import polyfit, poly1d
from scipy import stats

# Assuming `content` is a list of lines from your file, where each line looks like:
# '0.569921 vs -0.211127\t-1.228837 vs 0.122418\t...', you can replace the reading part
# with the actual file reading if starting from scratch.

n = int(sys.argv[1:][0])

# Initialize lists to hold the x and y values
x_values = []
y_values = []

file_path = '/Users/wuheng/Documents/github/pop_gen_cnn/demography/result_file'
with open(file_path, 'r') as file:
    content = file.readlines()


# Process each line to extract x and y values
for line in content:
    # Split the line into pairs
    pairs = line.strip().split('\t')
    x, y = pairs[n].split(' vs ')
    # Append the values to the lists, converting them to floats
    x_values.append(float(x))
    y_values.append(float(y))

res = stats.spearmanr(x_values, y_values)
print('Correlation: ' + str(res.correlation))
print('p-value: ' + str(res.pvalue))

# Compute a linear fit
coefficients = polyfit(x_values, y_values, 1)
print('y = ' + str(coefficients[0]) + 'x + ' + str(coefficients[1]))
fit_function = poly1d(coefficients)

# Generate y-values for the fit line using the fit function
fit_y_values = fit_function(x_values)

# Plotting with fit line
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, alpha=0.5, label='Data Points')  # Plotting the scatter points
plt.plot(x_values, fit_y_values, color='red', label='Fit Line')  # Plotting the fit line

# Adding the fit line equation as text
# Calculating a point along the fit line for placing the text
text_x = min(x_values) - 0.3
text_y = fit_function(text_x) + 0.5
plt.text(text_x, text_y, 'y = {:.2f}x {} {:.2f}'.format(coefficients[0], '-' if coefficients[1] < 0 else '+', abs(coefficients[1])), fontsize=12, color='black', verticalalignment='bottom')

# Adding Spearman correlation and p-value
#stats_text = 'Spearman Correlation: {:.2f}\nP-value: {:.2e}'.format(res.correlation, res.pvalue)
#plt.text(max(x_values)-3.5, max(fit_y_values)-1, stats_text, fontsize=12, verticalalignment='bottom')

if n == 0:
    plt.title('N0')
elif n == 1:
    plt.title('T1')
elif n == 2:
    plt.title('N1')
elif n == 3:
    plt.title('T2')
else:
    plt.title('N2')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.legend()
plt.show()
