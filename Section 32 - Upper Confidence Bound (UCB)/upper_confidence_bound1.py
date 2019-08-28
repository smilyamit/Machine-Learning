# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import math
N = 10000
d = 10
numbers_of_selections = [1] * d
sums_of_rewards = [1] * d
 
ads_selected = []
for n in range(0,N):
    average_reward = np.divide(numbers_of_selections,sums_of_rewards)
    nominator = [math.sqrt( (3/2) * math.log(n+1) )] * d
    delta = np.divide(nominator, numbers_of_selections)
    upper_bound = average_reward + delta
    
    ad_to_show = np.argmax(upper_bound)
    ads_selected.append(ad_to_show)
    numbers_of_selections[ad_to_show] = numbers_of_selections[ad_to_show] + dataset.values[n,ad_to_show]
    sums_of_rewards[ad_to_show] = sums_of_rewards[ad_to_show] +1
 
print(sum( numbers_of_selections)-d)

plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()