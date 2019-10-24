import os
import pandas as pd
import numpy as np
import math
import statistics
import random
random.seed()
import matplotlib.pyplot as plt
plt.style.use('ggplot')

m1 = np.matrix('1 1 1; 2 2 2; 3 3 3')
m2 = np.matrix('4 4 4; 5 5 5')
m3 = np.matrix('6 6 6; 7 7 7')

a = [m1,m2,m3]

print(np.concatenate(a))