import pandas as pd
import matplotlib.pyplot as plt

idx = range(260)

chartFrame = pd.DataFrame(index=idx, columns=['mom120', 'dd'])

chartFrame['mom120'] = 1/120
hh = list(idx)
temp = [(260 - h)/(130*260) for h in hh]
chartFrame['dd'] = temp

plt.plot(chartFrame)
plt.show()

print(chartFrame)