# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
from pyts.datasets import load_gunpoint
import csv

# Load the GunPoint dataset
X, _, _, _ = load_gunpoint(return_X_y=True)
print(X.shape)

csv_path = '../data/processed_data/目标光谱库/20.csv'
# csv_path = './30条参考光谱.csv'
with open(csv_path,'r',encoding='utf8')as fp:
    # 使用列表推导式，将读取到的数据装进列表
    reader = csv.reader(fp)
    X = []
    # 读取行数
    count = 0
    for row in reader:
        data = list(map(float,row))
        X.append(data)
        count += 1
    print(count)

# Get the Gramian angular summation fields for all the time series
gaf = GramianAngularField()
X_gaf = gaf.fit_transform(X)

# Plot the 50 Gramian angular fields
fig = plt.figure(figsize=(10, 5))

grid = ImageGrid(fig, 111, nrows_ncols=(3, 10), axes_pad=0.1, share_all=True,
                 cbar_mode='single')
print(grid)
for i, ax in enumerate(grid):
    im = ax.imshow(X_gaf[i], cmap='rainbow', origin='lower', vmin=-1., vmax=1.)
grid[0].get_yaxis().set_ticks([])
grid[0].get_xaxis().set_ticks([])
plt.colorbar(im, cax=grid.cbar_axes[0])
ax.cax.toggle_label(True)

'''s
fig.suptitle("Gramian angular summation fields for the 50 time series in the "
             "'GunPoint' dataset", y=0.92)
'''

plt.show()