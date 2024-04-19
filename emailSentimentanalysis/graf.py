import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Veri setini yükle
data = pd.read_csv("emails.csv")

# x, y ve z koordinatlarını belirle
x = data["x"]
y = data["y"]
z = data["z"]

# 3D grafik nesnesi oluştur
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Grafik çizimi
ax.scatter(x, y, z)

# Eksen etiketleri
ax.set_xlabel('X Ekseni')
ax.set_ylabel('Y Ekseni')
ax.set_zlabel('Z Ekseni')

# Grafik başlığı
ax.set_title('3D Grafik')

# Grafik gösterimi
plt.show()

