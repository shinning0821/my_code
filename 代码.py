import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.cluster import k_means_

path = r"D:\latex\概率论论文\test.png"
img = np.array(plt.imread(path))
imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
imgGray = imgGray / 255
imgCopy = imgGray.copy()
imgpixel = (imgCopy.flatten()).reshape((imgGray.shape[0]*imgGray.shape[1], 1))
kind = 2
kmeans = k_means_.KMeans(n_clusters=kind)
label = kmeans.fit(imgpixel)
imgLabel = np.array(label.labels_).reshape(imgGray.shape)
plt.figure()
plt.imshow(imgLabel, cmap="gray")
imgMrf = np.zeros_like(imgLabel)
cycle = 10
c = 0
sumList = [0] * kind
numList = [0] * kind
MeanList = [0] * kind
stdSumList = [0] * kind
stdList = [0] * kind
for i in range(1, imgLabel.shape[0] - 1):
    for j in range(1, imgLabel.shape[1] - 1):
        x = imgLabel[i, j]
        sumList[x] += imgGray[i, j]
        numList[x] += 1
for k in range(kind):
    MeanList[k] = sumList[k] / numList[k]

for i in range(1, imgLabel.shape[0] - 1):
    for j in range(1, imgLabel.shape[1] - 1):
        x = imgLabel[i, j]
        stdSumList[x] += (imgGray[i, j] - MeanList[x]) ** 2
for i in range(kind):
    stdList[i] = np.sqrt(stdSumList[i] / numList[i])
def gas(mean, std, x):
    return 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-0.5 * (x - mean)**2 / std**2)

while c < cycle:
    for i in range(1, imgLabel.shape[0] - 1):
        for j in range(1, imgLabel.shape[1] - 1):
            uList = [0] * kind
            for k in range(kind):
                template = np.ones((3, 3)) * k
                template[1, 1] = np.inf
                u = np.exp(- np.sum(template == imgLabel[i - 1: i + 2, j - 1: j + 2]) + 8) *\
                    gas(MeanList[k], stdList[k], imgGray[i, j])
                uList[k] = u
                sumList[k] += imgGray[i, j]
                numList[k] += 1
            imgMrf[i, j] = uList.index(max(uList))
    for i in range(kind):
        MeanList[i] = sumList[i] / numList[i]
    for i in range(1, imgLabel.shape[0] - 1):
        for j in range(1, imgLabel.shape[1] - 1):
            x = imgLabel[i, j]
            stdSumList[x] += (imgGray[i, j] - MeanList[x]) ** 2
    for i in range(kind):
        stdList[i] = np.sqrt(stdSumList[i] / numList[i])
    imgLabel = imgMrf.copy()
    c += 1
    print("第{}代结束".format(c))
plt.figure()
plt.imshow(imgLabel, cmap="gray")
plt.figure()
plt.imshow(1-imgLabel, cmap="gray")
plt.show()
