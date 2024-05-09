import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
data = np.loadtxt('',dtype=float)

fft = np.fft.fft(data)
amp = np.abs(fft)/len(data)*2.
freq = np.fft.fftfreq(data.size, d=1/365.25)
plt.plot(freq[freq >= 0], amp[freq >= 0])
plt.xlim(0, 5)
plt.title('半周年项信号[单位：ms]')
plt.show()
