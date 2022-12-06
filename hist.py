import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import os
from PIL import Image

nb_bins=256
count=np.zeros(nb_bins)
n=0

for path in os.listdir('./spectrograms-validation-images/'):
  for image in os.listdir('./spectrograms-validation-images/'+path):
    #print(image)
    img = Image.open('./spectrograms-validation-images/'+path+'/'+image)
    x = np.array(img)
    x = x.transpose(2,0,1)
    hist = np.histogram(x[0], bins=nb_bins, range=[0,255])
    count += hist[0]
    n=n+1
    if (not(n%1000)):
      print(n)
      #print(count)
    #if (n>10000):
    #  break
  else:
    continue
  break
bins = hist[1]
fig=plt.figure()
plt.bar(bins[1:-2], count[1:-1]/n)
plt.title('Pixel Histogram')
plt.xlabel('Bin')
plt.ylabel('Probability')
plt.savefig('histogram.png')
plt.show()
