import matplotlib.pyplot as plt
import numpy as np

loss     = [0.0148,0.0116,0.0103,0.0090,0.0079,0.0071,0.0062,0.0055,0.0049,0.0044,0.0040,0.0036,0.0032,0.0030,0.0027,0.0024,0.0023,0.0021,0.0020,0.0018]
val_loss = [0.0261,0.0248,0.0196,0.0189,0.0202,0.0171,0.0166,0.0187,0.0182,0.0163,0.0188,0.0163,0.0179,0.0168,0.0174,0.0171,0.0168,0.0162,0.0174,0.0168]

nb_epoch = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
print(len(nb_epoch))
plt.plot(nb_epoch, loss,     marker='.', label='loss')
plt.plot(nb_epoch, val_loss, marker='.', label='val_loss')
plt.legend(loc='best', fontsize=10)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
