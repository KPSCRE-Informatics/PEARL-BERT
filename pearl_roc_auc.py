###############################################################
# purpose: calculate auc and create roc
# input data source: prediction from fine-tune bert models
###############################################################

# import packages
###############################################################
import sys
import numpy as np
import pandas as pd
import scipy
from IPython.display import clear_output
import matplotlib.pyplot as plt
import os.path
from sklearn.metrics import roc_curve,roc_auc_score, auc
import matplotlib.pyplot as plt
###############################################################

# read predicted data, please change the file name to the your predicted dataset
df = pd.read_csv("study/data/simulated_sample_prediction.csv")

# drop unneccessary 
df = df.drop(columns=['label_0','label_1'])

# Convert the DataFrame to a NumPy array
pred = df.values

#print("auc:",roc_auc_score(y_pred[:,1],y_pred[:,0]))
print("auc:",roc_auc_score(pred[:,0],pred[:,1]))

# calculating roc curve
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr[1], tpr[1], _ = roc_curve(pred[:,0],pred[:,1])
roc_auc[1] = auc(fpr[1], tpr[1])

#  plot roc
fig=plt.figure(figsize=(8,8))
plt.rc('axes',labelsize=8)
plt.rc('axes',titlesize=8)
plt.rc('legend',fontsize=8)
plt.rc('figure',titlesize=10)

plt.subplot(1,2,1)
plt.axis('square')
lw = 2
plt.plot(
    fpr[1],
    tpr[1],
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc[1],
        )
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.show()

# save roc figure
fig.savefig("study/data/simulated_sample_prediction_roc.png")
