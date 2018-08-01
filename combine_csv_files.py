import re
import os
import csv
import pandas as pd
os.chdir("C:/Users/dogus/Dropbox/DgsPy_DBOX/Hepsiburada/comments/")
filenames = os.listdir()

##x = pd.read_csv('HBV0000079GT1_2.csv', header=None)
##print(x)

comb = pd.concat( [pd.read_csv(f) for f in filenames])
comb.to_csv('hepsib_cellphone_comments.csv', index=False)
