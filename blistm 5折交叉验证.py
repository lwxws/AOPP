from __future__ import print_function, division
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, concatenate
from keras.models import Model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
import sys
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection  import train_test_split
from sklearn.metrics import *
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from numpy import array,argmax,linalg as la
from keras.preprocessing.sequence import pad_sequences
import os
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import random
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from numpy import linalg as la
from numpy.linalg import eig
from plot_keras_history import plot_history
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras import initializers
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.layers import Dropout, Input, \
    Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# from sklearn.externals import joblib

pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=np.inf)



from keras.preprocessing.sequence import pad_sequences
def fasta2csv(inFasta):
    FastaRead = pd.read_csv(inFasta, header=None)
    print(FastaRead.shape)
    print(FastaRead.head())
    seqNum = int(FastaRead.shape[0] / 2)
    csvFile = open("testFasta.csv", "w")
    csvFile.write("PID,Seq\n")

    # print("Lines:",FastaRead.shape)
    # print("Seq Num:",seqNum)
    for i in range(seqNum):
        csvFile.write(str(FastaRead.iloc[2 * i, 0]) + "," + str(FastaRead.iloc[2 * i + 1, 0]) + "\n")

    csvFile.close()
    TrainSeqLabel = pd.read_csv("testFasta.csv", header=0)
    path = "testFasta.csv"
    if os.path.exists(path):
        os.remove(path)

    return TrainSeqLabel
inFastaTrain=r"C:\Users\41822\Desktop\DL\新数据集\Total-train.fasta"
inFastaTest=r"C:\Users\41822\Desktop\DL\新数据集\Total-test.fasta"

mainTrain = fasta2csv(inFastaTrain)
mainTest = fasta2csv(inFastaTest)

i=0
mainTrain["Tags"]=mainTrain["Seq"]
for pid in mainTrain["PID"]:
  mainTrain["Tags"][i]=pid[len(pid)-1]
  if mainTrain["Tags"][i]=="1":
    mainTrain["Tags"][i]=1
  else:
    mainTrain["Tags"][i]=0
  i=i+1
i=0
mainTest["Tags"]=mainTest["Seq"]
for pid in mainTest["PID"]:
  mainTest["Tags"][i]=pid[len(pid)-1]
  if mainTest["Tags"][i]=="1":
    mainTest["Tags"][i]=1
  else:
    mainTest["Tags"][i]=0
  i=i+1
AOP_y_train = mainTrain["Tags"].values
AOP_y_test = mainTest["Tags"].values
AOP_y_train_ = np.array([np.array(i) for i in AOP_y_train])
AOP_y_test_ = np.array([np.array(i) for i in AOP_y_test])
x_train = {}
protein_index = 1
for line in mainTrain["Seq"]:
  x_train[protein_index] = line
  protein_index = protein_index + 1
maxlen_train = max(len(x) for x in x_train.values())

x_test = {}
protein_index = 1
for line in mainTest["Seq"]:
  x_test[protein_index] = line
  protein_index = protein_index + 1
maxlen_test = max(len(x) for x in x_test.values())
maxlen = max(maxlen_train,maxlen_test)
def OE(seq_temp):
    seq = seq_temp
    chars = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y']
    fea = []
    #k = 6
    for i in range(len(seq)):
        if seq[i] =='A':
            tem_vec = 1
        elif seq[i]=='C':
            tem_vec = 2
        elif seq[i]=='D':
            tem_vec = 3
        elif seq[i]=='E' or seq[i]=='U':
            tem_vec = 4
        elif seq[i]=='F':
            tem_vec = 5
        elif seq[i]=='G':
            tem_vec = 6
        elif seq[i]=='H':
            tem_vec = 7
        elif seq[i]=='I':
            tem_vec = 8
        elif seq[i]=='K':
            tem_vec = 9
        elif seq[i]=='L':
            tem_vec = 10
        elif seq[i]=='M' or seq[i]=='O':
            tem_vec = 11
        elif seq[i]=='N':
            tem_vec = 12
        elif seq[i]=='P':
            tem_vec = 13
        elif seq[i]=='Q':
            tem_vec = 14
        elif seq[i]=='R':
            tem_vec = 15
        elif seq[i]=='S':
            tem_vec = 16
        elif seq[i]=='T':
            tem_vec = 17
        elif seq[i]=='V':
            tem_vec = 18
        elif seq[i]=='W':
            tem_vec = 19
        elif seq[i]=='X' or seq[i]=='B' or seq[i]=='Z':
            tem_vec = 20
        elif seq[i]=='Y':
            tem_vec = 21
        #fea = fea + tem_vec +[i]
        fea.append(tem_vec)
    return fea

x_train_oe = []
for i in x_train:
  oe_feature = OE(x_train[i])
  x_train_oe.append(oe_feature)
  #print(protein_seq_dict[i])
x_test_oe = []
for i in x_test:
  oe_feature = OE(x_test[i])
  x_test_oe.append(oe_feature)
x_train_ = np.array(pad_sequences(x_train_oe, padding='post', maxlen=maxlen))
x_test_ = np.array(pad_sequences(x_test_oe, padding='post', maxlen=maxlen))
print(x_train_ )
print(x_test_ )

handcraft_AAC_train = [[0] * 21 for _ in range(len(x_train_oe))]
for row in range(len(x_train_oe)):
  seq = x_train_oe[row]
  for i in seq:
    col = i-1
    handcraft_AAC_train[row][col] += 1/len(seq)
hc_AAC_train = np.array(handcraft_AAC_train)

handcraft_AAC_test = [[0] * 21 for _ in range(len(x_test_oe))]
for row in range(len(x_test_oe)):
  seq = x_test_oe[row]
  for i in seq:
    col = i-1
    handcraft_AAC_test[row][col] += 1/len(seq)
hc_AAC_test = np.array(handcraft_AAC_test)

comb = []
for i in range(1,22):
  for j in range(i,22):
    comb.append([i,j])
comb_index = {}
for i in range(len(comb)):
  comb_index[tuple(comb[i])] = i
print(comb)
handcraft_DPC_train = [[0] * len(comb) for _ in range(len(x_train_oe))]
for row in range(len(x_train_oe)):
  seq = x_train_oe[row]
  for i in range(len(seq)-1):
    a = sorted([seq[i],seq[i+1]])
    index = comb_index[tuple(a)]
    handcraft_DPC_train[row][index] += 1/(len(seq)-1)
hc_DPC_train = np.array(handcraft_DPC_train)
print(hc_DPC_train)
handcraft_DPC_test = [[0] * len(comb) for _ in range(len(x_test_oe))]
for row in range(len(x_test_oe)):
  seq = x_test_oe[row]
  for i in range(len(seq)-1):
    a = sorted([seq[i],seq[i+1]])
    index = comb_index[tuple(a)]
    handcraft_DPC_test[row][index] += 1/(len(seq)-1)
hc_DPC_test = np.array(handcraft_DPC_test)
print(hc_DPC_test )
def readFasta(file):
    if os.path.exists(file) == False:
        print('Error: "' + file + '" does not exist.')
        sys.exit(1)

    with open(file) as f:
        records = f.read()

    if re.search('>', records) == None:
        print('The input file seems not in fasta format.')
        sys.exit(1)

    records = records.split('>')[1:]
    myFasta = []
    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper())
        myFasta.append([name, sequence])

    return myFasta
def generateGroupPairs(groupKey):
    gPair = {}
    for key1 in groupKey:
        for key2 in groupKey:
            gPair[key1+'.'+key2] = 0
    return gPair


def CKSAAGP(fastas, gap = 1, **kw):

    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    AA = 'ARNDCQEGHILKMFPSTWYV'

    groupKey = group.keys()

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    gPairIndex = []
    for key1 in groupKey:
        for key2 in groupKey:
            gPairIndex.append(key1+'.'+key2)

    encodings = []
    header = ['#']
    for g in range(gap + 1):
        for p in gPairIndex:
            header.append(p+'.gap'+str(g))
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = [name]
        for g in range(gap + 1):
            gPair = generateGroupPairs(groupKey)
            sum = 0
            for p1 in range(len(sequence)):
                p2 = p1 + g + 1
                if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
                    gPair[index[sequence[p1]]+'.'+index[sequence[p2]]] = gPair[index[sequence[p1]]+'.'+index[sequence[p2]]] + 1
                    sum = sum + 1

            if sum == 0:
                for gp in gPairIndex:
                    code.append(0)
            else:
                for gp in gPairIndex:
                    code.append(gPair[gp] / sum)

        encodings.append(code)

    return encodings

handcraft_CKSAAGP_train = CKSAAGP(readFasta(inFastaTrain))
handcraft_CKS_train = []
for i in range(1,len(handcraft_CKSAAGP_train)):
  handcraft_CKS_train.append(handcraft_CKSAAGP_train[i][1:])
hc_CKS_train = np.array(handcraft_CKS_train)
print(hc_CKS_train )
handcraft_CKSAAGP_test = CKSAAGP(readFasta(inFastaTest))
handcraft_CKS_test = []
for i in range(1,len(handcraft_CKSAAGP_test)):
  handcraft_CKS_test.append(handcraft_CKSAAGP_test[i][1:])
hc_CKS_test = np.array(handcraft_CKS_test)
print(hc_CKS_test )

def TransDict_from_list(groups):
  transDict = dict()
  tar_list = ['0', '1', '2', '3', '4', '5', '6']
  result = {}
  index = 0
  for group in groups:
    g_members = sorted(group)  # Alphabetically sorted list
    for c in g_members:
        # print('c' + str(c))
        # print('g_members[0]' + str(g_members[0]))
        result[c] = str(tar_list[index])  # K:V map, use group's first letter as represent.
    index = index + 1
  return result


def translate_sequence(seq, TranslationDict):
  '''
  Given (seq) - a string/sequence to translate,
  Translates into a reduced alphabet, using a translation dict provided
  by the TransDict_from_list() method.
  Returns the string/sequence in the new, reduced alphabet.
  Remember - in Python string are immutable..
  '''
  import string
  from_list = []
  to_list = []
  for k, v in TranslationDict.items():
      from_list.append(k)
      to_list.append(v)
  # TRANS_seq = seq.translate(str.maketrans(zip(from_list,to_list)))
  TRANS_seq = seq.translate(str.maketrans(str(from_list), str(to_list)))
  # TRANS_seq = maketrans( TranslationDict, seq)
  return TRANS_seq
def get_3_protein_trids():
  nucle_com = []
  chars = ['0', '1', '2', '3', '4', '5', '6']
  base = len(chars)
  end = len(chars) ** 3
  for i in range(0, end):
      n = i
      ch0 = chars[n % base]
      n = n / base
      ch1 = chars[int(n % base)]
      n = n / base
      ch2 = chars[int(n % base)]
      nucle_com.append(ch0 + ch1 + ch2)
  return nucle_com
def get_4_nucleotide_composition(tris, seq, pythoncount=True):
  seq_len = len(seq)
  tri_feature = [0] * len(tris)
  k = len(tris[0])
  note_feature = [[0 for cols in range(len(seq) - k + 1)] for rows in range(len(tris))]
  if pythoncount:
      for val in tris:
          num = seq.count(val)
          tri_feature.append(float(num) / seq_len)
  else:
      # tmp_fea = [0] * len(tris)
      for x in range(len(seq) + 1 - k):
          kmer = seq[x:x + k]
          if kmer in tris:
              ind = tris.index(kmer)
              # tmp_fea[ind] = tmp_fea[ind] + 1
              note_feature[ind][x] = note_feature[ind][x] + 1
      # tri_feature = [float(val)/seq_len for val in tmp_fea]    #tri_feature type:list len:256
      u, s, v = la.svd(note_feature)
      for i in range(len(s)):
          tri_feature = tri_feature + u[i] * s[i] / seq_len
      # print tri_feature
      # pdb.set_trace()

  return tri_feature
def prepare_feature_kmer(infile):
  protein_seq_dict = {}
  protein_index = 1
  with open(infile, 'r') as fp:
    for line in fp:
      if line[0] != '>':
        seq = line[:-1]
        protein_seq_dict[protein_index] = seq
        protein_index = protein_index + 1
  kmer = []
  groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
  group_dict = TransDict_from_list(groups)
  protein_tris = get_3_protein_trids()
  # get protein feature
  # pdb.set_trace()
  for i in protein_seq_dict:  # and protein_fea_dict.has_key(protein) and RNA_fea_dict.has_key(RNA):
    protein_seq = translate_sequence(protein_seq_dict[i], group_dict)
    # print('oe:',shape(oe_feature))
    # pdb.set_trace()
    # RNA_tri_fea = get_4_nucleotide_composition(tris, RNA_seq, pythoncount=False)
    protein_tri_fea = get_4_nucleotide_composition(protein_tris, protein_seq, pythoncount =False)
    kmer.append(protein_tri_fea)
    protein_index = protein_index + 1
    # chem_fea.append(chem_tmp_fea)
  return np.array(kmer)

kmer_train = prepare_feature_kmer(inFastaTrain)
kmer_test = prepare_feature_kmer(inFastaTest)

#AAindex提取
def aaindex(sequence):
    sequence = sequence.rstrip()

    ANDN920101={'A':4.35,'L':4.17,'R':4.38,'K':4.36,'N':4.75,
    'M':4.52,'D':4.76,'F':4.66,'C':4.65,'P':4.44,
    'Q':4.37,'S':4.50,'E':4.29,'T':4.35,'G':3.97,
    'W':4.70,'H':4.63,'Y':4.60,'I':3.95,'V':3.95}

    ARGP820101={'A':0.61,'L':1.53,'R':0.60,'K':1.15,'N':0.06,
    'M':1.18,'D':0.46,'F':2.02,'C':1.07,'P':1.95,
    'Q':0.0,'S':0.05,'E':0.47,'T':0.05,'G':0.07,
    'W':2.65,'H':0.61,'Y':1.88,'I':2.22,'V':1.32}

    ARGP820102={'A':1.18,'L':3.23,'R':0.20,'K':0.06,'N':0.23,
    'M':2.67,'D':0.05,'F':1.96,'C':1.89,'P':0.76,
    'Q':0.72,'S':0.97,'E':0.11,'T':0.84,'G':0.49,
    'W':0.77,'H':0.31,'Y':0.39,'I':1.45,'V':1.08}

    ARGP820103={'A':1.56,'L':2.93,'R':0.45,'K':0.15,'N':0.27,
    'M':2.96,'D':0.14,'F':2.03,'C':1.23,'P':0.76,
    'Q':0.51,'S':0.81,'E':0.23,'T':0.91,'G':0.62,
    'W':1.08,'H':0.29,'Y':0.68,'I':1.67,'V':1.14}

    BEGF750101={'A':1.0,'L':1.0,'R':0.52,'K':0.60,'N':0.35,
    'M':1.0,'D':0.44,'F':0.60,'C':0.06,'P':0.06,
    'Q':0.44,'S':0.35,'E':0.73,'T':0.44,'G':0.35,
    'W':0.73,'H':0.60,'Y':0.44,'I':0.73,'V':0.82}

    BEGF750102={'A':0.77,'L':0.83,'R':0.72,'K':0.55,'N':0.55,
    'M':0.98,'D':0.65,'F':0.98,'C':0.65,'P':0.55,
    'Q':0.72,'S':0.55,'E':0.55,'T':0.83,'G':0.65,
    'W':0.77,'H':0.83,'Y':0.83,'I':0.98,'V':0.98}

    BEGF750103={'A':0.37,'L':0.53,'R':0.84,'K':0.75,'N':0.97,
    'M':0.64,'D':0.97,'F':0.53,'C':0.84,'P':0.97,
    'Q':0.64,'S':0.84,'E':0.53,'T':0.75,'G':0.97,
    'W':0.97,'H':0.75,'Y':0.84,'I':0.37,'V':0.37}

    BHAR880101={'A':0.357,'L':0.365,'R':0.529,'K':0.466,'N':0.463,
    'M':0.295,'D':0.511,'F':0.314,'C':0.346,'P':0.509,
    'Q':0.493,'S':0.507,'E':0.497,'T':0.444,'G':0.544,
    'W':0.305,'H':0.323,'Y':0.420,'I':0.462,'V':0.386}

    BIGC670101={'A':52.6,'L':102.0,'R':109.1,'K':105.1,'N':75.7,
    'M':97.7,'D':68.4,'F':113.9,'C':68.3,'P':73.6,
    'Q':89.7,'S':54.9,'E':84.7,'T':71.2,'G':36.3,
    'W':135.4,'H':91.9,'Y':116.2,'I':102.0,'V':85.1}

    BIOV880101={'A':16.0,'L':145.0,'R':-70.0,'K':-141.0,'N':-74.0,
    'M':124.0,'D':-78.0,'F':189.0,'C':168.0,'P':-20.0,
    'Q':-73.0,'S':-70.0,'E':-106.0,'T':-38.0,'G':-13.0,
    'W':145.0,'H':50.0,'Y':53.0,'I':151.0,'V':123.0}

    BIOV880102={'A':44.0,'L':108.0,'R':-68.0,'K':-188.0,'N':-72.0,
    'M':121.0,'D':-91.0,'F':148.0,'C':90.0,'P':-36.0,
    'Q':-117.0,'S':-60.0,'E':-139.0,'T':-54.0,'G':-8.0,
    'W':163.0,'H':47.0,'Y':22.0,'I':100.0,'V':117.0}

    BROC820101={'A':7.3,'L':20.0,'R':-3.6,'K':-3.7,'N':-5.7,
    'M':5.6,'D':-2.9,'F':19.2,'C':-9.2,'P':5.1,
    'Q':-0.3,'S':-4.1,'E':-7.1,'T':0.8,'G':-1.2,
    'W':16.3,'H':-2.1,'Y':5.9,'I':6.6,'V':3.5}

    BROC820102={'A':3.9,'L':15.0,'R':3.2,'K':-2.5,'N':-2.8,
    'M':4.1,'D':-2.8,'F':14.7,'C':-14.3,'P':5.6,
    'Q':1.8,'S':-3.5,'E':-7.5,'T':1.1,'G':-2.3,
    'W':17.8,'H':2.0,'Y':3.8,'I':11.0,'V':2.1}

    BULH740101={'A':-0.20,'L':-2.46,'R':-0.12,'K':-0.35,'N':0.08,
    'M':-1.47,'D':-0.20,'F':-2.33,'C':-0.45,'P':-0.98,
    'Q':0.16,'S':-0.39,'E':-0.30,'T':-0.52,'G':0.00,
    'W':-2.01,'H':-0.12,'Y':-2.24,'I':-2.26,'V':-1.56}

    BULH740102={'A':0.691,'L':0.842,'R':0.728,'K':0.767,'N':0.596,
    'M':0.709,'D':0.558,'F':0.756,'C':0.624,'P':0.730,
    'Q':0.649,'S':0.594,'E':0.632,'T':0.655,'G':0.592,
    'W':0.743,'H':0.646,'Y':0.743,'I':0.809,'V':0.777}

    BUNA790101={'A':8.249,'L':8.423,'R':8.274,'K':8.408,'N':8.747,
    'M':8.418,'D':8.410,'F':8.228,'C':8.312,'P':0.0,
    'Q':8.411,'S':8.380,'E':8.368,'T':8.236,'G':8.391,
    'W':8.094,'H':8.415,'Y':8.183,'I':8.195,'V':8.436}

    BUNA790102={'A':4.349,'L':4.385,'R':4.396,'K':4.358,'N':4.755,
    'M':4.513,'D':4.765,'F':4.663,'C':4.686,'P':4.471,
    'Q':4.373,'S':4.498,'E':4.295,'T':4.346,'G':3.972,
    'W':4.702,'H':4.630,'Y':4.604,'I':4.224,'V':4.184}

    BUNA790103={'A':6.5,'L':6.5,'R':6.9,'K':6.5,'N':7.5,
    'M':0.0,'D':7.0,'F':9.4,'C':7.7,'P':0.0,
    'Q':6.0,'S':6.5,'E':7.0,'T':6.9,'G':5.6,
    'W':0.0,'H':8.0,'Y':6.8,'I':7.0,'V':7.0}

    BURA740101={'A':0.486,'L':0.420,'R':0.262,'K':0.402,'N':0.193,
    'M':0.417,'D':0.288,'F':0.318,'C':0.200,'P':0.208,
    'Q':0.418,'S':0.200,'E':0.538,'T':0.272,'G':0.120,
    'W':0.462,'H':0.400,'Y':0.161,'I':0.370,'V':0.379}

    BURA740102={'A':0.288,'L':0.400,'R':0.362,'K':0.265,'N':0.229,
    'M':0.375,'D':0.271,'F':0.318,'C':0.533,'P':0.340,
    'Q':0.327,'S':0.354,'E':0.262,'T':0.388,'G':0.312,
    'W':0.231,'H':0.200,'Y':0.429,'I':0.411,'V':0.495}

    CHAM810101={'A':0.52,'L':0.98,'R':0.68,'K':0.68,'N':0.76,
    'M':0.78,'D':0.76,'F':0.70,'C':0.62,'P':0.36,
    'Q':0.68,'S':0.53,'E':0.68,'T':0.50,'G':0.00,
    'W':0.70,'H':0.70,'Y':0.70,'I':1.02,'V':0.76}

    CHAM820101={'A':0.046,'L':0.186,'R':0.291,'K':0.219,'N':0.134,
    'M':0.221,'D':0.105,'F':0.290,'C':0.128,'P':0.131,
    'Q':0.180,'S':0.062,'E':0.151,'T':0.108,'G':0.000,
    'W':0.409,'H':0.230,'Y':0.298,'I':0.186,'V':0.140}

    CHAM820102={'A':-0.368,'L':1.07,'R':-1.03,'K':0.0,'N':0.0,
    'M':0.656,'D':2.06,'F':1.06,'C':4.53,'P':-2.24,
    'Q':0.731,'S':-0.524,'E':1.77,'T':0.0,'G':-0.525,
    'W':1.60,'H':0.0,'Y':4.91,'I':0.791,'V':0.401}

    CHAM830101={'A':0.71,'L':0.69,'R':1.06,'K':0.99,'N':1.37,
    'M':0.59,'D':1.21,'F':0.71,'C':1.19,'P':1.61,
    'Q':0.87,'S':1.34,'E':0.84,'T':1.08,'G':1.52,
    'W':0.76,'H':1.07,'Y':1.07,'I':0.66,'V':0.63}

    CHAM830102={'A':-0.118,'L':-0.052,'R':0.124,'K':0.032,'N':0.289,
    'M':-0.258,'D':0.048,'F':0.015,'C':0.083,'P':0.0,
    'Q':-0.105,'S':0.225,'E':-0.245,'T':0.166,'G':0.104,
    'W':0.158,'H':0.138,'Y':0.094,'I':0.230,'V':0.513}

    CHAM830103={'A':0.0,'L':1.0,'R':1.0,'K':1.0,'N':1.0,
    'M':1.0,'D':1.0,'F':1.0,'C':1.0,'P':0.0,
    'Q':1.0,'S':1.0,'E':1.0,'T':2.0,'G':0.0,
    'W':1.0,'H':1.0,'Y':1.0,'I':2.0,'V':2.0}

    CHAM830104={'A':0.0,'L':2.0,'R':1.0,'K':1.0,'N':1.0,
    'M':1.0,'D':1.0,'F':1.0,'C':0.0,'P':0.0,
    'Q':1.0,'S':0.0,'E':1.0,'T':0.0,'G':0.0,
    'W':1.0,'H':1.0,'Y':1.0,'I':1.0,'V':0.0}

    CHAM830105={'A':0.0,'L':0.0,'R':1.0,'K':1.0,'N':0.0,
    'M':1.0,'D':0.0,'F':1.0,'C':0.0,'P':0.0,
    'Q':1.0,'S':0.0,'E':1.0,'T':0.0,'G':0.0,
    'W':1.5,'H':1.0,'Y':1.0,'I':0.0,'V':0.0}

    CHAM830106={'A':0.0,'L':2.0,'R':5.0,'K':4.0,'N':2.0,
    'M':3.0,'D':2.0,'F':4.0,'C':1.0,'P':0.0,
    'Q':3.0,'S':1.0,'E':3.0,'T':1.0,'G':0.0,
    'W':5.0,'H':3.0,'Y':5.0,'I':2.0,'V':1.0}

    CHAM830107={'A':0.0,'L':0.0,'R':0.0,'K':0.0,'N':1.0,
    'M':0.0,'D':1.0,'F':0.0,'C':0.0,'P':0.0,
    'Q':0.0,'S':0.0,'E':1.0,'T':0.0,'G':1.0,
    'W':0.0,'H':0.0,'Y':0.0,'I':0.0,'V':0.0}

    CHAM830108={'A':0.0,'L':0.0,'R':1.0,'K':1.0,'N':1.0,
    'M':1.0,'D':0.0,'F':1.0,'C':1.0,'P':0.0,
    'Q':1.0,'S':0.0,'E':0.0,'T':0.0,'G':0.0,
    'W':1.0,'H':1.0,'Y':1.0,'I':0.0,'V':0.0}

    CHOC750101={'A':91.5,'L':167.9,'R':202.0,'K':171.3,'N':135.2,
    'M':170.8,'D':124.5,'F':203.4,'C':117.7,'P':129.3,
    'Q':161.1,'S':99.1,'E':155.1,'T':122.1,'G':66.4,
    'W':237.6,'H':167.3,'Y':203.6,'I':168.8,'V':141.7}

    CHOC760101={'A':115.0,'L':170.0,'R':225.0,'K':200.0,'N':160.0,
    'M':185.0,'D':150.0,'F':210.0,'C':135.0,'P':145.0,
    'Q':180.0,'S':115.0,'E':190.0,'T':140.0,'G':75.0,
    'W':255.0,'H':195.0,'Y':230.0,'I':175.0,'V':155.0}

    CHOC760102={'A':25.0,'L':23.0,'R':90.0,'K':97.0,'N':63.0,
    'M':31.0,'D':50.0,'F':24.0,'C':19.0,'P':50.0,
    'Q':71.0,'S':44.0,'E':49.0,'T':47.0,'G':23.0,
    'W':32.0,'H':43.0,'Y':60.0,'I':18.0,'V':18.0}

    CHOC760103={'A':0.38,'L':0.45,'R':0.01,'K':0.03,'N':0.12,
    'M':0.40,'D':0.15,'F':0.50,'C':0.45,'P':0.18,
    'Q':0.07,'S':0.22,'E':0.18,'T':0.23,'G':0.36,
    'W':0.27,'H':0.17,'Y':0.15,'I':0.60,'V':0.54}

    CHOC760104={'A':0.20,'L':0.16,'R':0.00,'K':0.00,'N':0.03,
    'M':0.11,'D':0.04,'F':0.14,'C':0.22,'P':0.04,
    'Q':0.01,'S':0.08,'E':0.03,'T':0.08,'G':0.18,
    'W':0.04,'H':0.02,'Y':0.03,'I':0.19,'V':0.18}

    CHOP780101={'A':0.66,'L':0.59,'R':0.95,'K':1.01,'N':1.56,
    'M':0.60,'D':1.46,'F':0.60,'C':1.19,'P':1.52,
    'Q':0.98,'S':1.43,'E':0.74,'T':0.96,'G':1.56,
    'W':0.96,'H':0.95,'Y':1.14,'I':0.47,'V':0.50}

    CHOP780201={'A':1.42,'L':1.21,'R':0.98,'K':1.16,'N':0.67,
    'M':1.45,'D':1.01,'F':1.13,'C':0.70,'P':0.57,
    'Q':1.11,'S':0.77,'E':1.51,'T':0.83,'G':0.57,
    'W':1.08,'H':1.00,'Y':0.69,'I':1.08,'V':1.06}

    CHOP780202={'A':0.83,'L':1.30,'R':0.93,'K':0.74,'N':0.89,
    'M':1.05,'D':0.54,'F':1.38,'C':1.19,'P':0.55,
    'Q':1.10,'S':0.75,'E':0.37,'T':1.19,'G':0.75,
    'W':1.37,'H':0.87,'Y':1.47,'I':1.60,'V':1.70}

    CHOP780203={'A':0.74,'L':0.50,'R':1.01,'K':1.19,'N':1.46,
    'M':0.60,'D':1.52,'F':0.66,'C':0.96,'P':1.56,
    'Q':0.96,'S':1.43,'E':0.95,'T':0.98,'G':1.56,
    'W':0.60,'H':0.95,'Y':1.14,'I':0.47,'V':0.59}

    CHOP780204={'A':1.29,'L':0.58,'R':0.44,'K':0.66,'N':0.81,
    'M':0.71,'D':2.02,'F':0.61,'C':0.66,'P':2.01,
    'Q':1.22,'S':0.74,'E':2.44,'T':1.08,'G':0.76,
    'W':1.47,'H':0.73,'Y':0.68,'I':0.67,'V':0.61}

    CHOP780205={'A':1.20,'L':1.13,'R':1.25,'K':1.83,'N':0.59,
    'M':1.57,'D':0.61,'F':1.10,'C':1.11,'P':0.00,
    'Q':1.22,'S':0.96,'E':1.24,'T':0.75,'G':0.42,
    'W':0.40,'H':1.77,'Y':0.73,'I':0.98,'V':1.25}

    CHOP780206={'A':0.70,'L':0.85,'R':0.34,'K':1.01,'N':1.42,
    'M':0.83,'D':0.98,'F':0.93,'C':0.65,'P':1.10,
    'Q':0.75,'S':1.55,'E':1.04,'T':1.09,'G':1.41,
    'W':0.62,'H':1.22,'Y':0.99,'I':0.78,'V':0.75}

    CHOP780207={'A':0.52,'L':0.84,'R':1.24,'K':1.49,'N':1.64,
    'M':0.52,'D':1.06,'F':1.04,'C':0.94,'P':1.58,
    'Q':0.70,'S':0.93,'E':0.59,'T':0.86,'G':1.64,
    'W':0.16,'H':1.86,'Y':0.96,'I':0.87,'V':0.32}

    CHOP780208={'A':0.86,'L':1.30,'R':0.90,'K':1.00,'N':0.66,
    'M':1.43,'D':0.38,'F':1.50,'C':0.87,'P':0.66,
    'Q':1.65,'S':0.63,'E':0.35,'T':1.17,'G':0.63,
    'W':1.49,'H':0.54,'Y':1.07,'I':1.94,'V':1.69}

    CHOP780209={'A':0.75,'L':1.27,'R':0.90,'K':0.74,'N':1.21,
    'M':0.95,'D':0.85,'F':1.50,'C':1.11,'P':0.40,
    'Q':0.65,'S':0.79,'E':0.55,'T':0.75,'G':0.74,
    'W':1.19,'H':0.90,'Y':1.96,'I':1.35,'V':1.79}

    CHOP780210={'A':0.67,'L':0.46,'R':0.89,'K':1.09,'N':1.86,
    'M':0.52,'D':1.39,'F':0.30,'C':1.34,'P':1.58,
    'Q':1.09,'S':1.41,'E':0.92,'T':1.09,'G':1.46,
    'W':0.48,'H':0.78,'Y':1.23,'I':0.59,'V':0.42}

    CHOP780211={'A':0.74,'L':0.59,'R':1.05,'K':0.82,'N':1.13,
    'M':0.85,'D':1.32,'F':0.44,'C':0.53,'P':1.69,
    'Q':0.77,'S':1.49,'E':0.85,'T':1.16,'G':1.68,
    'W':1.59,'H':0.96,'Y':1.01,'I':0.53,'V':0.59}

    CHOP780212={'A':0.060,'L':0.061,'R':0.070,'K':0.055,'N':0.161,
    'M':0.068,'D':0.147,'F':0.059,'C':0.149,'P':0.102,
    'Q':0.074,'S':0.120,'E':0.056,'T':0.086,'G':0.102,
    'W':0.077,'H':0.140,'Y':0.082,'I':0.043,'V':0.062}

    CHOP780213={'A':0.076,'L':0.025,'R':0.106,'K':0.115,'N':0.083,
    'M':0.082,'D':0.110,'F':0.041,'C':0.053,'P':0.301,
    'Q':0.098,'S':0.139,'E':0.060,'T':0.108,'G':0.085,
    'W':0.013,'H':0.047,'Y':0.065,'I':0.034,'V':0.048}

    CHOP780214={'A':0.035,'L':0.036,'R':0.099,'K':0.072,'N':0.191,
    'M':0.014,'D':0.179,'F':0.065,'C':0.117,'P':0.034,
    'Q':0.037,'S':0.125,'E':0.077,'T':0.065,'G':0.190,
    'W':0.064,'H':0.093,'Y':0.114,'I':0.013,'V':0.028}

    CHOP780215={'A':0.058,'L':0.070,'R':0.085,'K':0.095,'N':0.091,
    'M':0.055,'D':0.081,'F':0.065,'C':0.128,'P':0.068,
    'Q':0.098,'S':0.106,'E':0.064,'T':0.079,'G':0.152,
    'W':0.167,'H':0.054,'Y':0.125,'I':0.056,'V':0.053}

    CHOP780216={'A':0.64,'L':0.36,'R':1.05,'K':1.13,'N':1.56,
    'M':0.51,'D':1.61,'F':0.62,'C':0.92,'P':2.04,
    'Q':0.84,'S':1.52,'E':0.80,'T':0.98,'G':1.63,
    'W':0.48,'H':0.77,'Y':1.08,'I':0.29,'V':0.43}

    CIDH920101={'A':-0.45,'L':1.29,'R':-0.24,'K':-0.36,'N':-0.20,
    'M':1.37,'D':-1.52,'F':1.48,'C':0.79,'P':-0.12,
    'Q':-0.99,'S':-0.98,'E':-0.80,'T':-0.70,'G':-1.00,
    'W':1.38,'H':1.07,'Y':1.49,'I':0.76,'V':1.26}

    CIDH920102={'A':-0.08,'L':1.24,'R':-0.09,'K':-0.09,'N':-0.70,
    'M':1.27,'D':-0.71,'F':1.53,'C':0.76,'P':-0.01,
    'Q':-0.40,'S':-0.93,'E':-1.31,'T':-0.59,'G':-0.84,
    'W':2.25,'H':0.43,'Y':1.53,'I':1.39,'V':1.09}

    CIDH920103={'A':0.36,'L':1.18,'R':-0.52,'K':-0.56,'N':-0.90,
    'M':1.21,'D':-1.09,'F':1.01,'C':0.70,'P':-0.06,
    'Q':-1.05,'S':-0.60,'E':-0.83,'T':-1.20,'G':-0.82,
    'W':1.31,'H':0.16,'Y':1.05,'I':2.17,'V':1.21}

    CIDH920104={'A':0.17,'L':0.96,'R':-0.70,'K':-0.62,'N':-0.90,
    'M':0.60,'D':-1.05,'F':1.29,'C':1.24,'P':-0.21,
    'Q':-1.20,'S':-0.83,'E':-1.19,'T':-0.62,'G':-0.57,
    'W':1.51,'H':-0.25,'Y':0.66,'I':2.06,'V':1.21}

    CIDH920105={'A':0.02,'L':1.14,'R':-0.42,'K':-0.41,'N':-0.77,
    'M':1.00,'D':-1.04,'F':1.35,'C':0.77,'P':-0.09,
    'Q':-1.10,'S':-0.97,'E':-1.14,'T':-0.77,'G':-0.80,
    'W':1.71,'H':0.26,'Y':1.11,'I':1.81,'V':1.13}

    COHE430101={'A':0.75,'L':0.90,'R':0.70,'K':0.82,'N':0.61,
    'M':0.75,'D':0.60,'F':0.77,'C':0.61,'P':0.76,
    'Q':0.67,'S':0.68,'E':0.66,'T':0.70,'G':0.64,
    'W':0.74,'H':0.67,'Y':0.71,'I':0.90,'V':0.86}

    CRAJ730101={'A':1.33,'L':1.29,'R':0.79,'K':1.03,'N':0.72,
    'M':1.40,'D':0.97,'F':1.15,'C':0.93,'P':0.49,
    'Q':1.42,'S':0.83,'E':1.66,'T':0.94,'G':0.58,
    'W':1.33,'H':1.49,'Y':0.49,'I':0.99,'V':0.96}

    CRAJ730102={'A':1.00,'L':1.53,'R':0.74,'K':1.18,'N':0.75,
    'M':1.40,'D':0.89,'F':1.26,'C':0.99,'P':0.36,
    'Q':0.87,'S':0.65,'E':0.37,'T':1.15,'G':0.56,
    'W':0.84,'H':0.36,'Y':1.41,'I':1.75,'V':1.61}

    CRAJ730103={'A':0.60,'L':0.70,'R':0.79,'K':1.10,'N':1.42,
    'M':0.67,'D':1.24,'F':1.05,'C':1.29,'P':1.47,
    'Q':0.92,'S':1.26,'E':0.64,'T':1.05,'G':1.38,
    'W':1.23,'H':0.95,'Y':1.35,'I':0.67,'V':0.48}

    DAWD720101={'A':2.5,'L':5.5,'R':7.5,'K':7.0,'N':5.0,
    'M':6.0,'D':2.5,'F':6.5,'C':3.0,'P':5.5,
    'Q':6.0,'S':3.0,'E':5.0,'T':5.0,'G':0.5,
    'W':7.0,'H':6.0,'Y':7.0,'I':5.5,'V':5.0}

    DAYM780101={'A':8.6,'L':7.4,'R':4.9,'K':6.6,'N':4.3,
    'M':1.7,'D':5.5,'F':3.6,'C':2.9,'P':5.2,
    'Q':3.9,'S':7.0,'E':6.0,'T':6.1,'G':8.4,
    'W':1.3,'H':2.0,'Y':3.4,'I':4.5,'V':6.6}

    DAYM780201={'A':100.0,'L':40.0,'R':65.0,'K':56.0,'N':134.0,
    'M':94.0,'D':106.0,'F':41.0,'C':20.0,'P':56.0,
    'Q':93.0,'S':120.0,'E':102.0,'T':97.0,'G':49.0,
    'W':18.0,'H':66.0,'Y':41.0,'I':96.0,'V':74.0}

    DESM900101={'A':1.56,'L':1.38,'R':0.59,'K':0.15,'N':0.51,
    'M':1.93,'D':0.23,'F':1.42,'C':1.80,'P':0.27,
    'Q':0.39,'S':0.96,'E':0.19,'T':1.11,'G':1.03,
    'W':0.91,'H':1.0,'Y':1.10,'I':1.27,'V':1.58}

    DESM900102={'A':1.26,'L':1.36,'R':0.38,'K':0.33,'N':0.59,
    'M':1.52,'D':0.27,'F':1.46,'C':1.60,'P':0.54,
    'Q':0.39,'S':0.98,'E':0.23,'T':1.01,'G':1.08,
    'W':1.06,'H':1.0,'Y':0.89,'I':1.44,'V':1.33}

    EISD840101={'A':0.25,'L':0.53,'R':-1.76,'K':-1.10,'N':-0.64,
    'M':0.26,'D':-0.72,'F':0.61,'C':0.04,'P':-0.07,
    'Q':-0.69,'S':-0.26,'E':-0.62,'T':-0.18,'G':0.16,
    'W':0.37,'H':-0.40,'Y':0.02,'I':0.73,'V':0.54}

    EISD860101={'A':0.67,'L':1.9,'R':-2.1,'K':-0.57,'N':-0.6,
    'M':2.4,'D':-1.2,'F':2.3,'C':0.38,'P':1.2,
    'Q':-0.22,'S':0.01,'E':-0.76,'T':0.52,'G':0.0,
    'W':2.6,'H':0.64,'Y':1.6,'I':1.9,'V':1.5}

    EISD860102={'A':0.0,'L':1.0,'R':10.0,'K':5.7,'N':1.3,
    'M':1.9,'D':1.9,'F':1.1,'C':0.17,'P':0.18,
    'Q':1.9,'S':0.73,'E':3.0,'T':1.5,'G':0.0,
    'W':1.6,'H':0.99,'Y':1.8,'I':1.2,'V':0.48}

    EISD860103={'A':0.0,'L':0.89,'R':-0.96,'K':-0.99,'N':-0.86,
    'M':0.94,'D':-0.98,'F':0.92,'C':0.76,'P':0.22,
    'Q':-1.0,'S':-0.67,'E':-0.89,'T':0.09,'G':0.0,
    'W':0.67,'H':-0.75,'Y':-0.93,'I':0.99,'V':0.84}

    FASG760101={'A':89.09,'L':131.17,'R':174.20,'K':146.19,'N':132.12,
    'M':149.21,'D':133.10,'F':165.19,'C':121.15,'P':115.13,
    'Q':146.15,'S':105.09,'E':147.13,'T':119.12,'G':75.07,
    'W':204.24,'H':155.16,'Y':181.19,'I':131.17,'V':117.15}

    FASG760102={'A':297.0,'L':337.0,'R':238.0,'K':224.0,'N':236.0,
    'M':283.0,'D':270.0,'F':284.0,'C':178.0,'P':222.0,
    'Q':185.0,'S':228.0,'E':249.0,'T':253.0,'G':290.0,
    'W':282.0,'H':277.0,'Y':344.0,'I':284.0,'V':293.0}

    FASG760103={'A':1.80,'L':-11.00,'R':12.50,'K':14.60,'N':-5.60,
    'M':-10.00,'D':5.05,'F':-34.50,'C':-16.50,'P':-86.20,
    'Q':6.30,'S':-7.50,'E':12.00,'T':-28.00,'G':0.00,
    'W':-33.70,'H':-38.50,'Y':-10.00,'I':12.40,'V':5.63}

    FASG760104={'A':9.69,'L':9.60,'R':8.99,'K':9.18,'N':8.80,
    'M':9.21,'D':9.60,'F':9.18,'C':8.35,'P':10.64,
    'Q':9.13,'S':9.21,'E':9.67,'T':9.10,'G':9.78,
    'W':9.44,'H':9.17,'Y':9.11,'I':9.68,'V':9.62}

    FASG760105={'A':2.34,'L':2.36,'R':1.82,'K':2.16,'N':2.02,
    'M':2.28,'D':1.88,'F':2.16,'C':1.92,'P':1.95,
    'Q':2.17,'S':2.19,'E':2.10,'T':2.09,'G':2.35,
    'W':2.43,'H':1.82,'Y':2.20,'I':2.36,'V':2.32}

    FAUJ830101={'A':0.31,'L':1.70,'R':-1.01,'K':-0.99,'N':-0.60,
    'M':1.23,'D':-0.77,'F':1.79,'C':1.54,'P':0.72,
    'Q':-0.22,'S':-0.04,'E':-0.64,'T':0.26,'G':0.00,
    'W':2.25,'H':0.13,'Y':0.96,'I':1.80,'V':1.22}

    FAUJ880101={'A':1.28,'L':2.59,'R':2.34,'K':1.89,'N':1.60,
    'M':2.35,'D':1.60,'F':2.94,'C':1.77,'P':2.67,
    'Q':1.56,'S':1.31,'E':1.56,'T':3.03,'G':0.00,
    'W':3.21,'H':2.99,'Y':2.94,'I':4.19,'V':3.67}

    FAUJ880102={'A':0.53,'L':0.92,'R':0.69,'K':0.78,'N':0.58,
    'M':0.77,'D':0.59,'F':0.71,'C':0.66,'P':0.0,
    'Q':0.71,'S':0.55,'E':0.72,'T':0.63,'G':0.00,
    'W':0.84,'H':0.64,'Y':0.71,'I':0.96,'V':0.89}

    FAUJ880103={'A':1.00,'L':4.00,'R':6.13,'K':4.77,'N':2.95,
    'M':4.43,'D':2.78,'F':5.89,'C':2.43,'P':2.72,
    'Q':3.95,'S':1.60,'E':3.78,'T':2.60,'G':0.00,
    'W':8.08,'H':4.66,'Y':6.47,'I':4.00,'V':3.00}

    FAUJ880104={'A':2.87,'L':4.92,'R':7.82,'K':6.89,'N':4.58,
    'M':6.36,'D':4.74,'F':4.62,'C':4.47,'P':4.11,
    'Q':6.11,'S':3.97,'E':5.97,'T':4.11,'G':2.06,
    'W':7.68,'H':5.23,'Y':4.73,'I':4.92,'V':4.11}

    FAUJ880105={'A':1.52,'L':1.52,'R':1.52,'K':1.52,'N':1.52,
    'M':1.52,'D':1.52,'F':1.52,'C':1.52,'P':1.52,
    'Q':1.52,'S':1.52,'E':1.52,'T':1.73,'G':1.00,
    'W':1.52,'H':1.52,'Y':1.52,'I':1.90,'V':1.90}

    FAUJ880106={'A':2.04,'L':4.45,'R':6.24,'K':4.87,'N':4.37,
    'M':4.80,'D':3.78,'F':6.02,'C':3.41,'P':4.31,
    'Q':3.53,'S':2.70,'E':3.31,'T':3.17,'G':1.00,
    'W':5.90,'H':5.66,'Y':6.72,'I':3.49,'V':3.17}

    FAUJ880107={'A':7.3,'L':10.1,'R':11.1,'K':10.9,'N':8.0,
    'M':10.4,'D':9.2,'F':13.9,'C':14.4,'P':17.8,
    'Q':10.6,'S':13.1,'E':11.4,'T':16.7,'G':0.0,
    'W':13.2,'H':10.2,'Y':13.9,'I':16.1,'V':17.2}

    FAUJ880108={'A':-0.01,'L':-0.01,'R':0.04,'K':0.00,'N':0.06,
    'M':0.04,'D':0.15,'F':0.03,'C':0.12,'P':0.00,
    'Q':0.05,'S':0.11,'E':0.07,'T':0.04,'G':0.00,
    'W':0.00,'H':0.08,'Y':0.03,'I':-0.01,'V':0.01}

    FAUJ880109={'A':0.0,'L':0.0,'R':4.0,'K':2.0,'N':2.0,
    'M':0.0,'D':1.0,'F':0.0,'C':0.0,'P':0.0,
    'Q':2.0,'S':1.0,'E':1.0,'T':1.0,'G':0.0,
    'W':1.0,'H':1.0,'Y':1.0,'I':0.0,'V':0.0}

    FAUJ880110={'A':0.0,'L':0.0,'R':3.0,'K':1.0,'N':3.0,
    'M':0.0,'D':4.0,'F':0.0,'C':0.0,'P':0.0,
    'Q':3.0,'S':2.0,'E':4.0,'T':2.0,'G':0.0,
    'W':0.0,'H':1.0,'Y':2.0,'I':0.0,'V':0.0}

    FAUJ880111={'A':0.0,'L':0.0,'R':1.0,'K':1.0,'N':0.0,
    'M':0.0,'D':0.0,'F':0.0,'C':0.0,'P':0.0,
    'Q':0.0,'S':0.0,'E':0.0,'T':0.0,'G':0.0,
    'W':0.0,'H':1.0,'Y':0.0,'I':0.0,'V':0.0}

    FAUJ880112={'A':0.,'L':0.,'R':0.,'K':0.,'N':0.,
    'M':0.,'D':1.,'F':0.,'C':0.,'P':0.,
    'Q':0.,'S':0.,'E':1.,'T':0.,'G':0.,
    'W':0.,'H':0.,'Y':0.,'I':0.,'V':0.}

    FAUJ880113={'A':4.76,'L':4.79,'R':4.30,'K':4.27,'N':3.64,
    'M':4.25,'D':5.69,'F':4.31,'C':3.67,'P':0.0,
    'Q':4.54,'S':3.83,'E':5.48,'T':3.87,'G':3.77,
    'W':4.75,'H':2.84,'Y':4.30,'I':4.81,'V':4.86}

    FINA770101={'A':1.08,'L':1.25,'R':1.05,'K':1.15,'N':0.85,
    'M':1.15,'D':0.85,'F':1.10,'C':0.95,'P':0.71,
    'Q':0.95,'S':0.75,'E':1.15,'T':0.75,'G':0.55,
    'W':1.10,'H':1.00,'Y':1.10,'I':1.05,'V':0.95}

    FINA910101={'A':1.0,'L':1.0,'R':0.70,'K':0.70,'N':1.70,
    'M':1.0,'D':3.20,'F':1.0,'C':1.0,'P':1.0,
    'Q':1.0,'S':1.70,'E':1.70,'T':1.70,'G':1.0,
    'W':1.0,'H':1.0,'Y':1.0,'I':0.60,'V':0.60}

    FINA910102={'A':1.0,'L':1.0,'R':0.70,'K':0.70,'N':1.0,
    'M':1.0,'D':1.70,'F':1.0,'C':1.0,'P':13.0,
    'Q':1.0,'S':1.0,'E':1.70,'T':1.0,'G':1.30,
    'W':1.0,'H':1.0,'Y':1.0,'I':1.0,'V':1.0}

    FINA910103={'A':1.20,'L':1.0,'R':1.70,'K':1.70,'N':1.20,
    'M':1.0,'D':0.70,'F':1.0,'C':1.0,'P':1.0,
    'Q':1.0,'S':1.50,'E':0.70,'T':1.0,'G':0.80,
    'W':1.0,'H':1.20,'Y':1.0,'I':0.80,'V':0.80}

    FINA910104={'A':1.0,'L':1.0,'R':1.70,'K':1.70,'N':1.0,
    'M':1.0,'D':0.70,'F':1.0,'C':1.0,'P':0.10,
    'Q':1.0,'S':1.0,'E':0.70,'T':1.0,'G':1.50,
    'W':1.0,'H':1.0,'Y':1.0,'I':1.0,'V':1.0}

    GARJ730101={'A':0.28,'L':1.00,'R':0.10,'K':0.09,'N':0.25,
    'M':0.74,'D':0.21,'F':2.18,'C':0.28,'P':0.39,
    'Q':0.35,'S':0.12,'E':0.33,'T':0.21,'G':0.17,
    'W':5.70,'H':0.21,'Y':1.26,'I':0.82,'V':0.60}

    GEIM800101={'A':1.29,'L':1.31,'R':1.,'K':1.33,'N':0.81,
    'M':1.54,'D':1.10,'F':1.13,'C':0.79,'P':0.63,
    'Q':1.07,'S':0.78,'E':1.49,'T':0.77,'G':0.63,
    'W':1.18,'H':1.33,'Y':0.71,'I':1.05,'V':0.81}

    GEIM800102={'A':1.13,'L':1.13,'R':1.09,'K':1.08,'N':1.06,
    'M':1.23,'D':0.94,'F':1.01,'C':1.32,'P':0.82,
    'Q':0.93,'S':1.01,'E':1.20,'T':1.17,'G':0.83,
    'W':1.32,'H':1.09,'Y':0.88,'I':1.05,'V':1.13}

    GEIM800103={'A':1.55,'L':1.25,'R':0.20,'K':1.20,'N':1.20,
    'M':1.37,'D':1.55,'F':0.40,'C':1.44,'P':0.21,
    'Q':1.13,'S':1.01,'E':1.67,'T':0.55,'G':0.59,
    'W':1.86,'H':1.21,'Y':1.08,'I':1.27,'V':0.64}

    GEIM800104={'A':1.19,'L':1.18,'R':1.0,'K':1.27,'N':0.94,
    'M':1.49,'D':1.07,'F':1.02,'C':0.95,'P':0.68,
    'Q':1.32,'S':0.81,'E':1.64,'T':0.85,'G':0.60,
    'W':1.18,'H':1.03,'Y':0.77,'I':1.12,'V':0.74}

    GEIM800105={'A':0.84,'L':1.10,'R':1.04,'K':0.86,'N':0.66,
    'M':0.88,'D':0.59,'F':1.15,'C':1.27,'P':0.80,
    'Q':1.02,'S':1.05,'E':0.57,'T':1.20,'G':0.94,
    'W':1.15,'H':0.81,'Y':1.39,'I':1.29,'V':1.56}

    GEIM800106={'A':0.86,'L':1.28,'R':1.15,'K':1.01,'N':0.60,
    'M':1.15,'D':0.66,'F':1.34,'C':0.91,'P':0.61,
    'Q':1.11,'S':0.91,'E':0.37,'T':1.14,'G':0.86,
    'W':1.13,'H':1.07,'Y':1.37,'I':1.17,'V':1.31}

    GEIM800107={'A':0.91,'L':1.23,'R':0.99,'K':0.86,'N':0.72,
    'M':0.96,'D':0.74,'F':1.26,'C':1.12,'P':0.65,
    'Q':0.90,'S':0.93,'E':0.41,'T':1.05,'G':0.91,
    'W':1.15,'H':1.01,'Y':1.21,'I':1.29,'V':1.58}

    GEIM800108={'A':0.91,'L':0.59,'R':1.0,'K':0.82,'N':1.64,
    'M':0.58,'D':1.40,'F':0.72,'C':0.93,'P':1.66,
    'Q':0.94,'S':1.23,'E':0.97,'T':1.04,'G':1.51,
    'W':0.67,'H':0.90,'Y':0.92,'I':0.65,'V':0.60}

    GEIM800109={'A':0.80,'L':0.80,'R':0.96,'K':0.94,'N':1.10,
    'M':0.39,'D':1.60,'F':1.20,'C':0.0,'P':2.10,
    'Q':1.60,'S':1.30,'E':0.40,'T':0.60,'G':2.0,
    'W':0.0,'H':0.96,'Y':1.80,'I':0.85,'V':0.80}

    GEIM800110={'A':1.10,'L':0.52,'R':0.93,'K':0.94,'N':1.57,
    'M':0.69,'D':1.41,'F':0.60,'C':1.05,'P':1.77,
    'Q':0.81,'S':1.13,'E':1.40,'T':0.88,'G':1.30,
    'W':0.62,'H':0.85,'Y':0.41,'I':0.67,'V':0.58}

    GEIM800111={'A':0.93,'L':0.59,'R':1.01,'K':0.91,'N':1.36,
    'M':0.60,'D':1.22,'F':0.71,'C':0.92,'P':1.67,
    'Q':0.83,'S':1.25,'E':1.05,'T':1.08,'G':1.45,
    'W':0.68,'H':0.96,'Y':0.98,'I':0.58,'V':0.62}

    GOLD730101={'A':0.75,'L':2.40,'R':0.75,'K':1.50,'N':0.69,
    'M':1.30,'D':0.00,'F':2.65,'C':1.00,'P':2.60,
    'Q':0.59,'S':0.00,'E':0.00,'T':0.45,'G':0.00,
    'W':3.00,'H':0.00,'Y':2.85,'I':2.95,'V':1.70}

    GOLD730102={'A':88.3,'L':168.5,'R':181.2,'K':175.6,'N':125.1,
    'M':162.2,'D':110.8,'F':189.0,'C':112.4,'P':122.2,
    'Q':148.7,'S':88.7,'E':140.5,'T':118.2,'G':60.0,
    'W':227.0,'H':152.6,'Y':193.0,'I':168.5,'V':141.4}

    GRAR740101={'A':0.00,'L':0.00,'R':0.65,'K':0.33,'N':1.33,
    'M':0.00,'D':1.38,'F':0.00,'C':2.75,'P':0.39,
    'Q':0.89,'S':1.42,'E':0.92,'T':0.71,'G':0.74,
    'W':0.13,'H':0.58,'Y':0.20,'I':0.00,'V':0.00}

    GRAR740102={'A':8.1,'L':4.9,'R':10.5,'K':11.3,'N':11.6,
    'M':5.7,'D':13.0,'F':5.2,'C':5.5,'P':8.0,
    'Q':10.5,'S':9.2,'E':12.3,'T':8.6,'G':9.0,
    'W':5.4,'H':10.4,'Y':6.2,'I':5.2,'V':5.9}

    GRAR740103={'A':31.0,'L':111.0,'R':124.0,'K':119.0,'N':56.0,
    'M':105.0,'D':54.0,'F':132.0,'C':55.0,'P':32.5,
    'Q':85.0,'S':32.0,'E':83.0,'T':61.0,'G':3.0,
    'W':170.0,'H':96.0,'Y':136.0,'I':111.0,'V':84.0}

    GUYH850101={'A':0.10,'L':-1.18,'R':1.91,'K':1.40,'N':0.48,
    'M':-1.59,'D':0.78,'F':-2.12,'C':-1.42,'P':0.73,
    'Q':0.95,'S':0.52,'E':0.83,'T':0.07,'G':0.33,
    'W':-0.51,'H':-0.50,'Y':-0.21,'I':-1.13,'V':-1.27}

    HOPA770101={'A':1.0,'L':0.8,'R':2.3,'K':5.3,'N':2.2,
    'M':0.7,'D':6.5,'F':1.4,'C':0.1,'P':0.9,
    'Q':2.1,'S':1.7,'E':6.2,'T':1.5,'G':1.1,
    'W':1.9,'H':2.8,'Y':2.1,'I':0.8,'V':0.9}

    HOPT810101={'A':-0.5,'L':-1.8,'R':3.0,'K':3.0,'N':0.2,
    'M':-1.3,'D':3.0,'F':-2.5,'C':-1.0,'P':0.0,
    'Q':0.2,'S':0.3,'E':3.0,'T':-0.4,'G':0.0,
    'W':-3.4,'H':-0.5,'Y':-2.3,'I':-1.8,'V':-1.5}

    HUTJ700101={'A':29.22,'L':48.03,'R':26.37,'K':57.10,'N':38.30,
    'M':69.32,'D':37.09,'F':48.52,'C':50.70,'P':36.13,
    'Q':44.02,'S':32.40,'E':41.84,'T':35.20,'G':23.71,
    'W':56.92,'H':59.64,'Y':51.73,'I':45.00,'V':40.35}

    HUTJ700102={'A':30.88,'L':50.62,'R':68.43,'K':63.21,'N':41.70,
    'M':55.32,'D':40.66,'F':51.06,'C':53.83,'P':39.21,
    'Q':46.62,'S':35.65,'E':44.98,'T':36.50,'G':24.74,
    'W':60.00,'H':65.99,'Y':51.15,'I':49.71,'V':42.75}

    HUTJ700103={'A':154.33,'L':232.30,'R':341.01,'K':300.46,'N':207.90,
    'M':202.65,'D':194.91,'F':204.74,'C':219.79,'P':179.93,
    'Q':235.51,'S':174.06,'E':223.16,'T':205.80,'G':127.90,
    'W':237.01,'H':242.54,'Y':229.15,'I':233.21,'V':207.60}

    ISOY800101={'A':1.53,'L':1.32,'R':1.17,'K':1.26,'N':0.60,
    'M':1.66,'D':1.00,'F':1.22,'C':0.89,'P':0.25,
    'Q':1.27,'S':0.65,'E':1.63,'T':0.86,'G':0.44,
    'W':1.05,'H':1.03,'Y':0.70,'I':1.07,'V':0.93}

    ISOY800102={'A':0.86,'L':1.01,'R':0.98,'K':0.77,'N':0.74,
    'M':1.06,'D':0.69,'F':1.16,'C':1.39,'P':1.16,
    'Q':0.89,'S':1.09,'E':0.66,'T':1.24,'G':0.70,
    'W':1.17,'H':1.06,'Y':1.28,'I':1.31,'V':1.40}

    ISOY800103={'A':0.78,'L':0.57,'R':1.06,'K':1.01,'N':1.56,
    'M':0.30,'D':1.50,'F':0.67,'C':0.60,'P':1.55,
    'Q':0.78,'S':1.19,'E':0.97,'T':1.09,'G':1.73,
    'W':0.74,'H':0.83,'Y':1.14,'I':0.40,'V':0.44}

    ISOY800104={'A':1.09,'L':0.44,'R':0.97,'K':1.25,'N':1.14,
    'M':0.45,'D':0.77,'F':0.50,'C':0.50,'P':2.96,
    'Q':0.83,'S':1.21,'E':0.92,'T':1.33,'G':1.25,
    'W':0.62,'H':0.67,'Y':0.94,'I':0.66,'V':0.56}

    ISOY800105={'A':0.35,'L':0.58,'R':0.75,'K':0.83,'N':2.12,
    'M':0.22,'D':2.16,'F':0.89,'C':0.50,'P':0.43,
    'Q':0.73,'S':1.24,'E':0.65,'T':0.85,'G':2.40,
    'W':0.62,'H':1.19,'Y':1.44,'I':0.12,'V':0.43}

    ISOY800106={'A':1.09,'L':1.30,'R':1.07,'K':1.20,'N':0.88,
    'M':0.55,'D':1.24,'F':0.80,'C':1.04,'P':1.78,
    'Q':1.09,'S':1.20,'E':1.14,'T':0.99,'G':0.27,
    'W':1.03,'H':1.07,'Y':0.69,'I':0.97,'V':0.77}

    ISOY800107={'A':1.34,'L':1.05,'R':2.78,'K':0.79,'N':0.92,
    'M':0.00,'D':1.77,'F':0.43,'C':1.44,'P':0.37,
    'Q':0.79,'S':0.87,'E':2.54,'T':1.14,'G':0.95,
    'W':1.79,'H':0.00,'Y':0.73,'I':0.52,'V':0.00}

    ISOY800108={'A':0.47,'L':0.53,'R':0.52,'K':0.98,'N':2.16,
    'M':0.68,'D':1.15,'F':0.61,'C':0.41,'P':0.63,
    'Q':0.95,'S':1.03,'E':0.64,'T':0.39,'G':3.03,
    'W':0.63,'H':0.89,'Y':0.83,'I':0.62,'V':0.76}

    JANJ780101={'A':27.8,'L':27.6,'R':94.7,'K':103.0,'N':60.1,
    'M':33.5,'D':60.6,'F':25.5,'C':15.5,'P':51.5,
    'Q':68.7,'S':42.0,'E':68.2,'T':45.0,'G':24.5,
    'W':34.7,'H':50.7,'Y':55.2,'I':22.8,'V':23.7}

    JANJ780102={'A':51.0,'L':60.0,'R':5.0,'K':3.0,'N':22.0,
    'M':52.0,'D':19.0,'F':58.0,'C':74.0,'P':25.0,
    'Q':16.0,'S':35.0,'E':16.0,'T':30.0,'G':52.0,
    'W':49.0,'H':34.0,'Y':24.0,'I':66.0,'V':64.0}

    JANJ780103={'A':15.0,'L':16.0,'R':67.0,'K':85.0,'N':49.0,
    'M':20.0,'D':50.0,'F':10.0,'C':5.0,'P':45.0,
    'Q':56.0,'S':32.0,'E':55.0,'T':32.0,'G':10.0,
    'W':17.0,'H':34.0,'Y':41.0,'I':13.0,'V':14.0}

    JANJ790101={'A':1.7,'L':2.4,'R':0.1,'K':0.05,'N':0.4,
    'M':1.9,'D':0.4,'F':2.2,'C':4.6,'P':0.6,
    'Q':0.3,'S':0.8,'E':0.3,'T':0.7,'G':1.8,
    'W':1.6,'H':0.8,'Y':0.5,'I':3.1,'V':2.9}

    JANJ790102={'A':0.3,'L':0.5,'R':-1.4,'K':-1.8,'N':-0.5,
    'M':0.4,'D':-0.6,'F':0.5,'C':0.9,'P':-0.3,
    'Q':-0.7,'S':-0.1,'E':-0.7,'T':-0.2,'G':0.3,
    'W':0.3,'H':-0.1,'Y':-0.4,'I':0.7,'V':0.6}

    JOND750101={'A':0.87,'L':2.17,'R':0.85,'K':1.64,'N':0.09,
    'M':1.67,'D':0.66,'F':2.87,'C':1.52,'P':2.77,
    'Q':0.00,'S':0.07,'E':0.67,'T':0.07,'G':0.10,
    'W':3.77,'H':0.87,'Y':2.67,'I':3.15,'V':1.87}

    JOND750102={'A':2.34,'L':2.36,'R':1.18,'K':2.18,'N':2.02,
    'M':2.28,'D':2.01,'F':1.83,'C':1.65,'P':1.99,
    'Q':2.17,'S':2.21,'E':2.19,'T':2.10,'G':2.34,
    'W':2.38,'H':1.82,'Y':2.20,'I':2.36,'V':2.32}

    JOND920101={'A':0.077,'L':0.091,'R':0.051,'K':0.059,'N':0.043,
    'M':0.024,'D':0.052,'F':0.040,'C':0.020,'P':0.051,
    'Q':0.041,'S':0.069,'E':0.062,'T':0.059,'G':0.074,
    'W':0.014,'H':0.023,'Y':0.032,'I':0.053,'V':0.066}

    JOND920102={'A':100.0,'L':54.0,'R':83.0,'K':72.0,'N':104.0,
    'M':93.0,'D':86.0,'F':51.0,'C':44.0,'P':58.0,
    'Q':84.0,'S':117.0,'E':77.0,'T':107.0,'G':50.0,
    'W':25.0,'H':91.0,'Y':50.0,'I':103.0,'V':98.0}

    JUKT750101={'A':5.3,'L':4.7,'R':2.6,'K':4.1,'N':3.0,
    'M':1.1,'D':3.6,'F':2.3,'C':1.3,'P':2.5,
    'Q':2.4,'S':4.5,'E':3.3,'T':3.7,'G':4.8,
    'W':0.8,'H':1.4,'Y':2.3,'I':3.1,'V':4.2}

    JUNJ780101={'A':685.0,'L':581.0,'R':382.0,'K':575.0,'N':397.0,
    'M':132.0,'D':400.0,'F':303.0,'C':241.0,'P':366.0,
    'Q':313.0,'S':593.0,'E':427.0,'T':490.0,'G':707.0,
    'W':99.0,'H':155.0,'Y':292.0,'I':394.0,'V':553.0}

    KANM800101={'A':1.36,'L':1.21,'R':1.00,'K':1.22,'N':0.89,
    'M':1.45,'D':1.04,'F':1.05,'C':0.82,'P':0.52,
    'Q':1.14,'S':0.74,'E':1.48,'T':0.81,'G':0.63,
    'W':0.97,'H':1.11,'Y':0.79,'I':1.08,'V':0.94}

    KANM800102={'A':0.81,'L':1.24,'R':0.85,'K':0.77,'N':0.62,
    'M':1.05,'D':0.71,'F':1.20,'C':1.17,'P':0.61,
    'Q':0.98,'S':0.92,'E':0.53,'T':1.18,'G':0.88,
    'W':1.18,'H':0.92,'Y':1.23,'I':1.48,'V':1.66}

    KANM800103={'A':1.45,'L':1.56,'R':1.15,'K':1.27,'N':0.64,
    'M':1.83,'D':0.91,'F':1.20,'C':0.70,'P':0.21,
    'Q':1.14,'S':0.48,'E':1.29,'T':0.77,'G':0.53,
    'W':1.17,'H':1.13,'Y':0.74,'I':1.23,'V':1.10}

    KANM800104={'A':0.75,'L':1.56,'R':0.79,'K':0.66,'N':0.33,
    'M':0.86,'D':0.31,'F':1.37,'C':1.46,'P':0.52,
    'Q':0.75,'S':0.82,'E':0.46,'T':1.36,'G':0.83,
    'W':0.79,'H':0.83,'Y':1.08,'I':1.87,'V':2.00}

    KARP850101={'A':1.041,'L':0.967,'R':1.038,'K':1.093,'N':1.117,
    'M':0.947,'D':1.033,'F':0.930,'C':0.960,'P':1.055,
    'Q':1.165,'S':1.169,'E':1.094,'T':1.073,'G':1.142,
    'W':0.925,'H':0.982,'Y':0.961,'I':1.002,'V':0.982}

    KARP850102={'A':0.946,'L':0.961,'R':1.028,'K':1.082,'N':1.006,
    'M':0.862,'D':1.089,'F':0.912,'C':0.878,'P':1.085,
    'Q':1.025,'S':1.048,'E':1.036,'T':1.051,'G':1.042,
    'W':0.917,'H':0.952,'Y':0.930,'I':0.892,'V':0.927}

    KARP850103={'A':0.892,'L':0.921,'R':0.901,'K':1.057,'N':0.930,
    'M':0.804,'D':0.932,'F':0.914,'C':0.925,'P':0.932,
    'Q':0.885,'S':0.923,'E':0.933,'T':0.934,'G':0.923,
    'W':0.803,'H':0.894,'Y':0.837,'I':0.872,'V':0.913}

    KHAG800101={'A':49.1,'L':15.6,'R':133.0,'K':0.0,'N':-3.6,
    'M':6.8,'D':0.0,'F':54.7,'C':0.0,'P':43.8,
    'Q':20.0,'S':44.4,'E':0.0,'T':31.0,'G':64.6,
    'W':70.5,'H':75.7,'Y':0.0,'I':18.9,'V':29.5}

    KLEP840101={'A':0.0,'L':0.0,'R':1.0,'K':1.0,'N':0.0,
    'M':0.0,'D':-1.0,'F':0.0,'C':0.0,'P':0.0,
    'Q':0.0,'S':0.0,'E':-1.0,'T':0.0,'G':0.0,
    'W':0.0,'H':0.0,'Y':0.0,'I':0.0,'V':0.0}

    KRIW710101={'A':4.60,'L':3.25,'R':6.50,'K':7.90,'N':5.90,
    'M':1.40,'D':5.70,'F':3.20,'C':-1.00,'P':7.00,
    'Q':6.10,'S':5.25,'E':5.60,'T':4.80,'G':7.60,
    'W':4.00,'H':4.50,'Y':4.35,'I':2.60,'V':3.40}

    KRIW790102={'A':0.28,'L':0.16,'R':0.34,'K':0.59,'N':0.31,
    'M':0.08,'D':0.33,'F':0.10,'C':0.11,'P':0.46,
    'Q':0.39,'S':0.27,'E':0.37,'T':0.26,'G':0.28,
    'W':0.15,'H':0.23,'Y':0.25,'I':0.12,'V':0.22}

    KRIW790103={'A':27.5,'L':93.5,'R':105.0,'K':100.0,'N':58.7,
    'M':94.1,'D':40.0,'F':115.5,'C':44.6,'P':41.9,
    'Q':80.7,'S':29.3,'E':62.0,'T':51.3,'G':0.0,
    'W':145.5,'H':79.0,'Y':117.3,'I':93.5,'V':71.5}

    KYTJ820101={'A':1.8,'L':3.8,'R':-4.5,'K':-3.9,'N':-3.5,
    'M':1.9,'D':-3.5,'F':2.8,'C':2.5,'P':-1.6,
    'Q':-3.5,'S':-0.8,'E':-3.5,'T':-0.7,'G':-0.4,
    'W':-0.9,'H':-3.2,'Y':-1.3,'I':4.5,'V':4.2}

    LAWE840101={'A':-0.48,'L':1.02,'R':-0.06,'K':-0.09,'N':-0.87,
    'M':0.81,'D':-0.75,'F':1.03,'C':-0.32,'P':2.03,
    'Q':-0.32,'S':0.05,'E':-0.71,'T':-0.35,'G':0.00,
    'W':0.66,'H':-0.51,'Y':1.24,'I':0.81,'V':0.56}

    LEVM760101={'A':-0.5,'L':-1.8,'R':3.0,'K':3.0,'N':0.2,
    'M':-1.3,'D':2.5,'F':-2.5,'C':-1.0,'P':-1.4,
    'Q':0.2,'S':0.3,'E':2.5,'T':-0.4,'G':0.0,
    'W':-3.4,'H':-0.5,'Y':-2.3,'I':-1.8,'V':-1.5}

    LEVM760102={'A':0.77,'L':2.08,'R':3.72,'K':2.94,'N':1.98,
    'M':2.34,'D':1.99,'F':2.97,'C':1.38,'P':1.42,
    'Q':2.58,'S':1.28,'E':2.63,'T':1.43,'G':0.00,
    'W':3.58,'H':2.76,'Y':3.36,'I':1.83,'V':1.49}

    LEVM760103={'A':121.9,'L':118.1,'R':121.4,'K':122.0,'N':117.5,
    'M':113.1,'D':121.2,'F':118.2,'C':113.7,'P':81.9,
    'Q':118.0,'S':117.9,'E':118.2,'T':117.1,'G':0.0,
    'W':118.4,'H':118.2,'Y':110.0,'I':118.9,'V':121.7}

    LEVM760104={'A':243.2,'L':205.6,'R':206.6,'K':210.9,'N':207.1,
    'M':204.0,'D':215.0,'F':203.7,'C':209.4,'P':237.4,
    'Q':205.4,'S':232.0,'E':213.6,'T':226.7,'G':300.0,
    'W':203.7,'H':219.9,'Y':195.6,'I':217.9,'V':220.3}

    LEVM760105={'A':0.77,'L':1.54,'R':2.38,'K':2.08,'N':1.45,
    'M':1.80,'D':1.43,'F':1.90,'C':1.22,'P':1.25,
    'Q':1.75,'S':1.08,'E':1.77,'T':1.24,'G':0.58,
    'W':2.21,'H':1.78,'Y':2.13,'I':1.56,'V':1.29}

    LEVM760106={'A':5.2,'L':7.0,'R':6.0,'K':6.0,'N':5.0,
    'M':6.8,'D':5.0,'F':7.1,'C':6.1,'P':6.2,
    'Q':6.0,'S':4.9,'E':6.0,'T':5.0,'G':4.2,
    'W':7.6,'H':6.0,'Y':7.1,'I':7.0,'V':6.4}

    LEVM760107={'A':0.025,'L':0.19,'R':0.20,'K':0.20,'N':0.10,
    'M':0.19,'D':0.10,'F':0.39,'C':0.10,'P':0.17,
    'Q':0.10,'S':0.025,'E':0.10,'T':0.10,'G':0.025,
    'W':0.56,'H':0.10,'Y':0.39,'I':0.19,'V':0.15}

    LEVM780101={'A':1.29,'L':1.30,'R':0.96,'K':1.23,'N':0.90,
    'M':1.47,'D':1.04,'F':1.07,'C':1.11,'P':0.52,
    'Q':1.27,'S':0.82,'E':1.44,'T':0.82,'G':0.56,
    'W':0.99,'H':1.22,'Y':0.72,'I':0.97,'V':0.91}

    LEVM780102={'A':0.90,'L':1.02,'R':0.99,'K':0.77,'N':0.76,
    'M':0.97,'D':0.72,'F':1.32,'C':0.74,'P':0.64,
    'Q':0.80,'S':0.95,'E':0.75,'T':1.21,'G':0.92,
    'W':1.14,'H':1.08,'Y':1.25,'I':1.45,'V':1.49}

    LEVM780103={'A':0.77,'L':0.58,'R':0.88,'K':0.96,'N':1.28,
    'M':0.41,'D':1.41,'F':0.59,'C':0.81,'P':1.91,
    'Q':0.98,'S':1.32,'E':0.99,'T':1.04,'G':1.64,
    'W':0.76,'H':0.68,'Y':1.05,'I':0.51,'V':0.47}

    LEVM780104={'A':1.32,'L':1.31,'R':0.98,'K':1.25,'N':0.95,
    'M':1.39,'D':1.03,'F':1.02,'C':0.92,'P':0.58,
    'Q':1.10,'S':0.76,'E':1.44,'T':0.79,'G':0.61,
    'W':0.97,'H':1.31,'Y':0.73,'I':0.93,'V':0.93}

    LEVM780105={'A':0.86,'L':1.04,'R':0.97,'K':0.77,'N':0.73,
    'M':0.93,'D':0.69,'F':1.21,'C':1.04,'P':0.68,
    'Q':1.00,'S':1.02,'E':0.66,'T':1.27,'G':0.89,
    'W':1.26,'H':0.85,'Y':1.31,'I':1.47,'V':1.43}

    LEVM780106={'A':0.79,'L':0.57,'R':0.90,'K':0.99,'N':1.25,
    'M':0.51,'D':1.47,'F':0.77,'C':0.79,'P':1.78,
    'Q':0.92,'S':1.30,'E':1.02,'T':0.97,'G':1.67,
    'W':0.79,'H':0.81,'Y':0.93,'I':0.50,'V':0.46}

    LEWP710101={'A':0.22,'L':0.19,'R':0.28,'K':0.27,'N':0.42,
    'M':0.38,'D':0.73,'F':0.08,'C':0.20,'P':0.46,
    'Q':0.26,'S':0.55,'E':0.08,'T':0.49,'G':0.58,
    'W':0.43,'H':0.14,'Y':0.46,'I':0.22,'V':0.08}

    LIFS790101={'A':0.92,'L':1.30,'R':0.93,'K':0.70,'N':0.60,
    'M':1.19,'D':0.48,'F':1.25,'C':1.16,'P':0.40,
    'Q':0.95,'S':0.82,'E':0.61,'T':1.12,'G':0.61,
    'W':1.54,'H':0.93,'Y':1.53,'I':1.81,'V':1.81}

    LIFS790102={'A':1.00,'L':1.42,'R':0.68,'K':0.59,'N':0.54,
    'M':1.49,'D':0.50,'F':1.30,'C':0.91,'P':0.35,
    'Q':0.28,'S':0.70,'E':0.59,'T':0.59,'G':0.79,
    'W':0.89,'H':0.38,'Y':1.08,'I':2.60,'V':2.63}

    LIFS790103={'A':0.90,'L':1.26,'R':1.02,'K':0.74,'N':0.62,
    'M':1.09,'D':0.47,'F':1.23,'C':1.24,'P':0.42,
    'Q':1.18,'S':0.87,'E':0.62,'T':1.30,'G':0.56,
    'W':1.75,'H':1.12,'Y':1.68,'I':1.54,'V':1.53}

    MANP780101={'A':12.97,'L':14.90,'R':11.72,'K':11.36,'N':11.42,
    'M':14.39,'D':10.85,'F':14.00,'C':14.63,'P':11.37,
    'Q':11.76,'S':11.23,'E':11.89,'T':11.69,'G':12.43,
    'W':13.93,'H':12.16,'Y':13.42,'I':15.67,'V':15.71}

    MAXF760101={'A':1.43,'L':1.36,'R':1.18,'K':1.27,'N':0.64,
    'M':1.53,'D':0.92,'F':1.19,'C':0.94,'P':0.49,
    'Q':1.22,'S':0.70,'E':1.67,'T':0.78,'G':0.46,
    'W':1.01,'H':0.98,'Y':0.69,'I':1.04,'V':0.98}

    MAXF760102={'A':0.86,'L':0.98,'R':0.94,'K':0.79,'N':0.74,
    'M':1.08,'D':0.72,'F':1.16,'C':1.17,'P':1.22,
    'Q':0.89,'S':1.04,'E':0.62,'T':1.18,'G':0.97,
    'W':1.07,'H':1.06,'Y':1.25,'I':1.24,'V':1.33}

    MAXF760103={'A':0.64,'L':0.37,'R':0.62,'K':0.89,'N':3.14,
    'M':1.07,'D':1.92,'F':0.86,'C':0.32,'P':0.50,
    'Q':0.80,'S':1.01,'E':1.01,'T':0.92,'G':0.63,
    'W':1.00,'H':2.05,'Y':1.31,'I':0.92,'V':0.87}

    MAXF760104={'A':0.17,'L':0.21,'R':0.76,'K':1.17,'N':2.62,
    'M':0.00,'D':1.08,'F':0.28,'C':0.95,'P':0.12,
    'Q':0.91,'S':0.57,'E':0.28,'T':0.23,'G':5.02,
    'W':0.00,'H':0.57,'Y':0.97,'I':0.26,'V':0.24}

    MAXF760105={'A':1.13,'L':0.65,'R':0.48,'K':1.13,'N':1.11,
    'M':0.00,'D':1.18,'F':0.45,'C':0.38,'P':0.00,
    'Q':0.41,'S':0.81,'E':1.02,'T':0.71,'G':3.84,
    'W':0.93,'H':0.30,'Y':0.38,'I':0.40,'V':0.48}

    MAXF760106={'A':1.00,'L':1.01,'R':1.18,'K':1.05,'N':0.87,
    'M':0.36,'D':1.39,'F':0.65,'C':1.09,'P':1.95,
    'Q':1.13,'S':1.56,'E':1.04,'T':1.23,'G':0.46,
    'W':1.10,'H':0.71,'Y':0.87,'I':0.68,'V':0.58}

    MCMT640101={'A':4.34,'L':18.78,'R':26.66,'K':21.29,'N':13.28,
    'M':21.64,'D':12.00,'F':29.40,'C':35.77,'P':10.93,
    'Q':17.56,'S':6.35,'E':17.26,'T':11.01,'G':0.00,
    'W':42.53,'H':21.81,'Y':31.53,'I':19.06,'V':13.92}

    MEEJ800101={'A':0.5,'L':8.8,'R':0.8,'K':0.1,'N':0.8,
    'M':4.8,'D':-8.2,'F':13.2,'C':-6.8,'P':6.1,
    'Q':-4.8,'S':1.2,'E':-16.9,'T':2.7,'G':0.0,
    'W':14.9,'H':-3.5,'Y':6.1,'I':13.9,'V':2.7}

    MEEJ800102={'A':-0.1,'L':10.0,'R':-4.5,'K':-3.2,'N':-1.6,
    'M':7.1,'D':-2.8,'F':13.9,'C':-2.2,'P':8.0,
    'Q':-2.5,'S':-3.7,'E':-7.5,'T':1.5,'G':-0.5,
    'W':18.1,'H':0.8,'Y':8.2,'I':11.8,'V':3.3}

    MEEJ810101={'A':1.1,'L':11.0,'R':-0.4,'K':-1.9,'N':-4.2,
    'M':5.4,'D':-1.6,'F':13.4,'C':7.1,'P':4.4,
    'Q':-2.9,'S':-3.2,'E':0.7,'T':-1.7,'G':-0.2,
    'W':17.1,'H':-0.7,'Y':7.4,'I':8.5,'V':5.9}

    MEEJ810102={'A':1.0,'L':9.6,'R':-2.0,'K':-3.0,'N':-3.0,
    'M':4.0,'D':-0.5,'F':12.6,'C':4.6,'P':3.1,
    'Q':-2.0,'S':-2.9,'E':1.1,'T':-0.6,'G':0.2,
    'W':15.1,'H':-2.2,'Y':6.7,'I':7.0,'V':4.6}

    MEIH800101={'A':0.93,'L':0.85,'R':0.98,'K':1.05,'N':0.98,
    'M':0.84,'D':1.01,'F':0.78,'C':0.88,'P':1.00,
    'Q':1.02,'S':1.02,'E':1.02,'T':0.99,'G':1.01,
    'W':0.83,'H':0.89,'Y':0.93,'I':0.79,'V':0.81}

    MEIH800102={'A':0.94,'L':0.82,'R':1.09,'K':1.23,'N':1.04,
    'M':0.83,'D':1.08,'F':0.73,'C':0.84,'P':1.04,
    'Q':1.11,'S':1.04,'E':1.12,'T':1.02,'G':1.01,
    'W':0.87,'H':0.92,'Y':1.03,'I':0.76,'V':0.81}

    MEIH800103={'A':87.0,'L':104.0,'R':81.0,'K':65.0,'N':70.0,
    'M':100.0,'D':71.0,'F':108.0,'C':104.0,'P':78.0,
    'Q':66.0,'S':83.0,'E':72.0,'T':83.0,'G':90.0,
    'W':94.0,'H':90.0,'Y':83.0,'I':105.0,'V':94.0}

    MIYS850101={'A':2.36,'L':3.93,'R':1.92,'K':1.23,'N':1.70,
    'M':4.22,'D':1.67,'F':4.37,'C':3.36,'P':1.89,
    'Q':1.75,'S':1.81,'E':1.74,'T':2.04,'G':2.06,
    'W':3.82,'H':2.41,'Y':2.91,'I':4.17,'V':3.49}

    NAGK730101={'A':1.29,'L':1.23,'R':0.83,'K':1.23,'N':0.77,
    'M':1.23,'D':1.00,'F':1.23,'C':0.94,'P':0.70,
    'Q':1.10,'S':0.78,'E':1.54,'T':0.87,'G':0.72,
    'W':1.06,'H':1.29,'Y':0.63,'I':0.94,'V':0.97}

    NAGK730102={'A':0.96,'L':1.26,'R':0.67,'K':0.81,'N':0.72,
    'M':1.29,'D':0.90,'F':1.37,'C':1.13,'P':0.75,
    'Q':1.18,'S':0.77,'E':0.33,'T':1.23,'G':0.90,
    'W':1.13,'H':0.87,'Y':1.07,'I':1.54,'V':1.41}

    NAGK730103={'A':0.72,'L':0.63,'R':1.33,'K':0.84,'N':1.38,
    'M':0.62,'D':1.04,'F':0.58,'C':1.01,'P':1.43,
    'Q':0.81,'S':1.34,'E':0.75,'T':1.03,'G':1.35,
    'W':0.87,'H':0.76,'Y':1.35,'I':0.80,'V':0.83}

    NAKH900101={'A':7.99,'L':9.16,'R':5.86,'K':6.01,'N':4.33,
    'M':2.50,'D':5.14,'F':3.83,'C':1.81,'P':4.95,
    'Q':3.98,'S':6.84,'E':6.10,'T':5.77,'G':6.91,
    'W':1.34,'H':2.17,'Y':3.15,'I':5.48,'V':6.65}

    NAKH900102={'A':3.73,'L':3.40,'R':3.34,'K':3.36,'N':2.33,
    'M':1.37,'D':2.23,'F':1.94,'C':2.30,'P':3.18,
    'Q':2.36,'S':2.83,'E':3.0,'T':2.63,'G':3.36,
    'W':1.15,'H':1.55,'Y':1.76,'I':2.52,'V':2.53}

    NAKH900103={'A':5.74,'L':15.36,'R':1.92,'K':3.20,'N':5.25,
    'M':5.30,'D':2.11,'F':6.51,'C':1.03,'P':4.79,
    'Q':2.30,'S':7.55,'E':2.63,'T':7.51,'G':5.66,
    'W':2.51,'H':2.30,'Y':4.08,'I':9.12,'V':5.12}

    NAKH900104={'A':-0.60,'L':1.82,'R':-1.18,'K':-0.84,'N':0.39,
    'M':2.04,'D':-1.36,'F':1.38,'C':-0.34,'P':-0.05,
    'Q':-0.71,'S':0.25,'E':-1.16,'T':0.66,'G':-0.37,
    'W':1.02,'H':0.08,'Y':0.53,'I':1.44,'V':-0.60}

    NAKH900105={'A':5.88,'L':16.52,'R':1.54,'K':2.58,'N':4.38,
    'M':6.00,'D':1.70,'F':6.58,'C':1.11,'P':5.29,
    'Q':2.30,'S':7.68,'E':2.60,'T':8.38,'G':5.29,
    'W':2.89,'H':2.33,'Y':3.51,'I':8.78,'V':4.66}

    NAKH900106={'A':-0.57,'L':2.16,'R':-1.29,'K':-1.02,'N':0.02,
    'M':2.55,'D':-1.54,'F':1.42,'C':-0.30,'P':0.11,
    'Q':-0.71,'S':0.30,'E':-1.17,'T':0.99,'G':-0.48,
    'W':1.35,'H':0.10,'Y':0.20,'I':1.31,'V':-0.79}

    NAKH900107={'A':5.39,'L':12.64,'R':2.81,'K':4.67,'N':7.31,
    'M':3.68,'D':3.07,'F':6.34,'C':0.86,'P':3.62,
    'Q':2.31,'S':7.24,'E':2.70,'T':5.44,'G':6.52,
    'W':1.64,'H':2.23,'Y':5.42,'I':9.94,'V':6.18}

    NAKH900108={'A':-0.70,'L':1.02,'R':-0.91,'K':-0.40,'N':1.28,
    'M':0.86,'D':-0.93,'F':1.29,'C':-0.41,'P':-0.42,
    'Q':-0.71,'S':0.14,'E':-1.13,'T':-0.13,'G':-0.12,
    'W':0.26,'H':0.04,'Y':1.29,'I':1.77,'V':-0.19}

    NAKH900109={'A':9.25,'L':10.94,'R':3.96,'K':3.50,'N':3.71,
    'M':3.14,'D':3.89,'F':6.36,'C':1.07,'P':4.36,
    'Q':3.17,'S':6.26,'E':4.80,'T':5.66,'G':8.51,
    'W':2.22,'H':1.88,'Y':3.28,'I':6.47,'V':7.55}

    NAKH900110={'A':0.34,'L':0.52,'R':-0.57,'K':-0.75,'N':-0.27,
    'M':0.47,'D':-0.56,'F':1.30,'C':-0.32,'P':-0.19,
    'Q':-0.34,'S':-0.20,'E':-0.43,'T':-0.04,'G':0.48,
    'W':0.77,'H':-0.19,'Y':0.07,'I':0.39,'V':0.36}

    NAKH900111={'A':10.17,'L':16.22,'R':1.21,'K':1.04,'N':1.36,
    'M':4.12,'D':1.18,'F':9.60,'C':1.48,'P':2.24,
    'Q':1.57,'S':5.38,'E':1.15,'T':5.61,'G':8.87,
    'W':2.67,'H':1.07,'Y':2.68,'I':10.91,'V':11.44}

    NAKH900112={'A':6.61,'L':21.66,'R':0.41,'K':1.15,'N':1.84,
    'M':7.17,'D':0.59,'F':7.76,'C':0.83,'P':3.51,
    'Q':1.20,'S':6.84,'E':1.63,'T':8.89,'G':4.88,
    'W':2.11,'H':1.14,'Y':2.57,'I':12.91,'V':6.30}

    NAKH900113={'A':1.61,'L':1.37,'R':0.40,'K':0.62,'N':0.73,
    'M':1.59,'D':0.75,'F':1.24,'C':0.37,'P':0.67,
    'Q':0.61,'S':0.68,'E':1.50,'T':0.92,'G':3.12,
    'W':1.63,'H':0.46,'Y':0.67,'I':1.61,'V':1.30}

    NAKH920101={'A':8.63,'L':8.44,'R':6.75,'K':6.25,'N':4.18,
    'M':2.14,'D':6.24,'F':2.73,'C':1.03,'P':6.28,
    'Q':4.76,'S':8.53,'E':7.82,'T':4.43,'G':6.80,
    'W':0.80,'H':2.70,'Y':2.54,'I':3.48,'V':5.44}

    NAKH920102={'A':10.88,'L':8.03,'R':6.01,'K':6.11,'N':5.75,
    'M':3.79,'D':6.13,'F':2.93,'C':0.69,'P':7.21,
    'Q':4.68,'S':7.25,'E':9.34,'T':3.51,'G':7.72,
    'W':0.47,'H':2.15,'Y':1.01,'I':1.80,'V':4.57}

    NAKH920103={'A':5.15,'L':8.11,'R':4.38,'K':5.25,'N':4.81,
    'M':1.60,'D':5.75,'F':3.52,'C':3.24,'P':5.65,
    'Q':4.45,'S':8.04,'E':7.05,'T':7.41,'G':6.38,
    'W':1.68,'H':2.69,'Y':3.42,'I':4.40,'V':7.00}

    NAKH920104={'A':5.04,'L':9.88,'R':3.73,'K':6.31,'N':5.94,
    'M':1.85,'D':5.26,'F':3.72,'C':2.20,'P':6.22,
    'Q':4.50,'S':8.05,'E':6.07,'T':5.20,'G':7.09,
    'W':2.10,'H':2.99,'Y':3.32,'I':4.32,'V':6.19}

    NAKH920105={'A':9.90,'L':22.28,'R':0.09,'K':0.16,'N':0.94,
    'M':1.85,'D':0.35,'F':6.47,'C':2.55,'P':2.38,
    'Q':0.87,'S':4.17,'E':0.08,'T':4.33,'G':8.14,
    'W':2.21,'H':0.20,'Y':3.42,'I':15.25,'V':14.34}

    NAKH920106={'A':6.69,'L':8.23,'R':6.65,'K':8.36,'N':4.49,
    'M':2.46,'D':4.97,'F':3.59,'C':1.70,'P':5.20,
    'Q':5.39,'S':7.40,'E':7.76,'T':5.18,'G':6.32,
    'W':1.06,'H':2.11,'Y':2.75,'I':4.51,'V':5.27}

    NAKH920107={'A':5.08,'L':8.03,'R':4.75,'K':4.93,'N':5.75,
    'M':2.61,'D':5.96,'F':4.36,'C':2.95,'P':4.84,
    'Q':4.24,'S':6.41,'E':6.04,'T':5.87,'G':8.20,
    'W':2.31,'H':2.10,'Y':4.55,'I':4.95,'V':6.07}

    NAKH920108={'A':9.36,'L':16.64,'R':0.27,'K':0.58,'N':2.31,
    'M':3.93,'D':0.94,'F':10.99,'C':2.56,'P':1.96,
    'Q':1.14,'S':5.58,'E':0.94,'T':4.68,'G':6.17,
    'W':2.20,'H':0.47,'Y':3.13,'I':13.73,'V':12.43}

    NISK800101={'A':0.23,'L':1.03,'R':-0.26,'K':-1.05,'N':-0.94,
    'M':0.66,'D':-1.13,'F':0.48,'C':1.78,'P':-0.76,
    'Q':-0.57,'S':-0.67,'E':-0.75,'T':-0.36,'G':-0.07,
    'W':0.90,'H':0.11,'Y':0.59,'I':1.19,'V':1.24}

    NISK860101={'A':-0.22,'L':5.01,'R':-0.93,'K':-4.18,'N':-2.65,
    'M':3.51,'D':-4.12,'F':5.27,'C':4.66,'P':-3.03,
    'Q':-2.76,'S':-2.84,'E':-3.64,'T':-1.20,'G':-1.62,
    'W':5.20,'H':1.28,'Y':2.15,'I':5.58,'V':4.45}

    NOZY710101={'A':0.5,'L':1.8,'R':0.0,'K':0.0,'N':0.0,
    'M':1.3,'D':0.0,'F':2.5,'C':0.0,'P':0.0,
    'Q':0.0,'S':0.0,'E':0.0,'T':0.4,'G':0.0,
    'W':3.4,'H':0.5,'Y':2.3,'I':1.8,'V':1.5}

    OOBM770101={'A':-1.895,'L':-1.966,'R':-1.475,'K':-1.374,'N':-1.560,
    'M':-1.963,'D':-1.518,'F':-1.864,'C':-2.035,'P':-1.699,
    'Q':-1.521,'S':-1.753,'E':-1.535,'T':-1.767,'G':-1.898,
    'W':-1.869,'H':-1.755,'Y':-1.686,'I':-1.951,'V':-1.981}

    OOBM770102={'A':-1.404,'L':-1.315,'R':-0.921,'K':-1.074,'N':-1.178,
    'M':-1.303,'D':-1.162,'F':-1.135,'C':-1.365,'P':-1.236,
    'Q':-1.116,'S':-1.297,'E':-1.163,'T':-1.252,'G':-1.364,
    'W':-1.030,'H':-1.215,'Y':-1.030,'I':-1.189,'V':-1.254}

    OOBM770103={'A':-0.491,'L':-0.650,'R':-0.554,'K':-0.300,'N':-0.382,
    'M':-0.659,'D':-0.356,'F':-0.729,'C':-0.670,'P':-0.463,
    'Q':-0.405,'S':-0.455,'E':-0.371,'T':-0.515,'G':-0.534,
    'W':-0.839,'H':-0.540,'Y':-0.656,'I':-0.762,'V':-0.728}

    OOBM770104={'A':-9.475,'L':-15.728,'R':-16.225,'K':-12.366,'N':-12.480,
    'M':-15.704,'D':-12.144,'F':-20.504,'C':-12.210,'P':-11.893,
    'Q':-13.689,'S':-10.518,'E':-13.815,'T':-12.369,'G':-7.592,
    'W':-26.166,'H':-17.550,'Y':-20.232,'I':-15.608,'V':-13.867}

    OOBM770105={'A':-7.020,'L':-10.520,'R':-10.131,'K':-9.666,'N':-9.424,
    'M':-10.424,'D':-9.296,'F':-12.485,'C':-8.190,'P':-8.652,
    'Q':-10.044,'S':-7.782,'E':-10.467,'T':-8.764,'G':-5.456,
    'W':-14.420,'H':-12.150,'Y':-12.360,'I':-9.512,'V':-8.778}

    OOBM850101={'A':2.01,'L':2.73,'R':0.84,'K':2.55,'N':0.03,
    'M':1.75,'D':-2.05,'F':2.68,'C':1.98,'P':0.41,
    'Q':1.02,'S':1.47,'E':0.93,'T':2.39,'G':0.12,
    'W':2.49,'H':-0.14,'Y':2.23,'I':3.70,'V':3.50}

    OOBM850102={'A':1.34,'L':0.54,'R':0.95,'K':0.61,'N':2.49,
    'M':0.70,'D':3.32,'F':0.80,'C':1.07,'P':2.12,
    'Q':1.49,'S':0.94,'E':2.20,'T':1.09,'G':2.07,
    'W':-4.65,'H':1.27,'Y':-0.17,'I':0.66,'V':1.32}

    OOBM850103={'A':0.46,'L':0.43,'R':-1.54,'K':-1.71,'N':1.31,
    'M':0.15,'D':-0.33,'F':0.52,'C':0.20,'P':-0.58,
    'Q':-1.12,'S':-0.83,'E':0.48,'T':-1.52,'G':0.64,
    'W':1.25,'H':-1.31,'Y':-2.21,'I':3.28,'V':0.54}

    OOBM850104={'A':-2.49,'L':-7.16,'R':2.55,'K':-9.97,'N':2.27,
    'M':-4.96,'D':8.86,'F':-6.64,'C':-3.13,'P':5.19,
    'Q':1.79,'S':-1.60,'E':4.04,'T':-4.75,'G':-0.56,
    'W':-17.84,'H':4.22,'Y':9.25,'I':-10.87,'V':-3.97}

    OOBM850105={'A':4.55,'L':3.24,'R':5.97,'K':10.68,'N':5.56,
    'M':2.18,'D':2.85,'F':4.37,'C':-0.78,'P':5.14,
    'Q':4.15,'S':6.78,'E':5.16,'T':8.60,'G':9.14,
    'W':1.97,'H':4.48,'Y':2.40,'I':2.10,'V':3.81}

    PALJ810101={'A':1.30,'L':1.30,'R':0.93,'K':1.23,'N':0.90,
    'M':1.32,'D':1.02,'F':1.09,'C':0.92,'P':0.63,
    'Q':1.04,'S':0.78,'E':1.43,'T':0.80,'G':0.63,
    'W':1.03,'H':1.33,'Y':0.71,'I':0.87,'V':0.95}

    PALJ810102={'A':1.32,'L':1.22,'R':1.04,'K':1.13,'N':0.74,
    'M':1.47,'D':0.97,'F':1.10,'C':0.70,'P':0.57,
    'Q':1.25,'S':0.77,'E':1.48,'T':0.86,'G':0.59,
    'W':1.02,'H':1.06,'Y':0.72,'I':1.01,'V':1.05}

    PALJ810103={'A':0.81,'L':1.03,'R':1.03,'K':0.77,'N':0.81,
    'M':0.96,'D':0.71,'F':1.13,'C':1.12,'P':0.75,
    'Q':1.03,'S':1.02,'E':0.59,'T':1.19,'G':0.94,
    'W':1.24,'H':0.85,'Y':1.35,'I':1.47,'V':1.44}

    PALJ810104={'A':0.90,'L':1.24,'R':0.75,'K':0.75,'N':0.82,
    'M':0.94,'D':0.75,'F':1.41,'C':1.12,'P':0.46,
    'Q':0.95,'S':0.70,'E':0.44,'T':1.20,'G':0.83,
    'W':1.28,'H':0.86,'Y':1.45,'I':1.59,'V':1.73}

    PALJ810105={'A':0.84,'L':0.49,'R':0.91,'K':0.95,'N':1.48,
    'M':0.52,'D':1.28,'F':0.88,'C':0.69,'P':1.47,
    'Q':1.0,'S':1.29,'E':0.78,'T':1.05,'G':1.76,
    'W':0.88,'H':0.53,'Y':1.28,'I':0.55,'V':0.51}

    PALJ810106={'A':0.65,'L':0.56,'R':0.93,'K':0.95,'N':1.45,
    'M':0.71,'D':1.47,'F':0.72,'C':1.43,'P':1.51,
    'Q':0.94,'S':1.46,'E':0.75,'T':0.96,'G':1.53,
    'W':0.90,'H':0.96,'Y':1.12,'I':0.57,'V':0.55}

    PALJ810107={'A':1.08,'L':1.04,'R':0.93,'K':1.01,'N':1.05,
    'M':1.11,'D':0.86,'F':0.96,'C':1.22,'P':0.91,
    'Q':0.95,'S':0.95,'E':1.09,'T':1.15,'G':0.85,
    'W':1.17,'H':1.02,'Y':0.80,'I':0.98,'V':1.03}

    PALJ810108={'A':1.34,'L':1.39,'R':0.91,'K':1.08,'N':0.83,
    'M':0.90,'D':1.06,'F':1.02,'C':1.27,'P':0.48,
    'Q':1.13,'S':1.05,'E':1.69,'T':0.74,'G':0.47,
    'W':0.64,'H':1.11,'Y':0.73,'I':0.84,'V':1.18}

    PALJ810109={'A':1.15,'L':1.22,'R':1.06,'K':1.20,'N':0.87,
    'M':1.45,'D':1.0,'F':0.92,'C':1.03,'P':0.72,
    'Q':1.43,'S':0.84,'E':1.37,'T':0.97,'G':0.64,
    'W':1.11,'H':0.95,'Y':0.72,'I':0.99,'V':0.82}

    PALJ810110={'A':0.89,'L':1.02,'R':1.06,'K':1.0,'N':0.67,
    'M':1.41,'D':0.71,'F':1.32,'C':1.04,'P':0.69,
    'Q':1.06,'S':0.86,'E':0.72,'T':1.15,'G':0.87,
    'W':1.06,'H':1.04,'Y':1.35,'I':1.14,'V':1.66}

    PALJ810111={'A':0.82,'L':0.94,'R':0.99,'K':0.73,'N':1.27,
    'M':1.30,'D':0.98,'F':1.56,'C':0.71,'P':0.69,
    'Q':1.01,'S':0.65,'E':0.54,'T':0.98,'G':0.94,
    'W':1.25,'H':1.26,'Y':1.26,'I':1.67,'V':1.22}

    PALJ810112={'A':0.98,'L':1.05,'R':1.03,'K':0.83,'N':0.66,
    'M':0.82,'D':0.74,'F':1.23,'C':1.01,'P':0.73,
    'Q':0.63,'S':0.98,'E':0.59,'T':1.20,'G':0.90,
    'W':1.26,'H':1.17,'Y':1.23,'I':1.38,'V':1.62}

    PALJ810113={'A':0.69,'L':0.0,'R':0.0,'K':1.18,'N':1.52,
    'M':0.88,'D':2.42,'F':2.20,'C':0.0,'P':1.34,
    'Q':1.44,'S':1.43,'E':0.63,'T':0.28,'G':2.64,
    'W':0.0,'H':0.22,'Y':1.53,'I':0.43,'V':0.14}

    PALJ810114={'A':0.87,'L':0.67,'R':1.30,'K':0.66,'N':1.36,
    'M':0.0,'D':1.24,'F':0.47,'C':0.83,'P':1.54,
    'Q':1.06,'S':1.08,'E':0.91,'T':1.12,'G':1.69,
    'W':1.24,'H':0.91,'Y':0.54,'I':0.27,'V':0.69}

    PALJ810115={'A':0.91,'L':0.77,'R':0.77,'K':1.27,'N':1.32,
    'M':0.76,'D':0.90,'F':0.37,'C':0.50,'P':1.62,
    'Q':1.06,'S':1.34,'E':0.53,'T':0.87,'G':1.61,
    'W':1.10,'H':1.08,'Y':1.24,'I':0.36,'V':0.52}

    PALJ810116={'A':0.92,'L':0.50,'R':0.90,'K':0.86,'N':1.57,
    'M':0.50,'D':1.22,'F':0.96,'C':0.62,'P':1.30,
    'Q':0.66,'S':1.40,'E':0.92,'T':1.11,'G':1.61,
    'W':0.57,'H':0.39,'Y':1.78,'I':0.79,'V':0.50}

    PARJ860101={'A':2.1,'L':-9.2,'R':4.2,'K':5.7,'N':7.0,
    'M':-4.2,'D':10.0,'F':-9.2,'C':1.4,'P':2.1,
    'Q':6.0,'S':6.5,'E':7.8,'T':5.2,'G':5.7,
    'W':-10.0,'H':2.1,'Y':-1.9,'I':-8.0,'V':-3.7}

    PLIV810101={'A':-2.89,'L':-1.61,'R':-3.30,'K':-3.31,'N':-3.41,
    'M':-1.84,'D':-3.38,'F':-1.63,'C':-2.49,'P':-2.50,
    'Q':-3.15,'S':-3.30,'E':-2.94,'T':-2.91,'G':-3.25,
    'W':-1.75,'H':-2.84,'Y':-2.42,'I':-1.72,'V':-2.08}

    PONP800101={'A':12.28,'L':14.10,'R':11.49,'K':10.80,'N':11.00,
    'M':14.33,'D':10.97,'F':13.43,'C':14.93,'P':11.19,
    'Q':11.28,'S':11.26,'E':11.19,'T':11.65,'G':12.01,
    'W':12.95,'H':12.84,'Y':13.29,'I':14.77,'V':15.07}

    PONP800102={'A':7.62,'L':9.37,'R':6.81,'K':5.72,'N':6.17,
    'M':9.83,'D':6.18,'F':8.99,'C':10.93,'P':6.64,
    'Q':6.67,'S':6.93,'E':6.38,'T':7.08,'G':7.31,
    'W':8.41,'H':7.85,'Y':8.53,'I':9.99,'V':10.38}

    PONP800103={'A':2.63,'L':2.98,'R':2.45,'K':2.12,'N':2.27,
    'M':3.18,'D':2.29,'F':3.02,'C':3.36,'P':2.46,
    'Q':2.45,'S':2.60,'E':2.31,'T':2.55,'G':2.55,
    'W':2.85,'H':2.57,'Y':2.79,'I':3.08,'V':3.21}

    PONP800104={'A':13.65,'L':14.01,'R':11.28,'K':11.96,'N':12.24,
    'M':13.40,'D':10.98,'F':14.08,'C':14.49,'P':11.51,
    'Q':11.30,'S':11.26,'E':12.55,'T':13.00,'G':15.36,
    'W':12.06,'H':11.59,'Y':12.64,'I':14.63,'V':12.88}

    PONP800105={'A':14.60,'L':16.49,'R':13.24,'K':13.28,'N':11.79,
    'M':16.23,'D':13.78,'F':14.18,'C':15.90,'P':14.10,
    'Q':12.02,'S':13.36,'E':13.59,'T':14.50,'G':14.18,
    'W':13.90,'H':15.35,'Y':14.76,'I':14.10,'V':16.30}

    PONP800106={'A':10.67,'L':13.07,'R':11.05,'K':9.93,'N':10.85,
    'M':15.00,'D':10.21,'F':13.27,'C':14.15,'P':10.62,
    'Q':11.71,'S':11.18,'E':11.71,'T':10.53,'G':10.95,
    'W':11.41,'H':12.07,'Y':11.52,'I':12.95,'V':13.86}

    PONP800107={'A':3.70,'L':5.88,'R':2.53,'K':1.79,'N':2.12,
    'M':5.21,'D':2.60,'F':6.60,'C':3.03,'P':2.12,
    'Q':2.70,'S':2.43,'E':3.30,'T':2.60,'G':3.13,
    'W':6.25,'H':3.57,'Y':3.03,'I':7.69,'V':7.14}

    PONP800108={'A':6.05,'L':7.37,'R':5.70,'K':4.88,'N':5.04,
    'M':6.39,'D':4.95,'F':6.62,'C':7.86,'P':5.65,
    'Q':5.45,'S':5.53,'E':5.10,'T':5.81,'G':6.16,
    'W':6.98,'H':5.80,'Y':6.73,'I':7.51,'V':7.62}

    PRAM820101={'A':0.305,'L':0.262,'R':0.227,'K':0.391,'N':0.322,
    'M':0.280,'D':0.335,'F':0.195,'C':0.339,'P':0.346,
    'Q':0.306,'S':0.326,'E':0.282,'T':0.251,'G':0.352,
    'W':0.291,'H':0.215,'Y':0.293,'I':0.278,'V':0.291}

    PRAM820102={'A':0.175,'L':0.104,'R':0.083,'K':0.058,'N':0.090,
    'M':0.054,'D':0.140,'F':0.104,'C':0.074,'P':0.136,
    'Q':0.093,'S':0.155,'E':0.135,'T':0.152,'G':0.201,
    'W':0.092,'H':0.125,'Y':0.081,'I':0.100,'V':0.096}

    PRAM820103={'A':0.687,'L':0.541,'R':0.590,'K':0.407,'N':0.489,
    'M':0.328,'D':0.632,'F':0.577,'C':0.263,'P':0.600,
    'Q':0.527,'S':0.692,'E':0.669,'T':0.713,'G':0.670,
    'W':0.632,'H':0.594,'Y':0.495,'I':0.564,'V':0.529}

    PRAM900101={'A':-6.70,'L':-11.70,'R':51.50,'K':36.80,'N':20.10,
    'M':-14.20,'D':38.50,'F':-15.50,'C':-8.40,'P':0.80,
    'Q':17.20,'S':-2.50,'E':34.30,'T':-5.0,'G':-4.20,
    'W':-7.90,'H':12.60,'Y':2.90,'I':-13.0,'V':-10.90}

    PRAM900102={'A':1.29,'L':1.30,'R':0.96,'K':1.23,'N':0.90,
    'M':1.47,'D':1.04,'F':1.07,'C':1.11,'P':0.52,
    'Q':1.27,'S':0.82,'E':1.44,'T':0.82,'G':0.56,
    'W':0.99,'H':1.22,'Y':0.72,'I':0.97,'V':0.91}

    PRAM900103={'A':0.90,'L':1.02,'R':0.99,'K':0.77,'N':0.76,
    'M':0.97,'D':0.72,'F':1.32,'C':0.74,'P':0.64,
    'Q':0.80,'S':0.95,'E':0.75,'T':1.21,'G':0.92,
    'W':1.14,'H':1.08,'Y':1.25,'I':1.45,'V':1.49}

    PRAM900104={'A':0.78,'L':0.59,'R':0.88,'K':0.96,'N':1.28,
    'M':0.39,'D':1.41,'F':0.58,'C':0.80,'P':1.91,
    'Q':0.97,'S':1.33,'E':1.0,'T':1.03,'G':1.64,
    'W':0.75,'H':0.69,'Y':1.05,'I':0.51,'V':0.47}

    PTIO830101={'A':1.10,'L':1.25,'R':0.95,'K':1.00,'N':0.80,
    'M':1.15,'D':0.65,'F':1.10,'C':0.95,'P':0.10,
    'Q':1.00,'S':0.75,'E':1.00,'T':0.75,'G':0.60,
    'W':1.10,'H':0.85,'Y':1.10,'I':1.10,'V':0.95}

    PTIO830102={'A':1.00,'L':2.00,'R':0.70,'K':0.70,'N':0.60,
    'M':1.90,'D':0.50,'F':3.10,'C':1.90,'P':0.20,
    'Q':1.00,'S':0.90,'E':0.70,'T':1.70,'G':0.30,
    'W':2.20,'H':0.80,'Y':2.80,'I':4.00,'V':4.00}

    QIAN880101={'A':0.12,'L':0.05,'R':0.04,'K':0.26,'N':-0.10,
    'M':0.00,'D':0.01,'F':0.05,'C':-0.25,'P':-0.19,
    'Q':-0.03,'S':-0.19,'E':-0.02,'T':-0.04,'G':-0.02,
    'W':-0.06,'H':-0.06,'Y':-0.14,'I':-0.07,'V':-0.03}

    QIAN880102={'A':0.26,'L':-0.02,'R':-0.14,'K':0.12,'N':-0.03,
    'M':0.00,'D':0.15,'F':0.12,'C':-0.15,'P':-0.08,
    'Q':-0.13,'S':0.01,'E':0.21,'T':-0.34,'G':-0.37,
    'W':-0.01,'H':0.10,'Y':-0.29,'I':-0.03,'V':0.02}

    QIAN880103={'A':0.64,'L':0.41,'R':-0.10,'K':-0.17,'N':0.09,
    'M':0.13,'D':0.33,'F':-0.03,'C':0.03,'P':-0.43,
    'Q':-0.23,'S':-0.10,'E':0.51,'T':-0.07,'G':-0.09,
    'W':-0.02,'H':-0.23,'Y':-0.38,'I':-0.22,'V':-0.01}

    QIAN880104={'A':0.29,'L':0.47,'R':-0.03,'K':-0.19,'N':-0.04,
    'M':0.27,'D':0.11,'F':0.24,'C':-0.05,'P':-0.34,
    'Q':0.26,'S':-0.17,'E':0.28,'T':-0.20,'G':-0.67,
    'W':0.25,'H':-0.26,'Y':-0.30,'I':0.00,'V':-0.01}

    QIAN880105={'A':0.68,'L':0.61,'R':-0.22,'K':0.03,'N':-0.09,
    'M':0.39,'D':-0.02,'F':0.06,'C':-0.15,'P':-0.76,
    'Q':-0.15,'S':-0.26,'E':0.44,'T':-0.10,'G':-0.73,
    'W':0.20,'H':-0.14,'Y':-0.04,'I':-0.08,'V':0.12}

    QIAN880106={'A':0.34,'L':0.20,'R':0.22,'K':-0.11,'N':-0.33,
    'M':0.43,'D':0.06,'F':0.15,'C':-0.18,'P':-0.81,
    'Q':0.01,'S':-0.35,'E':0.20,'T':-0.37,'G':-0.88,
    'W':0.07,'H':-0.09,'Y':-0.31,'I':-0.03,'V':0.13}

    QIAN880107={'A':0.57,'L':0.48,'R':0.23,'K':0.16,'N':-0.36,
    'M':0.41,'D':-0.46,'F':0.03,'C':-0.15,'P':-1.12,
    'Q':0.15,'S':-0.47,'E':0.26,'T':-0.54,'G':-0.71,
    'W':-0.10,'H':-0.05,'Y':-0.35,'I':0.00,'V':0.31}

    QIAN880108={'A':0.33,'L':0.57,'R':0.10,'K':0.23,'N':-0.19,
    'M':0.79,'D':-0.44,'F':0.48,'C':-0.03,'P':-1.86,
    'Q':0.19,'S':-0.23,'E':0.21,'T':-0.33,'G':-0.46,
    'W':0.15,'H':0.27,'Y':-0.19,'I':-0.33,'V':0.24}

    QIAN880109={'A':0.13,'L':0.50,'R':0.08,'K':0.37,'N':-0.07,
    'M':0.63,'D':-0.71,'F':0.15,'C':-0.09,'P':-1.40,
    'Q':0.12,'S':-0.28,'E':0.13,'T':-0.21,'G':-0.39,
    'W':0.02,'H':0.32,'Y':-0.10,'I':0.00,'V':0.17}

    QIAN880110={'A':0.31,'L':0.56,'R':0.18,'K':0.47,'N':-0.10,
    'M':0.58,'D':-0.81,'F':0.10,'C':-0.26,'P':-1.33,
    'Q':0.41,'S':-0.49,'E':-0.06,'T':-0.44,'G':-0.42,
    'W':0.14,'H':0.51,'Y':-0.08,'I':-0.15,'V':-0.01}

    QIAN880111={'A':0.21,'L':0.70,'R':0.07,'K':0.28,'N':-0.04,
    'M':0.61,'D':-0.58,'F':-0.06,'C':-0.12,'P':-1.03,
    'Q':0.13,'S':-0.28,'E':-0.23,'T':-0.25,'G':-0.15,
    'W':0.21,'H':0.37,'Y':0.16,'I':0.31,'V':0.00}

    QIAN880112={'A':0.18,'L':0.62,'R':0.21,'K':0.41,'N':-0.03,
    'M':0.21,'D':-0.32,'F':0.05,'C':-0.29,'P':-0.84,
    'Q':-0.27,'S':-0.05,'E':-0.25,'T':-0.16,'G':-0.40,
    'W':0.32,'H':0.28,'Y':0.11,'I':-0.03,'V':0.06}

    QIAN880113={'A':-0.08,'L':0.28,'R':0.05,'K':0.45,'N':-0.08,
    'M':0.11,'D':-0.24,'F':0.00,'C':-0.25,'P':-0.42,
    'Q':-0.28,'S':0.07,'E':-0.19,'T':-0.33,'G':-0.10,
    'W':0.36,'H':0.29,'Y':0.00,'I':-0.01,'V':-0.13}

    QIAN880114={'A':-0.18,'L':-0.23,'R':-0.13,'K':0.03,'N':0.28,
    'M':-0.42,'D':0.05,'F':-0.18,'C':-0.26,'P':-0.13,
    'Q':0.21,'S':0.41,'E':-0.06,'T':0.33,'G':0.23,
    'W':-0.10,'H':0.24,'Y':-0.10,'I':-0.42,'V':-0.07}

    QIAN880115={'A':-0.01,'L':-0.25,'R':0.02,'K':0.08,'N':0.41,
    'M':-0.57,'D':-0.09,'F':-0.12,'C':-0.27,'P':0.26,
    'Q':0.01,'S':0.44,'E':0.09,'T':0.35,'G':0.13,
    'W':-0.15,'H':0.22,'Y':0.15,'I':-0.27,'V':-0.09}

    QIAN880116={'A':-0.19,'L':-0.42,'R':0.03,'K':-0.09,'N':0.02,
    'M':-0.38,'D':-0.06,'F':-0.32,'C':-0.29,'P':0.05,
    'Q':0.02,'S':0.25,'E':-0.10,'T':0.22,'G':0.19,
    'W':-0.19,'H':-0.16,'Y':0.05,'I':-0.08,'V':-0.15}

    QIAN880117={'A':-0.14,'L':-0.57,'R':0.14,'K':0.04,'N':-0.27,
    'M':0.24,'D':-0.10,'F':0.08,'C':-0.64,'P':0.02,
    'Q':-0.11,'S':-0.12,'E':-0.39,'T':0.00,'G':0.46,
    'W':-0.10,'H':-0.04,'Y':0.18,'I':0.16,'V':0.29}

    QIAN880118={'A':-0.31,'L':0.09,'R':0.25,'K':-0.29,'N':-0.53,
    'M':0.29,'D':-0.54,'F':0.24,'C':-0.06,'P':-0.31,
    'Q':0.07,'S':0.11,'E':-0.52,'T':0.03,'G':0.37,
    'W':0.15,'H':-0.32,'Y':0.29,'I':0.57,'V':0.48}

    QIAN880119={'A':-0.10,'L':0.32,'R':0.19,'K':-0.46,'N':-0.89,
    'M':0.43,'D':-0.89,'F':0.36,'C':0.13,'P':-0.91,
    'Q':-0.04,'S':-0.12,'E':-0.34,'T':0.49,'G':-0.45,
    'W':0.34,'H':-0.34,'Y':0.42,'I':0.95,'V':0.76}

    QIAN880120={'A':-0.25,'L':0.23,'R':-0.02,'K':-0.59,'N':-0.77,
    'M':0.32,'D':-1.01,'F':0.48,'C':0.13,'P':-1.24,
    'Q':-0.12,'S':-0.31,'E':-0.62,'T':0.17,'G':-0.72,
    'W':0.45,'H':-0.16,'Y':0.77,'I':1.10,'V':0.69}

    QIAN880121={'A':-0.26,'L':0.25,'R':-0.09,'K':-0.55,'N':-0.34,
    'M':-0.05,'D':-0.55,'F':0.20,'C':0.47,'P':-1.28,
    'Q':-0.33,'S':-0.28,'E':-0.75,'T':0.08,'G':-0.56,
    'W':0.22,'H':-0.04,'Y':0.53,'I':0.94,'V':0.67}

    QIAN880122={'A':0.05,'L':0.32,'R':-0.11,'K':-0.51,'N':-0.40,
    'M':-0.10,'D':-0.11,'F':0.20,'C':0.36,'P':-0.79,
    'Q':-0.67,'S':0.03,'E':-0.35,'T':-0.15,'G':0.14,
    'W':0.09,'H':0.02,'Y':0.34,'I':0.47,'V':0.58}

    QIAN880123={'A':-0.44,'L':-0.12,'R':-0.13,'K':-0.33,'N':0.05,
    'M':-0.21,'D':-0.20,'F':-0.13,'C':0.13,'P':-0.48,
    'Q':-0.58,'S':0.27,'E':-0.28,'T':0.47,'G':0.08,
    'W':-0.22,'H':0.09,'Y':-0.11,'I':-0.04,'V':0.06}

    QIAN880124={'A':-0.31,'L':-0.44,'R':-0.10,'K':-0.44,'N':0.06,
    'M':-0.28,'D':0.13,'F':-0.04,'C':-0.11,'P':-0.29,
    'Q':-0.47,'S':0.34,'E':-0.05,'T':0.27,'G':0.45,
    'W':-0.08,'H':-0.06,'Y':0.06,'I':-0.25,'V':0.11}

    QIAN880125={'A':-0.02,'L':-0.26,'R':0.04,'K':-0.39,'N':0.03,
    'M':-0.14,'D':0.11,'F':-0.03,'C':-0.02,'P':-0.04,
    'Q':-0.17,'S':0.41,'E':0.10,'T':0.36,'G':0.38,
    'W':-0.01,'H':-0.09,'Y':-0.08,'I':-0.48,'V':-0.18}

    QIAN880126={'A':-0.06,'L':-0.46,'R':0.02,'K':-0.43,'N':0.10,
    'M':-0.52,'D':0.24,'F':-0.33,'C':-0.19,'P':0.37,
    'Q':-0.04,'S':0.43,'E':-0.04,'T':0.50,'G':0.17,
    'W':-0.32,'H':0.19,'Y':0.35,'I':-0.20,'V':0.00}

    QIAN880127={'A':-0.05,'L':0.04,'R':0.06,'K':-0.42,'N':0.00,
    'M':0.25,'D':0.15,'F':0.09,'C':0.30,'P':0.31,
    'Q':-0.08,'S':-0.11,'E':-0.02,'T':-0.06,'G':-0.14,
    'W':0.19,'H':-0.07,'Y':0.33,'I':0.26,'V':0.04}

    QIAN880128={'A':-0.19,'L':0.34,'R':0.17,'K':-0.20,'N':-0.38,
    'M':0.45,'D':0.09,'F':0.07,'C':0.41,'P':0.04,
    'Q':0.04,'S':-0.23,'E':-0.20,'T':-0.02,'G':0.28,
    'W':0.16,'H':-0.19,'Y':0.22,'I':-0.06,'V':0.05}

    QIAN880129={'A':-0.43,'L':-0.10,'R':0.06,'K':0.33,'N':0.00,
    'M':-0.01,'D':-0.31,'F':0.25,'C':0.19,'P':0.28,
    'Q':0.14,'S':-0.23,'E':-0.41,'T':-0.26,'G':-0.21,
    'W':0.15,'H':0.21,'Y':0.09,'I':0.29,'V':-0.10}

    QIAN880130={'A':-0.19,'L':-0.22,'R':-0.07,'K':0.00,'N':0.17,
    'M':-0.53,'D':-0.27,'F':-0.31,'C':0.42,'P':0.14,
    'Q':-0.29,'S':0.22,'E':-0.22,'T':0.10,'G':0.17,
    'W':-0.15,'H':0.17,'Y':-0.02,'I':-0.34,'V':-0.33}

    QIAN880131={'A':-0.25,'L':-0.55,'R':0.12,'K':0.14,'N':0.61,
    'M':-0.47,'D':0.60,'F':-0.29,'C':0.18,'P':0.89,
    'Q':0.09,'S':0.24,'E':-0.12,'T':0.16,'G':0.09,
    'W':-0.44,'H':0.42,'Y':-0.19,'I':-0.54,'V':-0.45}

    QIAN880132={'A':-0.27,'L':-0.54,'R':-0.40,'K':0.45,'N':0.71,
    'M':-0.76,'D':0.54,'F':-0.47,'C':0.00,'P':1.40,
    'Q':-0.08,'S':0.40,'E':-0.12,'T':-0.10,'G':1.14,
    'W':-0.46,'H':0.18,'Y':-0.05,'I':-0.74,'V':-0.86}

    QIAN880133={'A':-0.42,'L':-0.69,'R':-0.23,'K':0.09,'N':0.81,
    'M':-0.86,'D':0.95,'F':-0.39,'C':-0.18,'P':1.77,
    'Q':-0.01,'S':0.63,'E':-0.09,'T':0.29,'G':1.24,
    'W':-0.37,'H':0.05,'Y':-0.41,'I':-1.17,'V':-1.32}

    QIAN880134={'A':-0.24,'L':-0.80,'R':-0.04,'K':0.17,'N':0.45,
    'M':-0.71,'D':0.65,'F':-0.61,'C':-0.38,'P':2.27,
    'Q':0.01,'S':0.33,'E':0.07,'T':0.13,'G':0.85,
    'W':-0.44,'H':-0.21,'Y':-0.49,'I':-0.65,'V':-0.99}

    QIAN880135={'A':-0.14,'L':-0.80,'R':0.21,'K':-0.14,'N':0.35,
    'M':-0.56,'D':0.66,'F':-0.25,'C':-0.09,'P':1.59,
    'Q':0.11,'S':0.32,'E':0.06,'T':0.21,'G':0.36,
    'W':-0.17,'H':-0.31,'Y':-0.35,'I':-0.51,'V':-0.70}

    QIAN880136={'A':0.01,'L':-0.81,'R':-0.13,'K':-0.43,'N':-0.11,
    'M':-0.49,'D':0.78,'F':-0.20,'C':-0.31,'P':1.14,
    'Q':-0.13,'S':0.13,'E':0.09,'T':-0.02,'G':0.14,
    'W':-0.20,'H':-0.56,'Y':0.10,'I':-0.09,'V':-0.11}

    QIAN880137={'A':-0.30,'L':-0.18,'R':-0.09,'K':0.06,'N':-0.12,
    'M':-0.44,'D':0.44,'F':0.11,'C':0.03,'P':0.77,
    'Q':0.24,'S':-0.09,'E':0.18,'T':-0.27,'G':-0.12,
    'W':-0.09,'H':-0.20,'Y':-0.25,'I':-0.07,'V':-0.06}

    QIAN880138={'A':-0.23,'L':-0.36,'R':-0.20,'K':-0.15,'N':0.06,
    'M':-0.19,'D':0.34,'F':-0.02,'C':0.19,'P':0.78,
    'Q':0.47,'S':-0.29,'E':0.28,'T':-0.30,'G':0.14,
    'W':-0.18,'H':-0.22,'Y':0.07,'I':0.42,'V':0.29}

    QIAN880139={'A':0.08,'L':0.24,'R':-0.01,'K':-0.27,'N':-0.06,
    'M':0.16,'D':0.04,'F':0.34,'C':0.37,'P':0.16,
    'Q':0.48,'S':-0.35,'E':0.36,'T':-0.04,'G':-0.02,
    'W':-0.06,'H':-0.45,'Y':-0.20,'I':0.09,'V':0.18}

    RACS770101={'A':0.934,'L':0.825,'R':0.962,'K':1.040,'N':0.986,
    'M':0.804,'D':0.994,'F':0.773,'C':0.900,'P':1.047,
    'Q':1.047,'S':1.056,'E':0.986,'T':1.008,'G':1.015,
    'W':0.848,'H':0.882,'Y':0.931,'I':0.766,'V':0.825}

    RACS770102={'A':0.941,'L':0.798,'R':1.112,'K':1.232,'N':1.038,
    'M':0.781,'D':1.071,'F':0.723,'C':0.866,'P':1.093,
    'Q':1.150,'S':1.082,'E':1.100,'T':1.043,'G':1.055,
    'W':0.867,'H':0.911,'Y':1.050,'I':0.742,'V':0.817}

    RACS770103={'A':1.16,'L':0.51,'R':1.72,'K':3.90,'N':1.97,
    'M':0.40,'D':2.66,'F':0.43,'C':0.50,'P':2.04,
    'Q':3.87,'S':1.61,'E':2.40,'T':1.48,'G':1.63,
    'W':0.75,'H':0.86,'Y':1.72,'I':0.57,'V':0.59}

    RACS820101={'A':0.85,'L':1.03,'R':2.02,'K':0.88,'N':0.88,
    'M':1.17,'D':1.50,'F':0.85,'C':0.90,'P':1.47,
    'Q':1.71,'S':1.50,'E':1.79,'T':1.96,'G':1.54,
    'W':0.83,'H':1.59,'Y':1.34,'I':0.67,'V':0.89}

    RACS820102={'A':1.58,'L':1.21,'R':1.14,'K':1.27,'N':0.77,
    'M':1.41,'D':0.98,'F':1.00,'C':1.04,'P':1.46,
    'Q':1.24,'S':1.05,'E':1.49,'T':0.87,'G':0.66,
    'W':1.23,'H':0.99,'Y':0.68,'I':1.09,'V':0.88}

    RACS820103={'A':0.82,'L':0.00,'R':2.60,'K':2.86,'N':2.07,
    'M':0.00,'D':2.64,'F':0.00,'C':0.00,'P':0.00,
    'Q':0.00,'S':1.23,'E':2.62,'T':2.48,'G':1.63,
    'W':0.00,'H':0.00,'Y':1.90,'I':2.32,'V':1.62}

    RACS820104={'A':0.78,'L':0.91,'R':1.75,'K':0.85,'N':1.32,
    'M':0.41,'D':1.25,'F':1.07,'C':3.14,'P':1.73,
    'Q':0.93,'S':1.31,'E':0.94,'T':1.57,'G':1.13,
    'W':0.98,'H':1.03,'Y':1.31,'I':1.26,'V':1.11}

    RACS820105={'A':0.88,'L':1.09,'R':0.99,'K':0.83,'N':1.02,
    'M':1.71,'D':1.16,'F':1.52,'C':1.14,'P':0.87,
    'Q':0.93,'S':1.14,'E':1.01,'T':0.96,'G':0.70,
    'W':1.96,'H':1.87,'Y':1.68,'I':1.61,'V':1.56}

    RACS820106={'A':0.30,'L':0.96,'R':0.90,'K':0.71,'N':2.73,
    'M':1.89,'D':1.26,'F':1.20,'C':0.72,'P':0.83,
    'Q':0.97,'S':1.16,'E':1.33,'T':0.97,'G':3.09,
    'W':1.58,'H':1.33,'Y':0.86,'I':0.45,'V':0.64}

    RACS820107={'A':0.40,'L':0.57,'R':1.20,'K':0.87,'N':1.24,
    'M':0.00,'D':1.59,'F':1.27,'C':2.98,'P':0.38,
    'Q':0.50,'S':0.92,'E':1.26,'T':1.38,'G':1.89,
    'W':1.53,'H':2.71,'Y':1.79,'I':1.31,'V':0.95}

    RACS820108={'A':1.48,'L':1.33,'R':1.02,'K':1.36,'N':0.99,
    'M':1.41,'D':1.19,'F':1.30,'C':0.86,'P':0.25,
    'Q':1.42,'S':0.89,'E':1.43,'T':0.81,'G':0.46,
    'W':1.27,'H':1.27,'Y':0.91,'I':1.12,'V':0.93}

    RACS820109={'A':0.00,'L':0.00,'R':0.00,'K':0.00,'N':4.14,
    'M':0.00,'D':2.15,'F':2.11,'C':0.00,'P':1.99,
    'Q':0.00,'S':0.00,'E':0.00,'T':1.24,'G':6.49,
    'W':0.00,'H':0.00,'Y':1.90,'I':0.00,'V':0.00}

    RACS820110={'A':1.02,'L':1.06,'R':1.00,'K':0.94,'N':1.31,
    'M':1.33,'D':1.76,'F':0.41,'C':1.05,'P':2.73,
    'Q':1.05,'S':1.18,'E':0.83,'T':0.77,'G':2.39,
    'W':1.22,'H':0.40,'Y':1.09,'I':0.83,'V':0.88}

    RACS820111={'A':0.93,'L':1.03,'R':1.52,'K':1.00,'N':0.92,
    'M':1.31,'D':0.60,'F':1.51,'C':1.08,'P':1.37,
    'Q':0.94,'S':0.97,'E':0.73,'T':1.38,'G':0.78,
    'W':1.12,'H':1.08,'Y':1.65,'I':1.74,'V':1.70}

    RACS820112={'A':0.99,'L':1.26,'R':1.19,'K':0.91,'N':1.15,
    'M':1.00,'D':1.18,'F':1.25,'C':2.32,'P':0.00,
    'Q':1.52,'S':1.50,'E':1.36,'T':1.18,'G':1.40,
    'W':1.33,'H':1.06,'Y':1.09,'I':0.81,'V':1.01}

    RACS820113={'A':17.05,'L':10.89,'R':21.25,'K':16.46,'N':34.81,
    'M':20.61,'D':19.27,'F':16.26,'C':28.84,'P':23.94,
    'Q':15.42,'S':19.95,'E':20.12,'T':18.92,'G':38.14,
    'W':23.36,'H':23.07,'Y':26.49,'I':16.66,'V':17.06}

    RACS820114={'A':14.53,'L':14.30,'R':17.82,'K':14.07,'N':13.59,
    'M':20.61,'D':19.78,'F':19.61,'C':30.57,'P':52.63,
    'Q':22.18,'S':18.56,'E':18.19,'T':21.09,'G':37.16,
    'W':19.78,'H':22.63,'Y':26.36,'I':20.28,'V':21.87}

    RADA880101={'A':1.81,'L':4.92,'R':-14.92,'K':-5.55,'N':-6.64,
    'M':2.35,'D':-8.72,'F':2.98,'C':1.28,'P':0.0,
    'Q':-5.54,'S':-3.40,'E':-6.81,'T':-2.57,'G':0.94,
    'W':2.33,'H':-4.66,'Y':-0.14,'I':4.92,'V':4.04}

    RADA880102={'A':0.52,'L':1.76,'R':-1.32,'K':0.08,'N':-0.01,
    'M':1.32,'D':0.0,'F':2.09,'C':0.0,'P':0.0,
    'Q':-0.07,'S':0.04,'E':-0.79,'T':0.27,'G':0.0,
    'W':2.51,'H':0.95,'Y':1.63,'I':2.04,'V':1.18}

    RADA880103={'A':0.13,'L':-2.64,'R':-5.0,'K':-3.97,'N':-3.04,
    'M':-3.83,'D':-2.23,'F':-3.74,'C':-2.52,'P':0.0,
    'Q':-3.84,'S':-1.66,'E':-3.43,'T':-2.31,'G':1.45,
    'W':-8.21,'H':-5.61,'Y':-5.97,'I':-2.77,'V':-2.05}

    RADA880104={'A':1.29,'L':3.16,'R':-13.60,'K':-5.63,'N':-6.63,
    'M':1.03,'D':0.0,'F':0.89,'C':0.0,'P':0.0,
    'Q':-5.47,'S':-3.44,'E':-6.02,'T':-2.84,'G':0.94,
    'W':-0.18,'H':-5.61,'Y':-1.77,'I':2.88,'V':2.86}

    RADA880105={'A':1.42,'L':0.52,'R':-18.60,'K':-9.60,'N':-9.67,
    'M':-2.80,'D':0.0,'F':-2.85,'C':0.0,'P':0.0,
    'Q':-9.31,'S':-5.10,'E':-9.45,'T':-5.15,'G':2.39,
    'W':-8.39,'H':-11.22,'Y':-7.74,'I':0.11,'V':0.81}

    RADA880106={'A':93.7,'L':173.7,'R':250.4,'K':215.2,'N':146.3,
    'M':197.6,'D':142.6,'F':228.6,'C':135.2,'P':0.0,
    'Q':177.7,'S':109.5,'E':182.9,'T':142.1,'G':52.6,
    'W':271.6,'H':188.1,'Y':239.9,'I':182.2,'V':157.2}

    RADA880107={'A':-0.29,'L':-0.12,'R':-2.71,'K':-2.05,'N':-1.18,
    'M':-0.24,'D':-1.02,'F':0.0,'C':0.0,'P':0.0,
    'Q':-1.53,'S':-0.75,'E':-0.90,'T':-0.71,'G':-0.34,
    'W':-0.59,'H':-0.94,'Y':-1.02,'I':0.24,'V':0.09}

    RADA880108={'A':-0.06,'L':1.21,'R':-0.84,'K':-1.18,'N':-0.48,
    'M':1.27,'D':-0.80,'F':1.27,'C':1.36,'P':0.0,
    'Q':-0.73,'S':-0.50,'E':-0.77,'T':-0.27,'G':-0.41,
    'W':0.88,'H':0.49,'Y':0.33,'I':1.31,'V':1.09}

    RICJ880101={'A':0.7,'L':0.9,'R':0.4,'K':1.0,'N':1.2,
    'M':0.3,'D':1.4,'F':1.2,'C':0.6,'P':0.7,
    'Q':1.0,'S':1.6,'E':1.0,'T':0.3,'G':1.6,
    'W':1.1,'H':1.2,'Y':1.9,'I':0.9,'V':0.7}

    RICJ880102={'A':0.7,'L':0.9,'R':0.4,'K':1.0,'N':1.2,
    'M':0.3,'D':1.4,'F':1.2,'C':0.6,'P':0.7,
    'Q':1.0,'S':1.6,'E':1.0,'T':0.3,'G':1.6,
    'W':1.1,'H':1.2,'Y':1.9,'I':0.9,'V':0.7}

    RICJ880103={'A':0.5,'L':0.2,'R':0.4,'K':0.7,'N':3.5,
    'M':0.8,'D':2.1,'F':0.2,'C':0.6,'P':0.8,
    'Q':0.4,'S':2.3,'E':0.4,'T':1.6,'G':1.8,
    'W':0.3,'H':1.1,'Y':0.8,'I':0.2,'V':0.1}

    RICJ880104={'A':1.2,'L':0.9,'R':0.7,'K':0.6,'N':0.7,
    'M':0.3,'D':0.8,'F':0.5,'C':0.8,'P':2.6,
    'Q':0.7,'S':0.7,'E':2.2,'T':0.8,'G':0.3,
    'W':2.1,'H':0.7,'Y':1.8,'I':0.9,'V':1.1}

    RICJ880105={'A':1.6,'L':0.3,'R':0.9,'K':1.0,'N':0.7,
    'M':1.0,'D':2.6,'F':0.9,'C':1.2,'P':0.5,
    'Q':0.8,'S':0.8,'E':2.0,'T':0.7,'G':0.9,
    'W':1.7,'H':0.7,'Y':0.4,'I':0.7,'V':0.6}

    RICJ880106={'A':1.0,'L':0.6,'R':0.4,'K':0.8,'N':0.7,
    'M':1.0,'D':2.2,'F':0.6,'C':0.6,'P':0.4,
    'Q':1.5,'S':0.4,'E':3.3,'T':1.0,'G':0.6,
    'W':1.4,'H':0.7,'Y':1.2,'I':0.4,'V':1.1}

    RICJ880107={'A':1.1,'L':2.6,'R':1.5,'K':0.8,'N':0.0,
    'M':1.7,'D':0.3,'F':1.9,'C':1.1,'P':0.1,
    'Q':1.3,'S':0.4,'E':0.5,'T':0.5,'G':0.4,
    'W':3.1,'H':1.5,'Y':0.6,'I':1.1,'V':1.5}

    RICJ880108={'A':1.4,'L':1.1,'R':1.2,'K':1.9,'N':1.2,
    'M':1.7,'D':0.6,'F':1.0,'C':1.6,'P':0.3,
    'Q':1.4,'S':1.1,'E':0.9,'T':0.6,'G':0.6,
    'W':1.4,'H':0.9,'Y':0.2,'I':0.9,'V':0.8}

    RICJ880109={'A':1.8,'L':1.2,'R':1.3,'K':1.1,'N':0.9,
    'M':1.5,'D':1.0,'F':1.3,'C':0.7,'P':0.3,
    'Q':1.3,'S':0.6,'E':0.8,'T':1.,'G':0.5,
    'W':1.5,'H':1.0,'Y':0.8,'I':1.2,'V':1.2}

    RICJ880110={'A':1.8,'L':1.2,'R':1.0,'K':1.4,'N':0.6,
    'M':2.7,'D':0.7,'F':1.9,'C':0.0,'P':0.3,
    'Q':1.0,'S':0.5,'E':1.1,'T':0.5,'G':0.5,
    'W':1.1,'H':2.4,'Y':1.3,'I':1.3,'V':0.4}

    RICJ880111={'A':1.3,'L':1.4,'R':0.8,'K':1.0,'N':0.6,
    'M':2.8,'D':0.5,'F':2.9,'C':0.7,'P':0.0,
    'Q':0.2,'S':0.5,'E':0.7,'T':0.6,'G':0.5,
    'W':2.1,'H':1.9,'Y':0.8,'I':1.6,'V':1.4}

    RICJ880112={'A':0.7,'L':1.9,'R':0.8,'K':2.2,'N':0.8,
    'M':1.0,'D':0.6,'F':1.8,'C':0.2,'P':0.0,
    'Q':1.3,'S':0.6,'E':1.6,'T':0.7,'G':0.1,
    'W':0.4,'H':1.1,'Y':1.1,'I':1.4,'V':1.3}

    RICJ880113={'A':1.4,'L':0.8,'R':2.1,'K':1.9,'N':0.9,
    'M':1.3,'D':0.7,'F':0.3,'C':1.2,'P':0.2,
    'Q':1.6,'S':1.6,'E':1.7,'T':0.9,'G':0.2,
    'W':0.4,'H':1.8,'Y':0.3,'I':0.4,'V':0.7}

    RICJ880114={'A':1.1,'L':0.7,'R':1.0,'K':2.0,'N':1.2,
    'M':1.0,'D':0.4,'F':0.7,'C':1.6,'P':0.0,
    'Q':2.1,'S':1.7,'E':0.8,'T':1.0,'G':0.2,
    'W':0.0,'H':3.4,'Y':1.2,'I':0.7,'V':0.7}

    RICJ880115={'A':0.8,'L':0.7,'R':0.9,'K':1.3,'N':1.6,
    'M':0.8,'D':0.7,'F':0.5,'C':0.4,'P':0.7,
    'Q':0.9,'S':0.8,'E':0.3,'T':0.3,'G':3.9,
    'W':0.0,'H':1.3,'Y':0.8,'I':0.7,'V':0.2}

    RICJ880116={'A':1.0,'L':0.9,'R':1.4,'K':1.2,'N':0.9,
    'M':0.8,'D':1.4,'F':0.1,'C':0.8,'P':1.9,
    'Q':1.4,'S':0.7,'E':0.8,'T':0.8,'G':1.2,
    'W':0.4,'H':1.2,'Y':0.9,'I':1.1,'V':0.6}

    RICJ880117={'A':0.7,'L':0.5,'R':1.1,'K':1.3,'N':1.5,
    'M':0.0,'D':1.4,'F':1.2,'C':0.4,'P':1.5,
    'Q':1.1,'S':0.9,'E':0.7,'T':2.1,'G':0.6,
    'W':2.7,'H':1.0,'Y':0.5,'I':0.7,'V':1.0}

    ROBB760101={'A':6.5,'L':3.2,'R':-0.9,'K':2.3,'N':-5.1,
    'M':5.3,'D':0.5,'F':1.6,'C':-1.3,'P':-7.7,
    'Q':1.0,'S':-3.9,'E':7.8,'T':-2.6,'G':-8.6,
    'W':1.2,'H':1.2,'Y':-4.5,'I':0.6,'V':1.4}

    ROBB760102={'A':2.3,'L':-2.1,'R':-5.2,'K':-4.1,'N':0.3,
    'M':-3.5,'D':7.4,'F':-1.1,'C':0.8,'P':8.1,
    'Q':-0.7,'S':-3.5,'E':10.3,'T':2.3,'G':-5.2,
    'W':-0.9,'H':-2.8,'Y':-3.7,'I':-4.0,'V':-4.4}

    ROBB760103={'A':6.7,'L':5.5,'R':0.3,'K':0.5,'N':-6.1,
    'M':7.2,'D':-3.1,'F':2.8,'C':-4.9,'P':-22.8,
    'Q':0.6,'S':-3.0,'E':2.2,'T':-4.0,'G':-6.8,
    'W':4.0,'H':-1.0,'Y':-4.6,'I':3.2,'V':2.5}

    ROBB760104={'A':2.3,'L':0.1,'R':1.4,'K':7.3,'N':-3.3,
    'M':3.5,'D':-4.4,'F':1.6,'C':6.1,'P':-24.4,
    'Q':2.7,'S':-1.9,'E':2.5,'T':-3.7,'G':-8.3,
    'W':-0.9,'H':5.9,'Y':-0.6,'I':-0.5,'V':2.3}

    ROBB760105={'A':-2.3,'L':2.3,'R':0.4,'K':-3.3,'N':-4.1,
    'M':2.3,'D':-4.4,'F':2.6,'C':4.4,'P':-1.8,
    'Q':1.2,'S':-1.7,'E':-5.0,'T':1.3,'G':-4.2,
    'W':-1.0,'H':-2.5,'Y':4.0,'I':6.7,'V':6.8}

    ROBB760106={'A':-2.7,'L':3.7,'R':0.4,'K':-2.9,'N':-4.2,
    'M':3.7,'D':-4.4,'F':3.0,'C':3.7,'P':-6.6,
    'Q':0.8,'S':-2.4,'E':-8.1,'T':1.7,'G':-3.9,
    'W':0.3,'H':-3.0,'Y':3.3,'I':7.7,'V':7.1}

    ROBB760107={'A':0.0,'L':-3.7,'R':1.1,'K':-3.1,'N':-2.0,
    'M':-2.1,'D':-2.6,'F':0.7,'C':5.4,'P':7.4,
    'Q':2.4,'S':1.3,'E':3.1,'T':0.0,'G':-3.4,
    'W':-3.4,'H':0.8,'Y':4.8,'I':-0.1,'V':2.7}

    ROBB760108={'A':-5.0,'L':-5.6,'R':2.1,'K':1.0,'N':4.2,
    'M':-4.8,'D':3.1,'F':-1.8,'C':4.4,'P':2.6,
    'Q':0.4,'S':2.6,'E':-4.7,'T':0.3,'G':5.7,
    'W':3.4,'H':-0.3,'Y':2.9,'I':-4.6,'V':-6.0}

    ROBB760109={'A':-3.3,'L':-2.3,'R':0.0,'K':-1.2,'N':5.4,
    'M':-4.3,'D':3.9,'F':0.8,'C':-0.3,'P':6.5,
    'Q':-0.4,'S':1.8,'E':-1.8,'T':-0.7,'G':-1.2,
    'W':-0.8,'H':3.0,'Y':3.1,'I':-0.5,'V':-3.5}

    ROBB760110={'A':-4.7,'L':-6.2,'R':2.0,'K':2.8,'N':3.9,
    'M':-4.8,'D':1.9,'F':-3.7,'C':6.2,'P':3.6,
    'Q':-2.0,'S':2.1,'E':-4.2,'T':0.6,'G':5.7,
    'W':3.3,'H':-2.6,'Y':3.8,'I':-7.0,'V':-6.2}

    ROBB760111={'A':-3.7,'L':-2.8,'R':1.0,'K':1.3,'N':-0.6,
    'M':-1.6,'D':-0.6,'F':1.6,'C':4.0,'P':-6.0,
    'Q':3.4,'S':1.5,'E':-4.3,'T':1.2,'G':5.9,
    'W':6.5,'H':-0.8,'Y':1.3,'I':-0.5,'V':-4.6}

    ROBB760112={'A':-2.5,'L':-2.0,'R':-1.2,'K':-0.8,'N':4.6,
    'M':-4.1,'D':0.0,'F':-4.1,'C':-4.7,'P':5.8,
    'Q':-0.5,'S':2.5,'E':-4.4,'T':1.7,'G':4.9,
    'W':1.2,'H':1.6,'Y':-0.6,'I':-3.3,'V':-3.5}

    ROBB760113={'A':-5.1,'L':-5.4,'R':2.6,'K':1.0,'N':4.7,
    'M':-5.3,'D':3.1,'F':-2.4,'C':3.8,'P':3.5,
    'Q':0.2,'S':3.2,'E':-5.2,'T':0.0,'G':5.6,
    'W':2.9,'H':-0.9,'Y':3.2,'I':-4.5,'V':-6.3}

    ROBB790101={'A':-1.0,'L':2.0,'R':0.3,'K':-0.9,'N':-0.7,
    'M':1.8,'D':-1.2,'F':2.8,'C':2.1,'P':0.4,
    'Q':-0.1,'S':-1.2,'E':-0.7,'T':-0.5,'G':0.3,
    'W':3.0,'H':1.1,'Y':2.1,'I':4.0,'V':1.4}

    ROSG850101={'A':86.6,'L':164.1,'R':162.2,'K':115.5,'N':103.3,
    'M':172.9,'D':97.8,'F':194.1,'C':132.3,'P':92.9,
    'Q':119.2,'S':85.6,'E':113.9,'T':106.5,'G':62.9,
    'W':224.6,'H':155.8,'Y':177.7,'I':158.0,'V':141.0}

    ROSG850102={'A':0.74,'L':0.85,'R':0.64,'K':0.52,'N':0.63,
    'M':0.85,'D':0.62,'F':0.88,'C':0.91,'P':0.64,
    'Q':0.62,'S':0.66,'E':0.62,'T':0.70,'G':0.72,
    'W':0.85,'H':0.78,'Y':0.76,'I':0.88,'V':0.86}

    ROSM880101={'A':-0.67,'L':-3.02,'R':12.1,'K':6.13,'N':7.23,
    'M':-1.30,'D':8.72,'F':-3.24,'C':-0.34,'P':-1.75,
    'Q':6.39,'S':4.35,'E':7.35,'T':3.86,'G':0.00,
    'W':-2.86,'H':3.82,'Y':0.98,'I':-3.02,'V':-2.18}

    ROSM880102={'A':-0.67,'L':-3.02,'R':3.89,'K':2.46,'N':2.27,
    'M':-1.67,'D':1.57,'F':-3.24,'C':-2.00,'P':-1.75,
    'Q':2.12,'S':0.10,'E':1.78,'T':-0.42,'G':0.00,
    'W':-2.86,'H':1.09,'Y':0.98,'I':-3.02,'V':-2.18}

    ROSM880103={'A':0.4,'L':0.6,'R':0.3,'K':0.4,'N':0.9,
    'M':0.3,'D':0.8,'F':0.7,'C':0.5,'P':0.9,
    'Q':0.7,'S':0.4,'E':1.3,'T':0.4,'G':0.0,
    'W':0.6,'H':1.0,'Y':1.2,'I':0.4,'V':0.4}

    SIMZ760101={'A':0.73,'L':2.49,'R':0.73,'K':1.50,'N':-0.01,
    'M':1.30,'D':0.54,'F':2.65,'C':0.70,'P':2.60,
    'Q':-0.10,'S':0.04,'E':0.55,'T':0.44,'G':0.00,
    'W':3.00,'H':1.10,'Y':2.97,'I':2.97,'V':1.69}

    SNEP660101={'A':0.239,'L':0.281,'R':0.211,'K':0.228,'N':0.249,
    'M':0.253,'D':0.171,'F':0.234,'C':0.220,'P':0.165,
    'Q':0.260,'S':0.236,'E':0.187,'T':0.213,'G':0.160,
    'W':0.183,'H':0.205,'Y':0.193,'I':0.273,'V':0.255}

    SNEP660102={'A':0.330,'L':0.129,'R':-0.176,'K':-0.075,'N':-0.233,
    'M':-0.092,'D':-0.371,'F':-0.011,'C':0.074,'P':0.370,
    'Q':-0.254,'S':0.022,'E':-0.409,'T':0.136,'G':0.370,
    'W':-0.011,'H':-0.078,'Y':-0.138,'I':0.149,'V':0.245}

    SNEP660103={'A':-0.110,'L':-0.008,'R':0.079,'K':0.049,'N':-0.136,
    'M':-0.041,'D':-0.285,'F':0.438,'C':-0.184,'P':-0.016,
    'Q':-0.067,'S':-0.153,'E':-0.246,'T':-0.208,'G':-0.073,
    'W':0.493,'H':0.320,'Y':0.381,'I':0.001,'V':-0.155}

    SNEP660104={'A':-0.062,'L':-0.264,'R':-0.167,'K':-0.371,'N':0.166,
    'M':0.077,'D':-0.079,'F':0.074,'C':0.380,'P':-0.036,
    'Q':-0.025,'S':0.470,'E':-0.184,'T':0.348,'G':-0.017,
    'W':0.050,'H':0.056,'Y':0.220,'I':-0.309,'V':-0.212}

    SUEM840101={'A':1.071,'L':1.140,'R':1.033,'K':0.939,'N':0.784,
    'M':1.200,'D':0.680,'F':1.086,'C':0.922,'P':0.659,
    'Q':0.977,'S':0.760,'E':0.970,'T':0.817,'G':0.591,
    'W':1.107,'H':0.850,'Y':1.020,'I':1.140,'V':0.950}

    SUEM840102={'A':8.0,'L':33.0,'R':0.1,'K':1.0,'N':0.1,
    'M':54.0,'D':70.0,'F':18.0,'C':26.0,'P':42.0,
    'Q':33.0,'S':0.1,'E':6.0,'T':0.1,'G':0.1,
    'W':77.0,'H':0.1,'Y':66.0,'I':55.0,'V':0.1}

    SWER830101={'A':-0.40,'L':1.22,'R':-0.59,'K':-0.67,'N':-0.92,
    'M':1.02,'D':-1.31,'F':1.92,'C':0.17,'P':-0.49,
    'Q':-0.91,'S':-0.55,'E':-1.22,'T':-0.28,'G':-0.67,
    'W':0.50,'H':-0.64,'Y':1.67,'I':1.25,'V':0.91}

    TANS770101={'A':1.42,'L':1.29,'R':1.06,'K':1.24,'N':0.71,
    'M':1.21,'D':1.01,'F':1.16,'C':0.73,'P':0.65,
    'Q':1.02,'S':0.71,'E':1.63,'T':0.78,'G':0.50,
    'W':1.05,'H':1.20,'Y':0.67,'I':1.12,'V':0.99}

    TANS770102={'A':0.946,'L':1.192,'R':1.128,'K':1.203,'N':0.432,
    'M':0.000,'D':1.311,'F':0.963,'C':0.481,'P':2.093,
    'Q':1.615,'S':0.523,'E':0.698,'T':1.961,'G':0.360,
    'W':1.925,'H':2.168,'Y':0.802,'I':1.283,'V':0.409}

    TANS770103={'A':0.790,'L':1.111,'R':1.087,'K':0.735,'N':0.832,
    'M':1.092,'D':0.530,'F':1.052,'C':1.268,'P':1.249,
    'Q':1.038,'S':1.093,'E':0.643,'T':1.214,'G':0.725,
    'W':1.114,'H':0.864,'Y':1.340,'I':1.361,'V':1.428}

    TANS770104={'A':1.194,'L':0.595,'R':0.795,'K':1.060,'N':0.659,
    'M':0.831,'D':1.056,'F':0.377,'C':0.678,'P':3.159,
    'Q':1.290,'S':1.444,'E':0.928,'T':1.172,'G':1.015,
    'W':0.452,'H':0.611,'Y':0.816,'I':0.603,'V':0.640}

    TANS770105={'A':0.497,'L':0.656,'R':0.677,'K':0.932,'N':2.072,
    'M':0.425,'D':1.498,'F':1.348,'C':1.348,'P':0.179,
    'Q':0.711,'S':1.151,'E':0.651,'T':0.749,'G':1.848,
    'W':1.283,'H':1.474,'Y':1.283,'I':0.471,'V':0.654}

    TANS770106={'A':0.937,'L':0.808,'R':1.725,'K':1.254,'N':1.080,
    'M':0.886,'D':1.640,'F':0.803,'C':1.004,'P':0.748,
    'Q':1.078,'S':1.145,'E':0.679,'T':1.487,'G':0.901,
    'W':0.803,'H':1.085,'Y':1.227,'I':0.178,'V':0.625}

    TANS770107={'A':0.289,'L':0.000,'R':1.380,'K':1.288,'N':3.169,
    'M':0.000,'D':0.917,'F':0.393,'C':1.767,'P':0.000,
    'Q':2.372,'S':0.160,'E':0.285,'T':0.218,'G':4.259,
    'W':0.000,'H':1.061,'Y':0.654,'I':0.262,'V':0.167}

    TANS770108={'A':0.328,'L':0.414,'R':2.088,'K':0.835,'N':1.498,
    'M':0.982,'D':3.379,'F':1.336,'C':0.000,'P':0.415,
    'Q':0.000,'S':1.089,'E':0.000,'T':1.732,'G':0.500,
    'W':1.781,'H':1.204,'Y':0.000,'I':2.078,'V':0.946}

    TANS770109={'A':0.945,'L':0.758,'R':0.364,'K':0.947,'N':1.202,
    'M':1.028,'D':1.315,'F':0.622,'C':0.932,'P':0.579,
    'Q':0.704,'S':1.140,'E':1.014,'T':0.863,'G':2.355,
    'W':0.777,'H':0.525,'Y':0.907,'I':0.673,'V':0.561}

    TANS770110={'A':0.842,'L':0.665,'R':0.936,'K':1.045,'N':1.352,
    'M':0.668,'D':1.366,'F':0.881,'C':1.032,'P':1.385,
    'Q':0.998,'S':1.257,'E':0.758,'T':1.055,'G':1.349,
    'W':0.881,'H':1.079,'Y':1.101,'I':0.459,'V':0.643}

    VASM830101={'A':0.135,'L':0.215,'R':0.296,'K':0.170,'N':0.196,
    'M':0.239,'D':0.289,'F':0.087,'C':0.159,'P':0.151,
    'Q':0.236,'S':0.010,'E':0.184,'T':0.100,'G':0.051,
    'W':0.166,'H':0.223,'Y':0.066,'I':0.173,'V':0.285}

    VASM830102={'A':0.507,'L':0.619,'R':0.459,'K':0.559,'N':0.287,
    'M':0.431,'D':0.223,'F':0.077,'C':0.592,'P':0.739,
    'Q':0.383,'S':0.689,'E':0.445,'T':0.785,'G':0.390,
    'W':0.160,'H':0.310,'Y':0.060,'I':0.111,'V':0.356}

    VASM830103={'A':0.159,'L':0.083,'R':0.194,'K':0.159,'N':0.385,
    'M':0.198,'D':0.283,'F':0.682,'C':0.187,'P':0.366,
    'Q':0.236,'S':0.150,'E':0.206,'T':0.074,'G':0.049,
    'W':0.463,'H':0.233,'Y':0.737,'I':0.581,'V':0.301}

    VELV850101={'A':0.03731,'L':0.00000,'R':0.09593,'K':0.03710,'N':0.00359,
    'M':0.08226,'D':0.12630,'F':0.09460,'C':0.08292,'P':0.01979,
    'Q':0.07606,'S':0.08292,'E':0.00580,'T':0.09408,'G':0.00499,
    'W':0.05481,'H':0.02415,'Y':0.05159,'I':0.00000,'V':0.00569}

    VENT840101={'A':0.0,'L':1.0,'R':0.0,'K':0.0,'N':0.0,
    'M':0.0,'D':0.0,'F':1.0,'C':0.0,'P':0.0,
    'Q':0.0,'S':0.0,'E':0.0,'T':0.0,'G':0.0,
    'W':1.0,'H':0.0,'Y':1.0,'I':1.0,'V':1.0}

    VHEG790101={'A':-12.04,'L':-17.79,'R':39.23,'K':9.71,'N':4.25,
    'M':-8.86,'D':23.22,'F':-21.98,'C':3.95,'P':5.82,
    'Q':2.16,'S':-1.54,'E':16.81,'T':-4.15,'G':-7.85,
    'W':-16.19,'H':6.28,'Y':-1.51,'I':-18.32,'V':-16.22}

    WARP780101={'A':10.04,'L':8.79,'R':6.18,'K':4.40,'N':5.63,
    'M':9.15,'D':5.76,'F':7.98,'C':8.89,'P':7.79,
    'Q':5.41,'S':7.08,'E':5.37,'T':7.00,'G':7.99,
    'W':8.07,'H':7.49,'Y':6.90,'I':8.72,'V':8.88}

    WEBA780101={'A':0.89,'L':0.73,'R':0.88,'K':0.97,'N':0.89,
    'M':0.74,'D':0.87,'F':0.52,'C':0.85,'P':0.82,
    'Q':0.82,'S':0.96,'E':0.84,'T':0.92,'G':0.92,
    'W':0.20,'H':0.83,'Y':0.49,'I':0.76,'V':0.85}

    WERD780101={'A':0.52,'L':0.77,'R':0.49,'K':0.31,'N':0.42,
    'M':0.76,'D':0.37,'F':0.87,'C':0.83,'P':0.35,
    'Q':0.35,'S':0.49,'E':0.38,'T':0.38,'G':0.41,
    'W':0.86,'H':0.70,'Y':0.64,'I':0.79,'V':0.72}

    WERD780102={'A':0.16,'L':-0.44,'R':-0.20,'K':-0.12,'N':1.03,
    'M':-0.79,'D':-0.24,'F':-0.25,'C':-0.12,'P':-0.59,
    'Q':-0.55,'S':-0.01,'E':-0.45,'T':0.05,'G':-0.16,
    'W':-0.33,'H':-0.18,'Y':-0.42,'I':-0.19,'V':-0.46}

    WERD780103={'A':0.15,'L':0.06,'R':-0.37,'K':-0.16,'N':0.69,
    'M':0.11,'D':-0.22,'F':1.18,'C':-0.19,'P':0.11,
    'Q':-0.06,'S':0.13,'E':0.14,'T':0.28,'G':0.36,
    'W':-0.12,'H':-0.25,'Y':0.19,'I':0.02,'V':-0.08}

    WERD780104={'A':-0.07,'L':-0.17,'R':-0.40,'K':-0.45,'N':-0.57,
    'M':0.03,'D':-0.80,'F':0.40,'C':0.17,'P':-0.47,
    'Q':-0.26,'S':-0.11,'E':-0.63,'T':0.09,'G':0.27,
    'W':-0.61,'H':-0.49,'Y':-0.61,'I':0.06,'V':-0.11}

    WOEC730101={'A':7.0,'L':4.9,'R':9.1,'K':10.1,'N':10.0,
    'M':5.3,'D':13.0,'F':5.0,'C':5.5,'P':6.6,
    'Q':8.6,'S':7.5,'E':12.5,'T':6.6,'G':7.9,
    'W':5.3,'H':8.4,'Y':5.7,'I':4.9,'V':5.6}

    WOLR810101={'A':1.94,'L':2.28,'R':-19.92,'K':-9.52,'N':-9.68,
    'M':-1.48,'D':-10.95,'F':-0.76,'C':-1.24,'P':-3.68,
    'Q':-9.38,'S':-5.06,'E':-10.20,'T':-4.88,'G':2.39,
    'W':-5.88,'H':-10.27,'Y':-6.11,'I':2.15,'V':1.99}

    WOLS870101={'A':0.07,'L':-4.19,'R':2.88,'K':2.84,'N':3.22,
    'M':-2.49,'D':3.64,'F':-4.92,'C':0.71,'P':-1.22,
    'Q':2.18,'S':1.96,'E':3.08,'T':0.92,'G':2.23,
    'W':-4.75,'H':2.41,'Y':-1.39,'I':-4.44,'V':-2.69}

    WOLS870102={'A':-1.73,'L':-1.03,'R':2.52,'K':1.41,'N':1.45,
    'M':-0.27,'D':1.13,'F':1.30,'C':-0.97,'P':0.88,
    'Q':0.53,'S':-1.63,'E':0.39,'T':-2.09,'G':-5.36,
    'W':3.65,'H':1.74,'Y':2.32,'I':-1.68,'V':-2.53}

    WOLS870103={'A':0.09,'L':-0.98,'R':-3.44,'K':-3.14,'N':0.84,
    'M':-0.41,'D':2.36,'F':0.45,'C':4.13,'P':2.23,
    'Q':-1.14,'S':0.57,'E':-0.07,'T':-1.40,'G':0.30,
    'W':0.85,'H':1.11,'Y':0.01,'I':-1.03,'V':-1.29}

    YUTK870101={'A':8.5,'L':15.0,'R':0.0,'K':7.9,'N':8.2,
    'M':13.3,'D':8.5,'F':11.2,'C':11.0,'P':8.2,
    'Q':6.3,'S':7.4,'E':8.8,'T':8.8,'G':7.1,
    'W':9.9,'H':10.1,'Y':8.8,'I':16.8,'V':12.0}

    YUTK870102={'A':6.8,'L':12.2,'R':0.0,'K':7.5,'N':6.2,
    'M':8.4,'D':7.0,'F':8.3,'C':8.3,'P':6.9,
    'Q':8.5,'S':8.0,'E':4.9,'T':7.0,'G':6.4,
    'W':5.7,'H':9.2,'Y':6.8,'I':10.0,'V':9.4}

    YUTK870103={'A':18.08,'L':18.60,'R':0.0,'K':17.96,'N':17.47,
    'M':18.11,'D':17.36,'F':17.30,'C':18.17,'P':18.16,
    'Q':17.93,'S':17.57,'E':18.16,'T':17.54,'G':18.24,
    'W':17.19,'H':18.49,'Y':17.99,'I':18.62,'V':18.30}

    YUTK870104={'A':18.56,'L':19.01,'R':0.0,'K':18.36,'N':18.24,
    'M':18.49,'D':17.94,'F':17.95,'C':17.84,'P':18.77,
    'Q':18.51,'S':18.06,'E':17.97,'T':17.71,'G':18.57,
    'W':16.87,'H':18.64,'Y':18.23,'I':19.21,'V':18.98}

    ZASB820101={'A':-0.152,'L':-0.102,'R':-0.089,'K':-0.062,'N':-0.203,
    'M':-0.107,'D':-0.355,'F':0.001,'C':0.0,'P':-0.181,
    'Q':-0.181,'S':-0.203,'E':-0.411,'T':-0.170,'G':-0.190,
    'W':0.275,'H':0.0,'Y':0.0,'I':-0.086,'V':-0.125}

    ZIMJ680101={'A':0.83,'L':2.52,'R':0.83,'K':1.60,'N':0.09,
    'M':1.40,'D':0.64,'F':2.75,'C':1.48,'P':2.70,
    'Q':0.00,'S':0.14,'E':0.65,'T':0.54,'G':0.10,
    'W':0.31,'H':1.10,'Y':2.97,'I':3.07,'V':1.79}

    ZIMJ680102={'A':11.50,'L':21.40,'R':14.28,'K':15.71,'N':12.82,
    'M':16.25,'D':11.68,'F':19.80,'C':13.46,'P':17.43,
    'Q':14.45,'S':9.47,'E':13.57,'T':15.77,'G':3.40,
    'W':21.67,'H':13.69,'Y':18.03,'I':21.40,'V':21.57}

    ZIMJ680103={'A':0.00,'L':0.13,'R':52.00,'K':49.50,'N':3.38,
    'M':1.43,'D':49.70,'F':0.35,'C':1.48,'P':1.58,
    'Q':3.53,'S':1.67,'E':49.90,'T':1.66,'G':0.00,
    'W':2.10,'H':51.60,'Y':1.61,'I':0.13,'V':0.13}

    ZIMJ680104={'A':6.00,'L':5.98,'R':10.76,'K':9.74,'N':5.41,
    'M':5.74,'D':2.77,'F':5.48,'C':5.05,'P':6.30,
    'Q':5.65,'S':5.68,'E':3.22,'T':5.66,'G':5.97,
    'W':5.89,'H':7.59,'Y':5.66,'I':6.02,'V':5.96}

    ZIMJ680105={'A':9.9,'L':17.6,'R':4.6,'K':3.5,'N':5.4,
    'M':14.9,'D':2.8,'F':18.8,'C':2.8,'P':14.8,
    'Q':9.0,'S':6.9,'E':3.2,'T':9.5,'G':5.6,
    'W':17.1,'H':8.2,'Y':15.0,'I':17.1,'V':14.3}

    AURR980101={'A':0.94,'L':0.95,'R':1.15,'K':1.03,'N':0.79,
    'M':0.88,'D':1.19,'F':1.06,'C':0.60,'P':1.18,
    'Q':0.94,'S':0.69,'E':1.41,'T':0.87,'G':1.18,
    'W':0.91,'H':1.15,'Y':1.04,'I':1.07,'V':0.90}

    AURR980102={'A':0.98,'L':0.80,'R':1.14,'K':1.06,'N':1.05,
    'M':1.12,'D':1.05,'F':1.12,'C':0.41,'P':1.31,
    'Q':0.90,'S':1.02,'E':1.04,'T':0.80,'G':1.25,
    'W':0.90,'H':1.01,'Y':1.12,'I':0.88,'V':0.87}

    AURR980103={'A':1.05,'L':0.96,'R':0.81,'K':0.97,'N':0.91,
    'M':0.99,'D':1.39,'F':0.95,'C':0.60,'P':1.05,
    'Q':0.87,'S':0.96,'E':1.11,'T':1.03,'G':1.26,
    'W':1.06,'H':1.43,'Y':0.94,'I':0.95,'V':0.62}

    AURR980104={'A':0.75,'L':1.01,'R':0.90,'K':0.66,'N':1.24,
    'M':1.02,'D':1.72,'F':0.88,'C':0.66,'P':1.33,
    'Q':1.08,'S':1.20,'E':1.10,'T':1.13,'G':1.14,
    'W':0.68,'H':0.96,'Y':0.80,'I':0.80,'V':0.58}

    AURR980105={'A':0.67,'L':0.79,'R':0.76,'K':0.84,'N':1.28,
    'M':0.98,'D':1.58,'F':0.96,'C':0.37,'P':1.12,
    'Q':1.05,'S':1.25,'E':0.94,'T':1.41,'G':0.98,
    'W':0.94,'H':0.83,'Y':0.82,'I':0.78,'V':0.67}

    AURR980106={'A':1.10,'L':0.84,'R':1.05,'K':1.08,'N':0.72,
    'M':0.90,'D':1.14,'F':0.90,'C':0.26,'P':1.67,
    'Q':1.31,'S':0.81,'E':2.30,'T':0.77,'G':0.55,
    'W':1.26,'H':0.83,'Y':0.99,'I':1.06,'V':0.76}

    AURR980107={'A':1.39,'L':0.91,'R':0.95,'K':0.80,'N':0.67,
    'M':1.10,'D':1.64,'F':1.00,'C':0.52,'P':0.94,
    'Q':1.60,'S':0.69,'E':2.07,'T':0.92,'G':0.65,
    'W':1.10,'H':1.36,'Y':0.73,'I':0.64,'V':0.70}

    AURR980108={'A':1.43,'L':1.52,'R':1.33,'K':0.82,'N':0.55,
    'M':1.68,'D':0.90,'F':1.10,'C':0.52,'P':0.15,
    'Q':1.43,'S':0.61,'E':1.70,'T':0.75,'G':0.56,
    'W':1.68,'H':0.66,'Y':0.65,'I':1.18,'V':1.14}

    AURR980109={'A':1.55,'L':1.36,'R':1.39,'K':1.27,'N':0.60,
    'M':2.13,'D':0.61,'F':1.39,'C':0.59,'P':0.03,
    'Q':1.43,'S':0.44,'E':1.34,'T':0.65,'G':0.37,
    'W':1.10,'H':0.89,'Y':0.93,'I':1.47,'V':1.18}

    AURR980110={'A':1.80,'L':1.47,'R':1.73,'K':1.24,'N':0.73,
    'M':1.64,'D':0.90,'F':0.96,'C':0.55,'P':0.15,
    'Q':0.97,'S':0.67,'E':1.73,'T':0.70,'G':0.32,
    'W':0.68,'H':0.46,'Y':0.91,'I':1.09,'V':0.81}

    AURR980111={'A':1.52,'L':1.26,'R':1.49,'K':1.10,'N':0.58,
    'M':1.14,'D':1.04,'F':1.14,'C':0.26,'P':0.44,
    'Q':1.41,'S':0.66,'E':1.76,'T':0.73,'G':0.30,
    'W':0.68,'H':0.83,'Y':1.04,'I':1.25,'V':1.03}

    AURR980112={'A':1.49,'L':1.40,'R':1.41,'K':1.17,'N':0.67,
    'M':1.84,'D':0.94,'F':0.86,'C':0.37,'P':0.20,
    'Q':1.52,'S':0.68,'E':1.55,'T':0.79,'G':0.29,
    'W':1.52,'H':0.96,'Y':1.06,'I':1.04,'V':0.94}

    AURR980113={'A':1.73,'L':1.80,'R':1.24,'K':1.22,'N':0.70,
    'M':2.21,'D':0.68,'F':1.35,'C':0.63,'P':0.07,
    'Q':0.88,'S':0.65,'E':1.16,'T':0.46,'G':0.32,
    'W':1.57,'H':0.76,'Y':1.10,'I':1.15,'V':0.94}

    AURR980114={'A':1.33,'L':1.63,'R':1.39,'K':1.71,'N':0.64,
    'M':1.76,'D':0.60,'F':1.22,'C':0.44,'P':0.07,
    'Q':1.37,'S':0.42,'E':1.43,'T':0.57,'G':0.20,
    'W':1.00,'H':1.02,'Y':1.02,'I':1.58,'V':1.08}

    AURR980115={'A':1.87,'L':1.65,'R':1.66,'K':1.63,'N':0.70,
    'M':1.35,'D':0.91,'F':0.67,'C':0.33,'P':0.03,
    'Q':1.24,'S':0.71,'E':1.88,'T':0.50,'G':0.33,
    'W':1.00,'H':0.89,'Y':0.73,'I':0.90,'V':0.51}

    AURR980116={'A':1.19,'L':1.36,'R':1.45,'K':1.45,'N':1.33,
    'M':1.35,'D':0.72,'F':1.20,'C':0.44,'P':0.10,
    'Q':1.43,'S':1.02,'E':1.27,'T':0.82,'G':0.74,
    'W':0.58,'H':1.55,'Y':1.06,'I':0.61,'V':0.46}

    AURR980117={'A':0.77,'L':0.66,'R':1.11,'K':1.19,'N':1.39,
    'M':0.74,'D':0.79,'F':1.04,'C':0.44,'P':0.66,
    'Q':0.95,'S':0.64,'E':0.92,'T':0.82,'G':2.74,
    'W':0.58,'H':1.65,'Y':0.93,'I':0.64,'V':0.53}

    AURR980118={'A':0.93,'L':1.16,'R':0.96,'K':1.27,'N':0.82,
    'M':1.11,'D':1.15,'F':1.05,'C':0.67,'P':1.01,
    'Q':1.02,'S':0.71,'E':1.07,'T':0.84,'G':1.08,
    'W':1.06,'H':1.40,'Y':1.15,'I':1.14,'V':0.74}

    AURR980119={'A':1.09,'L':0.87,'R':1.29,'K':1.13,'N':1.03,
    'M':0.96,'D':1.17,'F':0.84,'C':0.26,'P':2.01,
    'Q':1.08,'S':0.76,'E':1.31,'T':0.79,'G':0.97,
    'W':0.91,'H':0.88,'Y':0.64,'I':0.97,'V':0.77}

    AURR980120={'A':0.71,'L':0.84,'R':1.09,'K':1.10,'N':0.95,
    'M':0.80,'D':1.43,'F':0.95,'C':0.65,'P':1.70,
    'Q':0.87,'S':0.65,'E':1.19,'T':0.086,'G':1.07,
    'W':1.25,'H':1.13,'Y':0.85,'I':1.05,'V':1.12}

    ONEK900101={'A':13.4,'L':13.0,'R':13.3,'K':13.0,'N':12.0,
    'M':12.8,'D':11.7,'F':12.1,'C':11.6,'P':6.5,
    'Q':12.8,'S':12.2,'E':12.2,'T':11.7,'G':11.3,
    'W':12.4,'H':11.6,'Y':12.1,'I':12.0,'V':11.9}

    ONEK900102={'A':-0.77,'L':-0.62,'R':-0.68,'K':-0.65,'N':-0.07,
    'M':-0.50,'D':-0.15,'F':-0.41,'C':-0.23,'P':3.0,
    'Q':-0.33,'S':-0.35,'E':-0.27,'T':-0.11,'G':0.00,
    'W':-0.45,'H':-0.06,'Y':-0.17,'I':-0.23,'V':-0.14}

    VINM940101={'A':0.984,'L':0.935,'R':1.008,'K':1.102,'N':1.048,
    'M':0.952,'D':1.068,'F':0.915,'C':0.906,'P':1.049,
    'Q':1.037,'S':1.046,'E':1.094,'T':0.997,'G':1.031,
    'W':0.904,'H':0.950,'Y':0.929,'I':0.927,'V':0.931}

    VINM940102={'A':1.315,'L':1.234,'R':1.310,'K':1.367,'N':1.380,
    'M':1.269,'D':1.372,'F':1.247,'C':1.196,'P':1.342,
    'Q':1.342,'S':1.381,'E':1.376,'T':1.324,'G':1.382,
    'W':1.186,'H':1.279,'Y':1.199,'I':1.241,'V':1.235}

    VINM940103={'A':0.994,'L':0.982,'R':1.026,'K':1.029,'N':1.022,
    'M':0.963,'D':1.022,'F':0.934,'C':0.939,'P':1.050,
    'Q':1.041,'S':1.025,'E':1.052,'T':0.998,'G':1.018,
    'W':0.938,'H':0.967,'Y':0.981,'I':0.977,'V':0.968}

    VINM940104={'A':0.783,'L':0.783,'R':0.807,'K':0.834,'N':0.799,
    'M':0.806,'D':0.822,'F':0.774,'C':0.785,'P':0.809,
    'Q':0.817,'S':0.811,'E':0.826,'T':0.795,'G':0.784,
    'W':0.796,'H':0.777,'Y':0.788,'I':0.776,'V':0.781}

    MUNV940101={'A':0.423,'L':0.494,'R':0.503,'K':0.615,'N':0.906,
    'M':0.444,'D':0.870,'F':0.706,'C':0.877,'P':1.945,
    'Q':0.594,'S':0.928,'E':0.167,'T':0.884,'G':1.162,
    'W':0.690,'H':0.802,'Y':0.778,'I':0.566,'V':0.706}

    MUNV940102={'A':0.619,'L':0.740,'R':0.753,'K':0.784,'N':1.089,
    'M':0.736,'D':0.932,'F':0.968,'C':1.107,'P':1.780,
    'Q':0.770,'S':0.969,'E':0.675,'T':1.053,'G':1.361,
    'W':0.910,'H':1.034,'Y':1.009,'I':0.876,'V':0.939}

    MUNV940103={'A':1.080,'L':0.789,'R':0.976,'K':1.026,'N':1.197,
    'M':0.812,'D':1.266,'F':0.685,'C':0.733,'P':1.412,
    'Q':1.050,'S':0.987,'E':1.085,'T':0.784,'G':1.104,
    'W':0.755,'H':0.906,'Y':0.665,'I':0.583,'V':0.546}

    MUNV940104={'A':0.978,'L':0.766,'R':0.784,'K':0.841,'N':0.915,
    'M':0.729,'D':1.038,'F':0.585,'C':0.573,'P':2.613,
    'Q':0.863,'S':0.784,'E':0.962,'T':0.569,'G':1.405,
    'W':0.671,'H':0.724,'Y':0.560,'I':0.502,'V':0.444}

    MUNV940105={'A':1.40,'L':1.33,'R':1.23,'K':1.34,'N':1.61,
    'M':1.12,'D':1.89,'F':1.07,'C':1.14,'P':3.90,
    'Q':1.33,'S':1.20,'E':1.42,'T':0.99,'G':2.06,
    'W':1.10,'H':1.25,'Y':0.98,'I':1.02,'V':0.87}

    WIMW960101={'A':4.08,'L':4.81,'R':3.91,'K':3.77,'N':3.83,
    'M':4.48,'D':3.02,'F':5.38,'C':4.49,'P':3.80,
    'Q':3.67,'S':4.12,'E':2.23,'T':4.11,'G':4.24,
    'W':6.10,'H':4.08,'Y':5.19,'I':4.52,'V':4.18}

    KIMC930101={'A':-0.35,'L':-0.48,'R':-0.44,'K':-0.41,'N':-0.38,
    'M':-0.46,'D':-0.41,'F':-0.55,'C':-0.47,'P':-0.23,
    'Q':-0.40,'S':-0.39,'E':-0.41,'T':-0.48,'G':0.0,
    'W':-0.48,'H':-0.46,'Y':-0.50,'I':-0.56,'V':-0.53}

    MONM990101={'A':0.5,'L':0.4,'R':1.7,'K':1.6,'N':1.7,
    'M':0.5,'D':1.6,'F':0.4,'C':0.6,'P':1.7,
    'Q':1.6,'S':0.7,'E':1.6,'T':0.4,'G':1.3,
    'W':0.7,'H':1.6,'Y':0.6,'I':0.6,'V':0.5}

    BLAM930101={'A':0.96,'L':0.92,'R':0.77,'K':0.73,'N':0.39,
    'M':0.86,'D':0.42,'F':0.59,'C':0.42,'P':-2.50,
    'Q':0.80,'S':0.53,'E':0.53,'T':0.54,'G':0.00,
    'W':0.58,'H':0.57,'Y':0.72,'I':0.84,'V':0.63}

    PARS000101={'A':0.343,'L':0.287,'R':0.353,'K':0.429,'N':0.409,
    'M':0.293,'D':0.429,'F':0.292,'C':0.319,'P':0.432,
    'Q':0.395,'S':0.416,'E':0.405,'T':0.362,'G':0.389,
    'W':0.268,'H':0.307,'Y':0.22,'I':0.296,'V':0.307}

    PARS000102={'A':0.320,'L':0.340,'R':0.327,'K':0.446,'N':0.384,
    'M':0.313,'D':0.424,'F':0.314,'C':0.198,'P':0.354,
    'Q':0.436,'S':0.376,'E':0.514,'T':0.339,'G':0.374,
    'W':0.291,'H':0.299,'Y':0.287,'I':0.306,'V':0.294}

    KUMS000101={'A':8.9,'L':7.4,'R':4.6,'K':6.1,'N':4.4,
    'M':2.3,'D':6.3,'F':3.3,'C':0.6,'P':4.2,
    'Q':2.8,'S':4.0,'E':6.9,'T':5.7,'G':9.4,
    'W':1.3,'H':2.2,'Y':4.5,'I':7.0,'V':8.2}

    KUMS000102={'A':9.2,'L':7.7,'R':3.6,'K':6.5,'N':5.1,
    'M':2.4,'D':6.0,'F':3.4,'C':1.0,'P':4.2,
    'Q':2.9,'S':5.5,'E':6.0,'T':5.7,'G':9.4,
    'W':1.2,'H':2.1,'Y':3.7,'I':6.0,'V':8.2}

    KUMS000103={'A':14.1,'L':9.1,'R':5.5,'K':7.7,'N':3.2,
    'M':3.3,'D':5.7,'F':5.0,'C':0.1,'P':0.7,
    'Q':3.7,'S':3.9,'E':8.8,'T':4.4,'G':4.1,
    'W':1.2,'H':2.0,'Y':4.5,'I':7.1,'V':5.9}

    KUMS000104={'A':13.4,'L':10.6,'R':3.9,'K':7.5,'N':3.7,
    'M':3.0,'D':4.6,'F':4.5,'C':0.8,'P':1.3,
    'Q':4.8,'S':3.8,'E':7.8,'T':4.6,'G':4.6,
    'W':1.0,'H':3.3,'Y':3.3,'I':6.5,'V':7.1}

    TAKK010101={'A':9.8,'L':17.0,'R':7.3,'K':10.5,'N':3.6,
    'M':11.9,'D':4.9,'F':23.0,'C':3.0,'P':15.0,
    'Q':2.4,'S':2.6,'E':4.4,'T':6.9,'G':0.0,
    'W':24.2,'H':11.9,'Y':17.2,'I':17.2,'V':15.3}

    FODM020101={'A':0.70,'L':1.44,'R':0.95,'K':0.91,'N':1.47,
    'M':0.91,'D':0.87,'F':1.34,'C':1.17,'P':0.12,
    'Q':0.73,'S':0.84,'E':0.96,'T':0.74,'G':0.64,
    'W':1.80,'H':1.39,'Y':1.68,'I':1.29,'V':1.20}

    NADH010101={'A':58.0,'L':95.0,'R':-184.0,'K':-24.0,'N':-93.0,
    'M':78.0,'D':-97.0,'F':92.0,'C':116.0,'P':-79.0,
    'Q':-139.0,'S':-34.0,'E':-131.0,'T':-7.0,'G':-11.0,
    'W':59.0,'H':-73.0,'Y':-11.0,'I':107.0,'V':100.0}

    NADH010102={'A':51.0,'L':103.0,'R':-144.0,'K':-205.0,'N':-84.0,
    'M':73.0,'D':-78.0,'F':108.0,'C':137.0,'P':-79.0,
    'Q':-128.0,'S':-26.0,'E':-115.0,'T':-3.0,'G':-13.0,
    'W':69.0,'H':-55.0,'Y':11.0,'I':106.0,'V':108.0}

    NADH010103={'A':41.0,'L':103.0,'R':-109.0,'K':-148.0,'N':-74.0,
    'M':77.0,'D':-47.0,'F':128.0,'C':169.0,'P':-81.0,
    'Q':-104.0,'S':-31.0,'E':-90.0,'T':10.0,'G':-18.0,
    'W':102.0,'H':-35.0,'Y':36.0,'I':104.0,'V':116.0}

    NADH010104={'A':32.0,'L':104.0,'R':-95.0,'K':-124.0,'N':-73.0,
    'M':82.0,'D':-29.0,'F':132.0,'C':182.0,'P':-82.0,
    'Q':-95.0,'S':-34.0,'E':-74.0,'T':20.0,'G':-22.0,
    'W':118.0,'H':-25.0,'Y':44.0,'I':106.0,'V':113.0}

    NADH010105={'A':24.0,'L':103.0,'R':-79.0,'K':-9.0,'N':-76.0,
    'M':90.0,'D':0.0,'F':131.0,'C':194.0,'P':-85.0,
    'Q':-87.0,'S':-36.0,'E':-57.0,'T':34.0,'G':-28.0,
    'W':116.0,'H':-31.0,'Y':43.0,'I':102.0,'V':111.0}

    NADH010106={'A':5.0,'L':82.0,'R':-57.0,'K':-38.0,'N':-77.0,
    'M':83.0,'D':45.0,'F':117.0,'C':224.0,'P':-103.0,
    'Q':-67.0,'S':-41.0,'E':-8.0,'T':79.0,'G':-47.0,
    'W':130.0,'H':-50.0,'Y':27.0,'I':83.0,'V':117.0}

    NADH010107={'A':-2.0,'L':36.0,'R':-41.0,'K':115.0,'N':-97.0,
    'M':62.0,'D':248.0,'F':120.0,'C':329.0,'P':-132.0,
    'Q':-37.0,'S':-52.0,'E':117.0,'T':174.0,'G':-66.0,
    'W':179.0,'H':-70.0,'Y':-7.0,'I':28.0,'V':114.0}

    MONM990201={'A':0.4,'L':0.3,'R':1.5,'K':1.4,'N':1.6,
    'M':0.5,'D':1.5,'F':0.3,'C':0.7,'P':1.6,
    'Q':1.4,'S':0.9,'E':1.3,'T':0.7,'G':1.1,
    'W':0.9,'H':1.4,'Y':0.9,'I':0.5,'V':0.4}

    KOEP990101={'A':-0.04,'L':-0.38,'R':-0.30,'K':-0.18,'N':0.25,
    'M':-0.09,'D':0.27,'F':-0.01,'C':0.57,'P':0.0,
    'Q':-0.02,'S':0.15,'E':-0.33,'T':0.39,'G':1.24,
    'W':0.21,'H':-0.11,'Y':0.05,'I':-0.26,'V':-0.06}

    KOEP990102={'A':-0.12,'L':0.15,'R':0.34,'K':0.29,'N':1.05,
    'M':-0.71,'D':1.12,'F':-0.67,'C':-0.63,'P':0.0,
    'Q':1.67,'S':1.45,'E':0.91,'T':-0.70,'G':0.76,
    'W':-0.14,'H':1.34,'Y':-0.49,'I':-0.77,'V':-0.70}

    CEDJ970101={'A':8.6,'L':8.8,'R':4.2,'K':6.3,'N':4.6,
    'M':2.5,'D':4.9,'F':3.7,'C':2.9,'P':4.9,
    'Q':4.0,'S':7.3,'E':5.1,'T':6.0,'G':7.8,
    'W':1.4,'H':2.1,'Y':3.6,'I':4.6,'V':6.7}

    CEDJ970102={'A':7.6,'L':9.4,'R':5.0,'K':5.8,'N':4.4,
    'M':2.1,'D':5.2,'F':4.0,'C':2.2,'P':5.4,
    'Q':4.1,'S':7.2,'E':6.2,'T':6.1,'G':6.9,
    'W':1.4,'H':2.1,'Y':3.2,'I':5.1,'V':6.7}

    CEDJ970103={'A':8.1,'L':11.0,'R':4.6,'K':4.4,'N':3.7,
    'M':2.8,'D':3.8,'F':5.6,'C':2.0,'P':4.7,
    'Q':3.1,'S':7.3,'E':4.6,'T':5.6,'G':7.0,
    'W':1.8,'H':2.0,'Y':3.3,'I':6.7,'V':7.7}

    CEDJ970104={'A':7.9,'L':8.6,'R':4.9,'K':6.7,'N':4.0,
    'M':2.4,'D':5.5,'F':3.9,'C':1.9,'P':5.3,
    'Q':4.4,'S':6.6,'E':7.1,'T':5.3,'G':7.1,
    'W':1.2,'H':2.1,'Y':3.1,'I':5.2,'V':6.8}

    CEDJ970105={'A':8.3,'L':7.4,'R':8.7,'K':7.9,'N':3.7,
    'M':2.3,'D':4.7,'F':2.7,'C':1.6,'P':6.9,
    'Q':4.7,'S':8.8,'E':6.5,'T':5.1,'G':6.3,
    'W':0.7,'H':2.1,'Y':2.4,'I':3.7,'V':5.3}

    FUKS010101={'A':4.47,'L':5.06,'R':8.48,'K':12.98,'N':3.89,
    'M':1.71,'D':7.05,'F':2.32,'C':0.29,'P':5.41,
    'Q':2.87,'S':4.27,'E':16.56,'T':3.83,'G':8.29,
    'W':0.67,'H':1.74,'Y':2.75,'I':3.30,'V':4.05}

    FUKS010102={'A':6.77,'L':4.43,'R':6.87,'K':10.20,'N':5.50,
    'M':1.87,'D':8.57,'F':1.92,'C':0.31,'P':4.79,
    'Q':5.24,'S':5.41,'E':12.93,'T':5.36,'G':7.95,
    'W':0.54,'H':2.80,'Y':2.26,'I':2.72,'V':3.57}

    FUKS010103={'A':7.43,'L':2.74,'R':4.51,'K':9.67,'N':9.12,
    'M':0.60,'D':8.71,'F':1.18,'C':0.42,'P':5.60,
    'Q':5.42,'S':9.60,'E':5.86,'T':8.95,'G':9.40,
    'W':1.18,'H':1.49,'Y':3.26,'I':1.76,'V':3.10}

    FUKS010104={'A':5.22,'L':4.52,'R':7.30,'K':12.68,'N':6.06,
    'M':1.85,'D':7.91,'F':1.68,'C':1.01,'P':5.70,
    'Q':6.00,'S':6.99,'E':10.66,'T':5.16,'G':5.81,
    'W':0.56,'H':2.27,'Y':2.16,'I':2.36,'V':4.10}

    FUKS010105={'A':9.88,'L':13.21,'R':3.71,'K':3.39,'N':2.35,
    'M':2.44,'D':3.50,'F':5.27,'C':1.12,'P':3.80,
    'Q':1.66,'S':4.10,'E':4.02,'T':4.98,'G':6.88,
    'W':1.11,'H':1.88,'Y':4.07,'I':10.08,'V':12.53}

    FUKS010106={'A':10.98,'L':12.79,'R':3.26,'K':2.54,'N':2.85,
    'M':3.10,'D':3.37,'F':4.97,'C':1.47,'P':3.42,
    'Q':2.30,'S':4.93,'E':3.51,'T':5.55,'G':7.48,
    'W':1.28,'H':2.20,'Y':3.55,'I':9.74,'V':10.69}

    FUKS010107={'A':9.95,'L':9.66,'R':3.05,'K':2.00,'N':4.84,
    'M':2.45,'D':4.46,'F':5.41,'C':1.30,'P':3.20,
    'Q':2.64,'S':6.03,'E':2.58,'T':5.62,'G':8.87,
    'W':2.60,'H':1.99,'Y':6.15,'I':7.73,'V':9.46}

    FUKS010108={'A':8.26,'L':16.46,'R':2.80,'K':1.89,'N':2.54,
    'M':2.67,'D':2.80,'F':7.32,'C':2.67,'P':3.30,
    'Q':2.86,'S':6.00,'E':2.67,'T':5.00,'G':5.62,
    'W':2.01,'H':1.98,'Y':3.96,'I':8.95,'V':10.24}

    FUKS010109={'A':7.39,'L':9.45,'R':5.91,'K':7.81,'N':3.06,
    'M':2.10,'D':5.14,'F':3.91,'C':0.74,'P':4.54,
    'Q':2.22,'S':4.18,'E':9.80,'T':4.45,'G':7.53,
    'W':0.90,'H':1.82,'Y':3.46,'I':6.96,'V':8.62}

    FUKS010110={'A':9.07,'L':9.00,'R':4.90,'K':6.01,'N':4.05,
    'M':2.54,'D':5.73,'F':3.59,'C':0.95,'P':4.04,
    'Q':3.63,'S':5.15,'E':7.77,'T':5.46,'G':7.69,
    'W':0.95,'H':2.47,'Y':2.96,'I':6.56,'V':7.47}

    FUKS010111={'A':8.82,'L':6.54,'R':3.71,'K':5.45,'N':6.77,
    'M':1.62,'D':6.38,'F':3.51,'C':0.90,'P':4.28,
    'Q':3.89,'S':7.64,'E':4.05,'T':7.12,'G':9.11,
    'W':1.96,'H':1.77,'Y':4.85,'I':5.05,'V':6.60}

    FUKS010112={'A':6.65,'L':10.15,'R':5.17,'K':7.59,'N':4.40,
    'M':2.24,'D':5.50,'F':4.34,'C':1.79,'P':4.56,
    'Q':4.52,'S':6.52,'E':6.89,'T':5.08,'G':5.72,
    'W':1.24,'H':2.13,'Y':3.01,'I':5.47,'V':7.00}

    AVBF000101={'A':0.163,'L':0.315,'R':0.220,'K':0.255,'N':0.124,
    'M':0.356,'D':0.212,'F':0.410,'C':0.316,'P':0.0,
    'Q':0.274,'S':0.290,'E':0.212,'T':0.412,'G':0.080,
    'W':0.325,'H':0.315,'Y':0.354,'I':0.474,'V':0.515}

    AVBF000102={'A':0.236,'L':0.293,'R':0.233,'K':0.231,'N':0.189,
    'M':0.367,'D':0.168,'F':0.328,'C':0.259,'P':0.0,
    'Q':0.314,'S':0.202,'E':0.306,'T':0.308,'G':-0.170,
    'W':0.197,'H':0.256,'Y':0.223,'I':0.391,'V':0.436}

    AVBF000103={'A':-0.490,'L':-0.450,'R':-0.429,'K':-0.409,'N':-0.387,
    'M':-0.375,'D':-0.375,'F':-0.309,'C':-0.352,'P':0.0,
    'Q':-0.422,'S':-0.426,'E':-0.382,'T':-0.240,'G':-0.647,
    'W':-0.325,'H':-0.357,'Y':-0.288,'I':-0.268,'V':-0.220}

    AVBF000104={'A':-0.871,'L':-0.798,'R':-0.727,'K':-0.715,'N':-0.741,
    'M':-0.717,'D':-0.737,'F':-0.649,'C':-0.666,'P':0.0,
    'Q':-0.728,'S':-0.679,'E':-0.773,'T':-0.629,'G':-0.822,
    'W':-0.669,'H':-0.685,'Y':-0.655,'I':-0.617,'V':-0.599}

    AVBF000105={'A':-0.393,'L':-0.281,'R':-0.317,'K':-0.294,'N':-0.268,
    'M':-0.274,'D':-0.247,'F':-0.189,'C':-0.222,'P':0.0,
    'Q':-0.291,'S':-0.280,'E':-0.260,'T':-0.152,'G':-0.570,
    'W':-0.206,'H':-0.244,'Y':-0.155,'I':-0.144,'V':-0.080}

    AVBF000106={'A':-0.378,'L':-0.266,'R':-0.369,'K':-0.335,'N':-0.245,
    'M':-0.260,'D':-0.113,'F':-0.187,'C':-0.206,'P':0.0,
    'Q':-0.290,'S':-0.251,'E':-0.165,'T':-0.093,'G':-0.560,
    'W':-0.188,'H':-0.295,'Y':-0.147,'I':-0.134,'V':-0.084}

    AVBF000107={'A':-0.729,'L':-0.462,'R':-0.535,'K':-0.508,'N':-0.597,
    'M':-0.518,'D':-0.545,'F':-0.454,'C':-0.408,'P':0.0,
    'Q':-0.492,'S':-0.278,'E':-0.532,'T':-0.367,'G':-0.860,
    'W':-0.455,'H':-0.519,'Y':-0.439,'I':-0.361,'V':-0.323}

    AVBF000108={'A':-0.623,'L':-0.527,'R':-0.567,'K':-0.581,'N':-0.619,
    'M':-0.571,'D':-0.626,'F':-0.461,'C':-0.571,'P':0.0,
    'Q':-0.559,'S':-0.458,'E':-0.572,'T':-0.233,'G':-0.679,
    'W':-0.327,'H':-0.508,'Y':-0.451,'I':-0.199,'V':-0.263}

    AVBF000109={'A':-0.376,'L':-0.317,'R':-0.280,'K':-0.412,'N':-0.403,
    'M':-0.312,'D':-0.405,'F':-0.237,'C':-0.441,'P':0.0,
    'Q':-0.362,'S':-0.374,'E':-0.362,'T':-0.243,'G':-0.392,
    'W':-0.111,'H':-0.345,'Y':-0.171,'I':-0.194,'V':-0.355}

    YANJ020101={'A':0.0,'L':0.89,'R':0.62,'K':0.77,'N':0.76,
    'M':0.77,'D':0.66,'F':0.92,'C':0.83,'P':0.94,
    'Q':0.59,'S':0.58,'E':0.73,'T':0.73,'G':0.0,
    'W':0.86,'H':0.92,'Y':0.93,'I':0.88,'V':0.88}

    MITS020101={'A':0.0,'L':0.0,'R':2.45,'K':3.67,'N':0.0,
    'M':0.0,'D':0.0,'F':0.0,'C':0.0,'P':0.0,
    'Q':1.25,'S':0.0,'E':1.27,'T':0.0,'G':0.0,
    'W':6.93,'H':1.45,'Y':5.06,'I':0.0,'V':0.0}

    TSAJ990101={'A':89.3,'L':163.1,'R':190.3,'K':165.1,'N':122.4,
    'M':165.8,'D':114.4,'F':190.8,'C':102.5,'P':121.6,
    'Q':146.9,'S':94.2,'E':138.8,'T':119.6,'G':63.8,
    'W':226.4,'H':157.5,'Y':194.6,'I':163.0,'V':138.2}

    TSAJ990102={'A':90.0,'L':164.0,'R':194.0,'K':167.3,'N':124.7,
    'M':167.0,'D':117.3,'F':191.9,'C':103.3,'P':122.9,
    'Q':149.4,'S':95.4,'E':142.2,'T':121.5,'G':64.9,
    'W':228.2,'H':160.0,'Y':197.0,'I':163.9,'V':139.0}

    COSI940101={'A':0.0373,'L':0.0000,'R':0.0959,'K':0.0371,'N':0.0036,
    'M':0.0823,'D':0.1263,'F':0.0946,'C':0.0829,'P':0.0198,
    'Q':0.0761,'S':0.0829,'E':0.0058,'T':0.0941,'G':0.0050,
    'W':0.0548,'H':0.0242,'Y':0.0516,'I':0.0000,'V':0.0057}

    PONP930101={'A':0.85,'L':1.99,'R':0.20,'K':-1.19,'N':-0.48,
    'M':1.42,'D':-1.10,'F':1.69,'C':2.10,'P':-1.14,
    'Q':-0.42,'S':-0.52,'E':-0.79,'T':-0.08,'G':0.0,
    'W':1.76,'H':0.22,'Y':1.37,'I':3.14,'V':2.53}

    WILM950101={'A':0.06,'L':3.50,'R':-0.85,'K':-1.62,'N':0.25,
    'M':0.21,'D':-0.20,'F':4.80,'C':0.49,'P':0.71,
    'Q':0.31,'S':-0.62,'E':-0.10,'T':0.65,'G':0.21,
    'W':2.29,'H':-2.24,'Y':1.89,'I':3.48,'V':1.59}

    WILM950102={'A':2.62,'L':6.57,'R':1.26,'K':-2.78,'N':-1.27,
    'M':-3.12,'D':-2.84,'F':9.14,'C':0.73,'P':-0.12,
    'Q':-1.69,'S':-1.39,'E':-0.45,'T':1.81,'G':-1.15,
    'W':5.91,'H':-0.74,'Y':1.39,'I':4.38,'V':2.30}

    WILM950103={'A':-1.64,'L':0.83,'R':-3.28,'K':-2.36,'N':0.83,
    'M':4.26,'D':0.70,'F':-1.36,'C':9.30,'P':3.12,
    'Q':-0.04,'S':1.59,'E':1.18,'T':2.31,'G':-1.85,
    'W':2.61,'H':7.17,'Y':2.37,'I':3.02,'V':0.52}

    WILM950104={'A':-2.34,'L':1.09,'R':1.60,'K':1.56,'N':2.81,
    'M':0.62,'D':-0.48,'F':2.57,'C':5.03,'P':-0.15,
    'Q':0.16,'S':1.93,'E':1.30,'T':0.19,'G':-1.06,
    'W':3.59,'H':-3.00,'Y':-2.58,'I':7.26,'V':2.06}

    KUHL950101={'A':0.78,'L':0.56,'R':1.58,'K':1.10,'N':1.20,
    'M':0.66,'D':1.35,'F':0.47,'C':0.55,'P':0.69,
    'Q':1.19,'S':1.00,'E':1.45,'T':1.05,'G':0.68,
    'W':0.70,'H':0.99,'Y':1.00,'I':0.47,'V':0.51}

    GUOD860101={'A':253.0,'L':100.0,'R':-7.0,'K':-26.0,'N':-7.0,
    'M':68.0,'D':2.0,'F':100.0,'C':32.0,'P':25.0,
    'Q':0.0,'S':-2.0,'E':14.0,'T':7.0,'G':-2.0,
    'W':109.0,'H':-26.0,'Y':56.0,'I':91.0,'V':62.0}

    JURD980101={'A':1.10,'L':3.80,'R':-5.10,'K':-4.11,'N':-3.50,
    'M':1.90,'D':-3.60,'F':2.80,'C':2.50,'P':-1.90,
    'Q':-3.68,'S':-0.50,'E':-3.20,'T':-0.70,'G':-0.64,
    'W':-0.46,'H':-3.20,'Y':-1.3,'I':4.50,'V':4.2}

    BASU050101={'A':0.1366,'L':0.4251,'R':0.0363,'K':-0.0101,'N':-0.0345,
    'M':0.1747,'D':-0.1233,'F':0.4076,'C':0.2745,'P':0.0019,
    'Q':0.0325,'S':-0.0433,'E':-0.0484,'T':0.0589,'G':-0.0464,
    'W':0.2362,'H':0.0549,'Y':0.3167,'I':0.4172,'V':0.4084}

    BASU050102={'A':0.0728,'L':0.3819,'R':0.0394,'K':-0.0053,'N':-0.0390,
    'M':0.1613,'D':-0.0552,'F':0.4201,'C':0.3557,'P':-0.0492,
    'Q':0.0126,'S':-0.0282,'E':-0.0295,'T':0.0239,'G':-0.0589,
    'W':0.4114,'H':0.0874,'Y':0.3113,'I':0.3805,'V':0.2947}

    BASU050103={'A':0.1510,'L':0.3926,'R':-0.0103,'K':-0.0158,'N':0.0381,
    'M':0.2160,'D':0.0047,'F':0.3455,'C':0.3222,'P':0.0844,
    'Q':0.0246,'S':0.0040,'E':-0.0639,'T':0.1462,'G':0.0248,
    'W':0.2657,'H':0.1335,'Y':0.2998,'I':0.4238,'V':0.3997}

    SUYM030101={'A':-0.058,'L':0.138,'R':0.000,'K':-0.112,'N':0.027,
    'M':0.275,'D':0.016,'F':0.240,'C':0.447,'P':-0.478,
    'Q':-0.073,'S':-0.177,'E':-0.128,'T':-0.163,'G':0.331,
    'W':0.564,'H':0.195,'Y':0.322,'I':0.060,'V':-0.052}

    PUNT030101={'A':-0.17,'L':-0.28,'R':0.37,'K':0.32,'N':0.18,
    'M':-0.26,'D':0.37,'F':-0.41,'C':-0.06,'P':0.13,
    'Q':0.26,'S':0.05,'E':0.15,'T':0.02,'G':0.01,
    'W':-0.15,'H':-0.02,'Y':-0.09,'I':-0.28,'V':-0.17}

    PUNT030102={'A':-0.15,'L':-0.36,'R':0.32,'K':0.24,'N':0.22,
    'M':-0.19,'D':0.41,'F':-0.22,'C':-0.15,'P':0.15,
    'Q':0.03,'S':0.16,'E':0.30,'T':-0.08,'G':0.08,
    'W':-0.28,'H':0.06,'Y':-0.03,'I':-0.29,'V':-0.24}

    GEOR030101={'A':0.964,'L':1.085,'R':1.143,'K':0.944,'N':0.944,
    'M':1.032,'D':0.916,'F':1.119,'C':0.778,'P':1.299,
    'Q':1.047,'S':0.947,'E':1.051,'T':1.017,'G':0.835,
    'W':0.895,'H':1.014,'Y':1.0,'I':0.922,'V':0.955}

    GEOR030102={'A':0.974,'L':1.11,'R':1.129,'K':0.946,'N':0.988,
    'M':0.923,'D':0.892,'F':1.122,'C':0.972,'P':1.362,
    'Q':1.092,'S':0.932,'E':1.054,'T':1.023,'G':0.845,
    'W':0.879,'H':0.949,'Y':0.902,'I':0.928,'V':0.923}

    GEOR030103={'A':0.938,'L':1.0,'R':1.137,'K':0.952,'N':0.902,
    'M':1.077,'D':0.857,'F':1.11,'C':0.6856,'P':1.266,
    'Q':0.916,'S':0.956,'E':1.139,'T':1.018,'G':0.892,
    'W':0.971,'H':1.109,'Y':1.157,'I':0.986,'V':0.959}

    GEOR030104={'A':1.042,'L':1.193,'R':1.069,'K':0.979,'N':0.828,
    'M':0.998,'D':0.97,'F':0.981,'C':0.5,'P':1.332,
    'Q':1.111,'S':0.984,'E':0.992,'T':0.992,'G':0.743,
    'W':0.96,'H':1.034,'Y':1.12,'I':0.852,'V':1.001}

    GEOR030105={'A':1.065,'L':1.192,'R':1.131,'K':0.478,'N':0.762,
    'M':1.369,'D':0.836,'F':1.368,'C':1.015,'P':1.241,
    'Q':0.861,'S':1.097,'E':0.736,'T':0.822,'G':1.022,
    'W':1.017,'H':0.973,'Y':0.836,'I':1.189,'V':1.14}

    GEOR030106={'A':0.99,'L':1.106,'R':1.132,'K':1.003,'N':0.873,
    'M':1.093,'D':0.915,'F':1.121,'C':0.644,'P':1.314,
    'Q':0.999,'S':0.911,'E':1.053,'T':0.988,'G':0.785,
    'W':0.939,'H':1.054,'Y':1.09,'I':0.95,'V':0.957}

    GEOR030107={'A':0.892,'L':0.994,'R':1.154,'K':0.944,'N':1.144,
    'M':0.782,'D':0.925,'F':1.058,'C':1.035,'P':1.309,
    'Q':1.2,'S':0.986,'E':1.115,'T':1.11,'G':0.917,
    'W':0.841,'H':0.992,'Y':0.866,'I':0.817,'V':0.9}

    GEOR030108={'A':1.092,'L':1.276,'R':1.239,'K':1.008,'N':0.927,
    'M':1.171,'D':0.919,'F':1.09,'C':0.662,'P':0.8,
    'Q':1.124,'S':0.886,'E':1.199,'T':0.832,'G':0.698,
    'W':0.981,'H':1.012,'Y':1.075,'I':0.912,'V':0.908}

    GEOR030109={'A':0.843,'L':0.885,'R':1.038,'K':0.893,'N':0.956,
    'M':0.878,'D':0.906,'F':1.151,'C':0.896,'P':1.816,
    'Q':0.968,'S':1.003,'E':0.9,'T':1.189,'G':0.978,
    'W':0.852,'H':1.05,'Y':0.945,'I':0.946,'V':0.999}

    ZHOH040101={'A':2.18,'L':4.71,'R':2.71,'K':2.12,'N':1.85,
    'M':3.63,'D':1.75,'F':5.88,'C':3.89,'P':2.09,
    'Q':2.16,'S':1.66,'E':1.89,'T':2.18,'G':1.17,
    'W':6.46,'H':2.51,'Y':5.01,'I':4.50,'V':3.77}

    ZHOH040102={'A':1.79,'L':4.72,'R':3.20,'K':2.50,'N':2.83,
    'M':3.91,'D':2.33,'F':4.84,'C':2.22,'P':2.45,
    'Q':2.37,'S':1.82,'E':2.52,'T':2.45,'G':0.70,
    'W':5.64,'H':3.06,'Y':4.46,'I':4.59,'V':3.67}

    ZHOH040103={'A':13.4,'L':20.8,'R':8.5,'K':6.1,'N':7.6,
    'M':15.7,'D':8.2,'F':23.9,'C':22.6,'P':9.9,
    'Q':8.5,'S':8.2,'E':7.3,'T':10.3,'G':7.0,
    'W':24.5,'H':11.3,'Y':19.5,'I':20.3,'V':19.5}

    BAEK050101={'A':0.0166,'L':0.2523,'R':-0.0762,'K':-0.2134,'N':-0.0786,
    'M':0.0197,'D':-0.1278,'F':0.3561,'C':0.5724,'P':-0.4188,
    'Q':-0.1051,'S':-0.1629,'E':-0.1794,'T':-0.0701,'G':-0.0442,
    'W':0.3836,'H':0.1643,'Y':0.2500,'I':0.2758,'V':0.1782}

    HARY940101={'A':90.1,'L':164.6,'R':192.8,'K':170.0,'N':127.5,
    'M':167.7,'D':117.1,'F':193.5,'C':113.2,'P':123.1,
    'Q':149.4,'S':94.2,'E':140.8,'T':120.0,'G':63.8,
    'W':197.1,'H':159.3,'Y':231.7,'I':164.9,'V':139.1}

    PONJ960101={'A':91.5,'L':163.4,'R':196.1,'K':162.5,'N':138.3,
    'M':165.9,'D':135.2,'F':198.8,'C':114.4,'P':123.4,
    'Q':156.4,'S':102.0,'E':154.6,'T':126.0,'G':67.5,
    'W':209.8,'H':163.2,'Y':237.2,'I':162.6,'V':138.4}

    DIGM050101={'A':1.076,'L':1.054,'R':1.361,'K':1.105,'N':1.056,
    'M':0.974,'D':1.290,'F':0.869,'C':0.753,'P':0.820,
    'Q':0.729,'S':1.342,'E':1.118,'T':0.871,'G':1.346,
    'W':0.666,'H':0.985,'Y':0.531,'I':0.926,'V':1.131}

    WOLR790101={'A':1.12,'L':1.18,'R':-2.55,'K':-0.80,'N':-0.83,
    'M':0.55,'D':-0.83,'F':0.67,'C':0.59,'P':0.54,
    'Q':-0.78,'S':-0.05,'E':-0.92,'T':-0.02,'G':1.20,
    'W':-0.19,'H':-0.93,'Y':-0.23,'I':1.16,'V':1.13}

    OLSK800101={'A':1.38,'L':1.47,'R':0.00,'K':0.15,'N':0.37,
    'M':1.78,'D':0.52,'F':1.72,'C':1.43,'P':0.85,
    'Q':0.22,'S':0.86,'E':0.71,'T':0.89,'G':1.34,
    'W':0.82,'H':0.66,'Y':0.47,'I':2.32,'V':1.99}

    KIDA850101={'A':-0.27,'L':-1.10,'R':1.87,'K':1.70,'N':0.81,
    'M':-0.73,'D':0.81,'F':-1.43,'C':-1.05,'P':-0.75,
    'Q':1.10,'S':0.42,'E':1.17,'T':0.63,'G':-0.16,
    'W':-1.57,'H':0.28,'Y':-0.56,'I':-0.77,'V':-0.40}

    GUYH850102={'A':0.05,'L':-0.62,'R':0.12,'K':0.57,'N':0.29,
    'M':-0.38,'D':0.41,'F':-0.45,'C':-0.84,'P':0.46,
    'Q':0.46,'S':0.12,'E':0.38,'T':0.38,'G':0.31,
    'W':-0.98,'H':-0.41,'Y':-0.25,'I':-0.69,'V':-0.46}

    GUYH850103={'A':0.54,'L':-1.08,'R':-0.16,'K':0.48,'N':0.38,
    'M':-0.97,'D':0.65,'F':-1.51,'C':-1.13,'P':-0.22,
    'Q':0.05,'S':0.65,'E':0.38,'T':0.27,'G':0.0,
    'W':-1.61,'H':-0.59,'Y':-1.13,'I':-2.15,'V':-0.75}

    GUYH850104={'A':-0.31,'L':-0.53,'R':1.30,'K':1.79,'N':0.49,
    'M':-0.38,'D':0.58,'F':-0.45,'C':-0.87,'P':0.34,
    'Q':0.70,'S':0.10,'E':0.68,'T':0.21,'G':-0.33,
    'W':-0.27,'H':0.13,'Y':0.40,'I':-0.66,'V':-0.62}

    GUYH850105={'A':-0.27,'L':-0.44,'R':2.00,'K':1.17,'N':0.61,
    'M':-0.31,'D':0.50,'F':-0.55,'C':-0.23,'P':0.36,
    'Q':1.00,'S':0.17,'E':0.33,'T':0.18,'G':-0.22,
    'W':0.05,'H':0.37,'Y':0.48,'I':-0.80,'V':-0.65}

    ROSM880104={'A':0.39,'L':1.82,'R':0.0,'K':0.32,'N':-1.91,
    'M':0.96,'D':-0.71,'F':2.27,'C':0.25,'P':0.0,
    'Q':-1.30,'S':-1.24,'E':-0.18,'T':-1.00,'G':0.00,
    'W':2.13,'H':-0.60,'Y':1.47,'I':1.82,'V':1.30}

    ROSM880105={'A':0.39,'L':1.82,'R':-3.95,'K':-2.77,'N':-1.91,
    'M':0.96,'D':-3.81,'F':2.27,'C':0.25,'P':0.0,
    'Q':-1.30,'S':-1.24,'E':-2.91,'T':-1.00,'G':0.00,
    'W':2.13,'H':-0.64,'Y':1.47,'I':1.82,'V':1.30}

    JACR890101={'A':0.18,'L':0.41,'R':-5.40,'K':-2.53,'N':-1.30,
    'M':0.44,'D':-2.36,'F':0.50,'C':0.27,'P':-0.20,
    'Q':-1.22,'S':-0.40,'E':-2.10,'T':-0.34,'G':0.09,
    'W':-0.01,'H':-1.48,'Y':-0.08,'I':0.37,'V':0.32}

    COWR900101={'A':0.42,'L':1.80,'R':-1.56,'K':-2.03,'N':-1.03,
    'M':1.18,'D':-0.51,'F':1.74,'C':0.84,'P':0.86,
    'Q':-0.96,'S':-0.64,'E':-0.37,'T':-0.26,'G':0.00,
    'W':1.46,'H':-2.28,'Y':0.51,'I':1.81,'V':1.34}

    BLAS910101={'A':0.616,'L':0.943,'R':0.000,'K':0.283,'N':0.236,
    'M':0.738,'D':0.028,'F':1.000,'C':0.680,'P':0.711,
    'Q':0.251,'S':0.359,'E':0.043,'T':0.450,'G':0.501,
    'W':0.878,'H':0.165,'Y':0.880,'I':0.943,'V':0.825}

    CASG920101={'A':0.2,'L':0.5,'R':-0.7,'K':-1.6,'N':-0.5,
    'M':0.5,'D':-1.4,'F':1.0,'C':1.9,'P':-1.0,
    'Q':-1.1,'S':-0.7,'E':-1.3,'T':-0.4,'G':-0.1,
    'W':1.6,'H':0.4,'Y':0.5,'I':1.4,'V':0.7}

    CORJ870101={'A':50.76,'L':53.89,'R':48.66,'K':42.92,'N':45.80,
    'M':52.75,'D':43.17,'F':53.45,'C':58.74,'P':45.39,
    'Q':46.09,'S':47.24,'E':43.48,'T':49.26,'G':50.27,
    'W':53.59,'H':49.33,'Y':51.79,'I':57.30,'V':56.12}

    CORJ870102={'A':-0.414,'L':1.215,'R':-0.584,'K':-0.670,'N':-0.916,
    'M':1.020,'D':-1.310,'F':1.938,'C':0.162,'P':-0.503,
    'Q':-0.905,'S':-0.563,'E':-1.218,'T':-0.289,'G':-0.684,
    'W':0.514,'H':-0.630,'Y':1.699,'I':1.237,'V':0.899}

    CORJ870103={'A':-0.96,'L':6.81,'R':0.75,'K':-5.62,'N':-1.94,
    'M':4.76,'D':-5.68,'F':5.06,'C':4.54,'P':-4.47,
    'Q':-5.30,'S':-1.92,'E':-3.86,'T':-3.99,'G':-1.28,
    'W':0.21,'H':-0.62,'Y':3.34,'I':5.54,'V':5.39}

    CORJ870104={'A':-0.26,'L':1.52,'R':0.08,'K':-1.01,'N':-0.46,
    'M':1.09,'D':-1.30,'F':1.09,'C':0.83,'P':-0.62,
    'Q':-0.83,'S':-0.55,'E':-0.73,'T':-0.71,'G':-0.40,
    'W':-0.13,'H':-0.18,'Y':0.69,'I':1.10,'V':1.15}

    CORJ870105={'A':-0.73,'L':4.91,'R':-1.03,'K':-5.99,'N':-5.29,
    'M':3.34,'D':-6.13,'F':5.20,'C':0.64,'P':-4.32,
    'Q':-0.96,'S':-3.00,'E':-2.90,'T':-1.91,'G':-2.67,
    'W':0.51,'H':3.03,'Y':2.87,'I':5.04,'V':3.98}

    CORJ870106={'A':-1.35,'L':9.88,'R':-3.89,'K':-11.92,'N':-10.96,
    'M':7.47,'D':-11.88,'F':11.35,'C':4.37,'P':-10.86,
    'Q':-1.34,'S':-6.21,'E':-4.56,'T':-4.83,'G':-5.82,
    'W':1.80,'H':6.54,'Y':7.61,'I':10.93,'V':8.20}

    CORJ870107={'A':-0.56,'L':4.09,'R':-0.26,'K':-4.08,'N':-2.87,
    'M':3.11,'D':-4.31,'F':3.67,'C':1.78,'P':-3.22,
    'Q':-2.31,'S':-1.85,'E':-2.35,'T':-1.97,'G':-1.35,
    'W':-0.11,'H':0.81,'Y':2.17,'I':3.83,'V':3.31}

    CORJ870108={'A':1.37,'L':-8.68,'R':1.33,'K':7.70,'N':6.29,
    'M':-7.13,'D':8.93,'F':-7.96,'C':-4.47,'P':6.25,
    'Q':3.88,'S':4.08,'E':4.04,'T':4.02,'G':3.39,
    'W':0.79,'H':-1.65,'Y':-4.73,'I':-7.92,'V':-6.94}

    MIYS990101={'A':-0.02,'L':-2.29,'R':0.44,'K':1.01,'N':0.63,
    'M':-1.36,'D':0.72,'F':-2.22,'C':-0.96,'P':0.47,
    'Q':0.56,'S':0.55,'E':0.74,'T':0.25,'G':0.38,
    'W':-1.28,'H':0.00,'Y':-0.88,'I':-1.89,'V':-1.34}

    MIYS990102={'A':0.00,'L':-0.37,'R':0.07,'K':0.17,'N':0.10,
    'M':-0.22,'D':0.12,'F':-0.36,'C':-0.16,'P':0.08,
    'Q':0.09,'S':0.09,'E':0.12,'T':0.04,'G':0.06,
    'W':-0.21,'H':0.00,'Y':-0.14,'I':-0.31,'V':-0.22}

    MIYS990103={'A':-0.03,'L':-0.38,'R':0.09,'K':0.32,'N':0.13,
    'M':-0.30,'D':0.17,'F':-0.34,'C':-0.36,'P':0.20,
    'Q':0.13,'S':0.10,'E':0.23,'T':0.01,'G':0.09,
    'W':-0.24,'H':-0.04,'Y':-0.23,'I':-0.33,'V':-0.29}

    MIYS990104={'A':-0.04,'L':-0.37,'R':0.07,'K':0.33,'N':0.13,
    'M':-0.30,'D':0.19,'F':-0.38,'C':-0.38,'P':0.19,
    'Q':0.14,'S':0.12,'E':0.23,'T':0.03,'G':0.09,
    'W':-0.33,'H':-0.04,'Y':-0.29,'I':-0.34,'V':-0.29}

    MIYS990105={'A':-0.02,'L':-0.32,'R':0.08,'K':0.30,'N':0.10,
    'M':-0.25,'D':0.19,'F':-0.33,'C':-0.32,'P':0.11,
    'Q':0.15,'S':0.11,'E':0.21,'T':0.05,'G':-0.02,
    'W':-0.27,'H':-0.02,'Y':-0.23,'I':-0.28,'V':-0.23}

    ENGD860101={'A':-1.6,'L':-2.8,'R':12.3,'K':8.8,'N':4.8,
    'M':-3.4,'D':9.2,'F':-3.7,'C':-2.0,'P':0.2,
    'Q':4.1,'S':-0.6,'E':8.2,'T':-1.2,'G':-1.0,
    'W':-1.9,'H':3.0,'Y':0.7,'I':-3.1,'V':-2.6}

    FASG890101={'A':-0.21,'L':-4.68,'R':2.11,'K':3.88,'N':0.96,
    'M':-3.66,'D':1.36,'F':-4.65,'C':-6.04,'P':0.75,
    'Q':1.52,'S':1.74,'E':2.30,'T':0.78,'G':0.00,
    'W':-3.32,'H':-1.23,'Y':-1.01,'I':-4.81,'V':-3.50}

    KARS160101={'A':2.00,'L':5.00,'R':8.00,'K':6.00,'N':5.00,
    'M':5.00,'D':5.00,'F':8.00,'C':3.00,'P':4.00,
    'Q':6.00,'S':3.00,'E':6.00,'T':4.00,'G':1.00,
    'W':11.00,'H':7.00,'Y':9.00,'I':5.00,'V':4.00}

    KARS160102={'A':1.00,'L':4.00,'R':7.00,'K':5.00,'N':4.00,
    'M':4.00,'D':4.00,'F':8.00,'C':2.00,'P':4.00,
    'Q':5.00,'S':2.00,'E':5.00,'T':3.00,'G':0.00,
    'W':12.00,'H':6.00,'Y':9.00,'I':4.00,'V':3.00}

    KARS160103={'A':2.00,'L':8.00,'R':12.00,'K':10.00,'N':8.00,
    'M':8.00,'D':8.00,'F':14.00,'C':4.00,'P':8.00,
    'Q':10.00,'S':4.00,'E':10.00,'T':6.00,'G':0.00,
    'W':24.00,'H':14.00,'Y':18.00,'I':8.00,'V':6.00}

    KARS160104={'A':1.00,'L':4.00,'R':6.00,'K':4.00,'N':4.00,
    'M':4.00,'D':4.00,'F':6.00,'C':2.00,'P':4.00,
    'Q':4.00,'S':2.00,'E':5.00,'T':3.00,'G':1.00,
    'W':8.00,'H':6.000,'Y':7.00,'I':4.00,'V':3.00}

    KARS160105={'A':1.00,'L':5.00,'R':8.120,'K':7.00,'N':5.00,
    'M':5.40,'D':5.17,'F':7.00,'C':2.33,'P':4.00,
    'Q':5.860,'S':1.670,'E':6.00,'T':3.250,'G':0.00,
    'W':11.10,'H':6.71,'Y':8.88,'I':3.25,'V':3.25}

    KARS160106={'A':1.00,'L':3.00,'R':6.00,'K':5.00,'N':3.00,
    'M':3.00,'D':3.00,'F':6.000,'C':1.00,'P':4.00,
    'Q':4.00,'S':2.00,'E':4.00,'T':1.00,'G':0.00,
    'W':9.000,'H':6.000,'Y':6.000,'I':3.00,'V':1.00}

    KARS160107={'A':1.00,'L':6.00,'R':12.00,'K':9.00,'N':6.00,
    'M':7.00,'D':6.00,'F':11.000,'C':3.00,'P':4.000,
    'Q':8.00,'S':3.00,'E':8.00,'T':4.00,'G':0.00,
    'W':14.000,'H':9.00,'Y':13.000,'I':6.00,'V':4.00}

    KARS160108={'A':1.00,'L':1.60,'R':1.50,'K':1.667,'N':1.60,
    'M':1.60,'D':1.60,'F':1.750,'C':1.333,'P':2.00,
    'Q':1.667,'S':1.333,'E':1.667,'T':1.50,'G':0.00,
    'W':2.182,'H':2.00,'Y':2.000,'I':1.600,'V':1.50}

    KARS160109={'A':2.00,'L':11.029,'R':12.499,'K':10.363,'N':11.539,
    'M':9.49,'D':11.539,'F':14.851,'C':6.243,'P':12.00,
    'Q':12.207,'S':5.00,'E':11.530,'T':9.928,'G':0.00,
    'W':13.511,'H':12.876,'Y':12.868,'I':10.851,'V':9.928}

    KARS160110={'A':0.00,'L':-4.729,'R':-4.307,'K':-3.151,'N':-4.178,
    'M':-2.812,'D':-4.178,'F':-4.801,'C':-2.243,'P':-4.00,
    'Q':-4.255,'S':1.00,'E':-3.425,'T':-3.928,'G':0.00,
    'W':-6.324,'H':-3.721,'Y':-4.793,'I':-6.085,'V':-3.928}

    KARS160111={'A':1.00,'L':3.20,'R':3.500,'K':3.00,'N':3.20,
    'M':2.80,'D':3.20,'F':4.25,'C':2.00,'P':4.00,
    'Q':3.333,'S':2.00,'E':3.333,'T':3.00,'G':0.00,
    'W':4.00,'H':4.286,'Y':4.333,'I':1.80,'V':3.00}

    KARS160112={'A':2.00,'L':1.052,'R':-2.590,'K':-0.536,'N':0.528,
    'M':0.678,'D':0.528,'F':-1.672,'C':2.00,'P':4.00,
    'Q':-1.043,'S':2.00,'E':-0.538,'T':3.00,'G':0.00,
    'W':-2.576,'H':-1.185,'Y':-2.054,'I':-1.517,'V':3.00}

    KARS160113={'A':6.00,'L':12.00,'R':19.00,'K':12.00,'N':12.00,
    'M':18.00,'D':12.00,'F':18.00,'C':6.00,'P':12.00,
    'Q':12.00,'S':6.00,'E':12.00,'T':6.00,'G':1.00,
    'W':24.00,'H':15.00,'Y':18.00,'I':12.00,'V':6.00}

    KARS160114={'A':6.00,'L':15.60,'R':31.444,'K':24.50,'N':16.50,
    'M':27.20,'D':16.40,'F':23.25,'C':16.670,'P':12.00,
    'Q':21.167,'S':13.33,'E':21.00,'T':12.40,'G':3.50,
    'W':27.50,'H':23.10,'Y':27.78,'I':15.60,'V':10.50}

    KARS160115={'A':6.00,'L':12.00,'R':20.00,'K':18.00,'N':14.00,
    'M':18.00,'D':12.00,'F':18.00,'C':12.00,'P':12.00,
    'Q':15.00,'S':8.00,'E':14.00,'T':8.00,'G':1.00,
    'W':18.00,'H':18.00,'Y':20.00,'I':12.00,'V':6.00}

    KARS160116={'A':6.00,'L':18.00,'R':38.00,'K':31.00,'N':20.00,
    'M':34.00,'D':20.00,'F':24.00,'C':22.00,'P':12.00,
    'Q':24.00,'S':20.00,'E':26.00,'T':14.00,'G':6.00,
    'W':36.00,'H':31.00,'Y':38.00,'I':18.00,'V':12.00}

    KARS160117={'A':12.00,'L':30.00,'R':45.00,'K':37.00,'N':33.007,
    'M':40.00,'D':34.00,'F':48.00,'C':28.00,'P':24.00,
    'Q':39.00,'S':22.00,'E':40.00,'T':27.00,'G':7.00,
    'W':68.00,'H':47.00,'Y':56.00,'I':30.00,'V':24.007}

    KARS160118={'A':6.00,'L':6.00,'R':5.00,'K':6.17,'N':6.60,
    'M':8.00,'D':6.80,'F':6.00,'C':9.33,'P':6.00,
    'Q':6.50,'S':7.33,'E':6.67,'T':5.40,'G':3.50,
    'W':5.667,'H':4.70,'Y':6.22,'I':6.00,'V':6.00}

    KARS160119={'A':12.00,'L':25.021,'R':23.343,'K':22.739,'N':27.708,
    'M':31.344,'D':28.634,'F':26.993,'C':28.00,'P':24.00,
    'Q':27.831,'S':20.00,'E':28.731,'T':23.819,'G':7.00,
    'W':29.778,'H':24.243,'Y':28.252,'I':24.841,'V':24.00}

    KARS160120={'A':0.00,'L':0.00,'R':0.00,'K':-0.179,'N':0.00,
    'M':0.00,'D':0.00,'F':0.00,'C':0.00,'P':0.00,
    'Q':0.00,'S':0.00,'E':0.00,'T':-4.227,'G':0.00,
    'W':0.211,'H':-1.734,'Y':-0.96,'I':-1.641,'V':0.00}

    KARS160121={'A':6.00,'L':9.60,'R':10.667,'K':10.167,'N':10.00,
    'M':13.60,'D':10.40,'F':12.00,'C':11.333,'P':12.00,
    'Q':10.50,'S':8.667,'E':10.667,'T':9.00,'G':3.50,
    'W':12.75,'H':10.400,'Y':12.222,'I':9.60,'V':9.00}

    KARS160122={'A':0.00,'L':3.113,'R':4.20,'K':1.372,'N':3.00,
    'M':2.656,'D':2.969,'F':2.026,'C':6.00,'P':12.00,
    'Q':1.849,'S':6.00,'E':1.822,'T':6.00,'G':0.00,
    'W':2.044,'H':1.605,'Y':1.599,'I':3.373,'V':6.00}

    KRIW790101={'A':4.32,'L':3.93,'R':6.55,'K':7.92,'N':6.24,
    'M':2.44,'D':6.04,'F':2.59,'C':1.73,'P':7.19,
    'Q':6.13,'S':5.37,'E':6.17,'T':5.16,'G':6.09,
    'W':2.78,'H':5.66,'Y':3.58,'I':2.31,'V':3.31}

    aaindex_values = []

    aaindex_listT = [ANDN920101, ARGP820101, ARGP820102, ARGP820103, BEGF750101, BEGF750102, BEGF750103, BHAR880101,
                        BIGC670101, BIOV880101, BIOV880102, BROC820101, BROC820102, BULH740101, BULH740102, BUNA790101,
                        BUNA790102, BUNA790103, BURA740101, BURA740102, CHAM810101, CHAM820101, CHAM820102, CHAM830101,
                        CHAM830102, CHAM830103, CHAM830104, CHAM830105, CHAM830106, CHAM830107, CHAM830108, CHOC750101,
                        CHOC760101, CHOC760102, CHOC760103, CHOC760104, CHOP780101, CHOP780201, CHOP780202, CHOP780203,
                        CHOP780204, CHOP780205, CHOP780206, CHOP780207, CHOP780208, CHOP780209, CHOP780210, CHOP780211,
                        CHOP780212, CHOP780213, CHOP780214, CHOP780215, CHOP780216, CIDH920101, CIDH920102, CIDH920103,
                        CIDH920104, CIDH920105, COHE430101, CRAJ730101, CRAJ730102, CRAJ730103, DAWD720101, DAYM780101,
                        DAYM780201, DESM900101, DESM900102, EISD840101, EISD860101, EISD860102, EISD860103, FASG760101,
                        FASG760102, FASG760103, FASG760104, FASG760105, FAUJ830101, FAUJ880101, FAUJ880102, FAUJ880103,
                        FAUJ880104, FAUJ880105, FAUJ880106, FAUJ880107, FAUJ880108, FAUJ880109, FAUJ880110, FAUJ880111,
                        FAUJ880112, FAUJ880113, FINA770101, FINA910101, FINA910102, FINA910103, FINA910104, GARJ730101,
                        GEIM800101, GEIM800102, GEIM800103, GEIM800104, GEIM800105, GEIM800106, GEIM800107, GEIM800108,
                        GEIM800109, GEIM800110, GEIM800111, GOLD730101, GOLD730102, GRAR740101, GRAR740102, GRAR740103,
                        GUYH850101, HOPA770101, HOPT810101, HUTJ700101, HUTJ700102, HUTJ700103, ISOY800101, ISOY800102,
                        ISOY800103, ISOY800104, ISOY800105, ISOY800106, ISOY800107, ISOY800108, JANJ780101, JANJ780102,
                        JANJ780103, JANJ790101, JANJ790102, JOND750101, JOND750102, JOND920101, JOND920102, JUKT750101,
                        JUNJ780101, KANM800101, KANM800102, KANM800103, KANM800104, KARP850101, KARP850102, KARP850103,
                        KHAG800101, KLEP840101, KRIW710101, KRIW790101, KRIW790102, KRIW790103, KYTJ820101, LAWE840101,
                        LEVM760101, LEVM760102, LEVM760103, LEVM760104, LEVM760105, LEVM760106, LEVM760107, LEVM780101,
                        LEVM780102, LEVM780103, LEVM780104, LEVM780105, LEVM780106, LEWP710101, LIFS790101, LIFS790102,
                        LIFS790103, MANP780101, MAXF760101, MAXF760102, MAXF760103, MAXF760104, MAXF760105, MAXF760106,
                        MCMT640101, MEEJ800101, MEEJ800102, MEEJ810101, MEEJ810102, MEIH800101, MEIH800102, MEIH800103,
                        MIYS850101, NAGK730101, NAGK730102, NAGK730103, NAKH900101, NAKH900102, NAKH900103, NAKH900104,
                        NAKH900105, NAKH900106, NAKH900107, NAKH900108, NAKH900109, NAKH900110, NAKH900111, NAKH900112,
                        NAKH900113, NAKH920101, NAKH920102, NAKH920103, NAKH920104, NAKH920105, NAKH920106, NAKH920107,
                        NAKH920108, NISK800101, NISK860101, NOZY710101, OOBM770101, OOBM770102, OOBM770103, OOBM770104,
                        OOBM770105, OOBM850101, OOBM850102, OOBM850103, OOBM850104, OOBM850105, PALJ810101, PALJ810102,
                        PALJ810103, PALJ810104, PALJ810105, PALJ810106, PALJ810107, PALJ810108, PALJ810109, PALJ810110,
                        PALJ810111, PALJ810112, PALJ810113, PALJ810114, PALJ810115, PALJ810116, PARJ860101, PLIV810101,
                        PONP800101, PONP800102, PONP800103, PONP800104, PONP800105, PONP800106, PONP800107, PONP800108,
                        PRAM820101, PRAM820102, PRAM820103, PRAM900101, PRAM900102, PRAM900103, PRAM900104, PTIO830101,
                        PTIO830102, QIAN880101, QIAN880102, QIAN880103, QIAN880104, QIAN880105, QIAN880106, QIAN880107,
                        QIAN880108, QIAN880109, QIAN880110, QIAN880111, QIAN880112, QIAN880113, QIAN880114, QIAN880115,
                        QIAN880116, QIAN880117, QIAN880118, QIAN880119, QIAN880120, QIAN880121, QIAN880122, QIAN880123,
                        QIAN880124, QIAN880125, QIAN880126, QIAN880127, QIAN880128, QIAN880129, QIAN880130, QIAN880131,
                        QIAN880132, QIAN880133, QIAN880134, QIAN880135, QIAN880136, QIAN880137, QIAN880138, QIAN880139,
                        RACS770101, RACS770102, RACS770103, RACS820101, RACS820102, RACS820103, RACS820104, RACS820105,
                        RACS820106, RACS820107, RACS820108, RACS820109, RACS820110, RACS820111, RACS820112, RACS820113,
                        RACS820114, RADA880101, RADA880102, RADA880103, RADA880104, RADA880105, RADA880106, RADA880107,
                        RADA880108, RICJ880101, RICJ880102, RICJ880103, RICJ880104, RICJ880105, RICJ880106, RICJ880107,
                        RICJ880108, RICJ880109, RICJ880110, RICJ880111, RICJ880112, RICJ880113, RICJ880114, RICJ880115,
                        RICJ880116, RICJ880117, ROBB760101, ROBB760102, ROBB760103, ROBB760104, ROBB760105, ROBB760106,
                        ROBB760107, ROBB760108, ROBB760109, ROBB760110, ROBB760111, ROBB760112, ROBB760113, ROBB790101,
                        ROSG850101, ROSG850102, ROSM880101, ROSM880102, ROSM880103, SIMZ760101, SNEP660101, SNEP660102,
                        SNEP660103, SNEP660104, SUEM840101, SUEM840102, SWER830101, TANS770101, TANS770102, TANS770103,
                        TANS770104, TANS770105, TANS770106, TANS770107, TANS770108, TANS770109, TANS770110, VASM830101,
                        VASM830102, VASM830103, VELV850101, VENT840101, VHEG790101, WARP780101, WEBA780101, WERD780101,
                        WERD780102, WERD780103, WERD780104, WOEC730101, WOLR810101, WOLS870101, WOLS870102, WOLS870103,
                        YUTK870101, YUTK870102, YUTK870103, YUTK870104, ZASB820101, ZIMJ680101, ZIMJ680102, ZIMJ680103,
                        ZIMJ680104, ZIMJ680105, AURR980101, AURR980102, AURR980103, AURR980104, AURR980105, AURR980106,
                        AURR980107, AURR980108, AURR980109, AURR980110, AURR980111, AURR980112, AURR980113, AURR980114,
                        AURR980115, AURR980116, AURR980117, AURR980118, AURR980119, AURR980120, ONEK900101, ONEK900102,
                        VINM940101, VINM940102, VINM940103, VINM940104, MUNV940101, MUNV940102, MUNV940103, MUNV940104,
                        MUNV940105, WIMW960101, KIMC930101, MONM990101, BLAM930101, PARS000101, PARS000102, KUMS000101,
                        KUMS000102, KUMS000103, KUMS000104, TAKK010101, FODM020101, NADH010101, NADH010102, NADH010103,
                        NADH010104, NADH010105, NADH010106, NADH010107, MONM990201, KOEP990101, KOEP990102, CEDJ970101,
                        CEDJ970102, CEDJ970103, CEDJ970104, CEDJ970105, FUKS010101, FUKS010102, FUKS010103, FUKS010104,
                        FUKS010105, FUKS010106, FUKS010107, FUKS010108, FUKS010109, FUKS010110, FUKS010111, FUKS010112,
                        AVBF000101, AVBF000102, AVBF000103, AVBF000104, AVBF000105, AVBF000106, AVBF000107, AVBF000108,
                        AVBF000109, YANJ020101, MITS020101, TSAJ990101, TSAJ990102, COSI940101, PONP930101, WILM950101,
                        WILM950102, WILM950103, WILM950104, KUHL950101, GUOD860101, JURD980101, BASU050101, BASU050102,
                        BASU050103, SUYM030101, PUNT030101, PUNT030102, GEOR030101, GEOR030102, GEOR030103, GEOR030104,
                        GEOR030105, GEOR030106, GEOR030107, GEOR030108, GEOR030109, ZHOH040101, ZHOH040102, ZHOH040103,
                        BAEK050101, HARY940101, PONJ960101, DIGM050101, WOLR790101, OLSK800101, KIDA850101, GUYH850102,
                        GUYH850103, GUYH850104, GUYH850105, ROSM880104, ROSM880105, JACR890101, COWR900101, BLAS910101,
                        CASG920101, CORJ870101, CORJ870102, CORJ870103, CORJ870104, CORJ870105, CORJ870106, CORJ870107,
                        CORJ870108, MIYS990101, MIYS990102, MIYS990103, MIYS990104, MIYS990105, ENGD860101, FASG890101,
                        KARS160101, KARS160102, KARS160103, KARS160104, KARS160105, KARS160106, KARS160107, KARS160108,
                        KARS160109, KARS160110, KARS160111, KARS160112, KARS160113, KARS160114, KARS160115, KARS160116,
                        KARS160117, KARS160118, KARS160119, KARS160120, KARS160121, KARS160122]


    aaindex_listT_str = "ANDN920101, ARGP820101, ARGP820102, ARGP820103, BEGF750101, BEGF750102, BEGF750103, BHAR880101,\
                    BIGC670101, BIOV880101, BIOV880102, BROC820101, BROC820102, BULH740101, BULH740102, BUNA790101,\
                    BUNA790102, BUNA790103, BURA740101, BURA740102, CHAM810101, CHAM820101, CHAM820102, CHAM830101,\
                    CHAM830102, CHAM830103, CHAM830104, CHAM830105, CHAM830106, CHAM830107, CHAM830108, CHOC750101,\
                    CHOC760101, CHOC760102, CHOC760103, CHOC760104, CHOP780101, CHOP780201, CHOP780202, CHOP780203,\
                    CHOP780204, CHOP780205, CHOP780206, CHOP780207, CHOP780208, CHOP780209, CHOP780210, CHOP780211,\
                    CHOP780212, CHOP780213, CHOP780214, CHOP780215, CHOP780216, CIDH920101, CIDH920102, CIDH920103,\
                    CIDH920104, CIDH920105, COHE430101, CRAJ730101, CRAJ730102, CRAJ730103, DAWD720101, DAYM780101,\
                    DAYM780201, DESM900101, DESM900102, EISD840101, EISD860101, EISD860102, EISD860103, FASG760101,\
                    FASG760102, FASG760103, FASG760104, FASG760105, FAUJ830101, FAUJ880101, FAUJ880102, FAUJ880103,\
                    FAUJ880104, FAUJ880105, FAUJ880106, FAUJ880107, FAUJ880108, FAUJ880109, FAUJ880110, FAUJ880111,\
                    FAUJ880112, FAUJ880113, FINA770101, FINA910101, FINA910102, FINA910103, FINA910104, GARJ730101,\
                    GEIM800101, GEIM800102, GEIM800103, GEIM800104, GEIM800105, GEIM800106, GEIM800107, GEIM800108,\
                    GEIM800109, GEIM800110, GEIM800111, GOLD730101, GOLD730102, GRAR740101, GRAR740102, GRAR740103,\
                    GUYH850101, HOPA770101, HOPT810101, HUTJ700101, HUTJ700102, HUTJ700103, ISOY800101, ISOY800102,\
                    ISOY800103, ISOY800104, ISOY800105, ISOY800106, ISOY800107, ISOY800108, JANJ780101, JANJ780102,\
                    JANJ780103, JANJ790101, JANJ790102, JOND750101, JOND750102, JOND920101, JOND920102, JUKT750101,\
                    JUNJ780101, KANM800101, KANM800102, KANM800103, KANM800104, KARP850101, KARP850102, KARP850103,\
                    KHAG800101, KLEP840101, KRIW710101, KRIW790101, KRIW790102, KRIW790103, KYTJ820101, LAWE840101,\
                    LEVM760101, LEVM760102, LEVM760103, LEVM760104, LEVM760105, LEVM760106, LEVM760107, LEVM780101,\
                    LEVM780102, LEVM780103, LEVM780104, LEVM780105, LEVM780106, LEWP710101, LIFS790101, LIFS790102,\
                    LIFS790103, MANP780101, MAXF760101, MAXF760102, MAXF760103, MAXF760104, MAXF760105, MAXF760106,\
                    MCMT640101, MEEJ800101, MEEJ800102, MEEJ810101, MEEJ810102, MEIH800101, MEIH800102, MEIH800103,\
                    MIYS850101, NAGK730101, NAGK730102, NAGK730103, NAKH900101, NAKH900102, NAKH900103, NAKH900104,\
                    NAKH900105, NAKH900106, NAKH900107, NAKH900108, NAKH900109, NAKH900110, NAKH900111, NAKH900112,\
                    NAKH900113, NAKH920101, NAKH920102, NAKH920103, NAKH920104, NAKH920105, NAKH920106, NAKH920107,\
                    NAKH920108, NISK800101, NISK860101, NOZY710101, OOBM770101, OOBM770102, OOBM770103, OOBM770104,\
                    OOBM770105, OOBM850101, OOBM850102, OOBM850103, OOBM850104, OOBM850105, PALJ810101, PALJ810102,\
                    PALJ810103, PALJ810104, PALJ810105, PALJ810106, PALJ810107, PALJ810108, PALJ810109, PALJ810110,\
                    PALJ810111, PALJ810112, PALJ810113, PALJ810114, PALJ810115, PALJ810116, PARJ860101, PLIV810101,\
                    PONP800101, PONP800102, PONP800103, PONP800104, PONP800105, PONP800106, PONP800107, PONP800108,\
                    PRAM820101, PRAM820102, PRAM820103, PRAM900101, PRAM900102, PRAM900103, PRAM900104, PTIO830101,\
                    PTIO830102, QIAN880101, QIAN880102, QIAN880103, QIAN880104, QIAN880105, QIAN880106, QIAN880107,\
                    QIAN880108, QIAN880109, QIAN880110, QIAN880111, QIAN880112, QIAN880113, QIAN880114, QIAN880115,\
                    QIAN880116, QIAN880117, QIAN880118, QIAN880119, QIAN880120, QIAN880121, QIAN880122, QIAN880123,\
                    QIAN880124, QIAN880125, QIAN880126, QIAN880127, QIAN880128, QIAN880129, QIAN880130, QIAN880131,\
                    QIAN880132, QIAN880133, QIAN880134, QIAN880135, QIAN880136, QIAN880137, QIAN880138, QIAN880139,\
                    RACS770101, RACS770102, RACS770103, RACS820101, RACS820102, RACS820103, RACS820104, RACS820105,\
                    RACS820106, RACS820107, RACS820108, RACS820109, RACS820110, RACS820111, RACS820112, RACS820113,\
                    RACS820114, RADA880101, RADA880102, RADA880103, RADA880104, RADA880105, RADA880106, RADA880107,\
                    RADA880108, RICJ880101, RICJ880102, RICJ880103, RICJ880104, RICJ880105, RICJ880106, RICJ880107,\
                    RICJ880108, RICJ880109, RICJ880110, RICJ880111, RICJ880112, RICJ880113, RICJ880114, RICJ880115,\
                    RICJ880116, RICJ880117, ROBB760101, ROBB760102, ROBB760103, ROBB760104, ROBB760105, ROBB760106,\
                    ROBB760107, ROBB760108, ROBB760109, ROBB760110, ROBB760111, ROBB760112, ROBB760113, ROBB790101,\
                    ROSG850101, ROSG850102, ROSM880101, ROSM880102, ROSM880103, SIMZ760101, SNEP660101, SNEP660102,\
                    SNEP660103, SNEP660104, SUEM840101, SUEM840102, SWER830101, TANS770101, TANS770102, TANS770103,\
                    TANS770104, TANS770105, TANS770106, TANS770107, TANS770108, TANS770109, TANS770110, VASM830101,\
                    VASM830102, VASM830103, VELV850101, VENT840101, VHEG790101, WARP780101, WEBA780101, WERD780101,\
                    WERD780102, WERD780103, WERD780104, WOEC730101, WOLR810101, WOLS870101, WOLS870102, WOLS870103,\
                    YUTK870101, YUTK870102, YUTK870103, YUTK870104, ZASB820101, ZIMJ680101, ZIMJ680102, ZIMJ680103,\
                    ZIMJ680104, ZIMJ680105, AURR980101, AURR980102, AURR980103, AURR980104, AURR980105, AURR980106,\
                    AURR980107, AURR980108, AURR980109, AURR980110, AURR980111, AURR980112, AURR980113, AURR980114,\
                    AURR980115, AURR980116, AURR980117, AURR980118, AURR980119, AURR980120, ONEK900101, ONEK900102,\
                    VINM940101, VINM940102, VINM940103, VINM940104, MUNV940101, MUNV940102, MUNV940103, MUNV940104,\
                    MUNV940105, WIMW960101, KIMC930101, MONM990101, BLAM930101, PARS000101, PARS000102, KUMS000101,\
                    KUMS000102, KUMS000103, KUMS000104, TAKK010101, FODM020101, NADH010101, NADH010102, NADH010103,\
                    NADH010104, NADH010105, NADH010106, NADH010107, MONM990201, KOEP990101, KOEP990102, CEDJ970101,\
                    CEDJ970102, CEDJ970103, CEDJ970104, CEDJ970105, FUKS010101, FUKS010102, FUKS010103, FUKS010104,\
                    FUKS010105, FUKS010106, FUKS010107, FUKS010108, FUKS010109, FUKS010110, FUKS010111, FUKS010112,\
                    AVBF000101, AVBF000102, AVBF000103, AVBF000104, AVBF000105, AVBF000106, AVBF000107, AVBF000108,\
                    AVBF000109, YANJ020101, MITS020101, TSAJ990101, TSAJ990102, COSI940101, PONP930101, WILM950101,\
                    WILM950102, WILM950103, WILM950104, KUHL950101, GUOD860101, JURD980101, BASU050101, BASU050102,\
                    BASU050103, SUYM030101, PUNT030101, PUNT030102, GEOR030101, GEOR030102, GEOR030103, GEOR030104,\
                    GEOR030105, GEOR030106, GEOR030107, GEOR030108, GEOR030109, ZHOH040101, ZHOH040102, ZHOH040103,\
                    BAEK050101, HARY940101, PONJ960101, DIGM050101, WOLR790101, OLSK800101, KIDA850101, GUYH850102,\
                    GUYH850103, GUYH850104, GUYH850105, ROSM880104, ROSM880105, JACR890101, COWR900101, BLAS910101,\
                    CASG920101, CORJ870101, CORJ870102, CORJ870103, CORJ870104, CORJ870105, CORJ870106, CORJ870107,\
                    CORJ870108, MIYS990101, MIYS990102, MIYS990103, MIYS990104, MIYS990105, ENGD860101, FASG890101,\
                    KARS160101, KARS160102, KARS160103, KARS160104, KARS160105, KARS160106, KARS160107, KARS160108,\
                    KARS160109, KARS160110, KARS160111, KARS160112, KARS160113, KARS160114, KARS160115, KARS160116,\
                    KARS160117, KARS160118, KARS160119, KARS160120, KARS160121, KARS160122"

    aaindex_listT_str = aaindex_listT_str.split(",")
    aaindex_col = []
    for j in aaindex_listT_str:
        aaindex_col.append(j.replace(' ', ''))
    # print(aaindex_col)

    for i in aaindex_listT:
        a_a = ((sequence.count("A") * i["A"])) / len(sequence)
        c_c = ((sequence.count("C") * i["C"])) / len(sequence)
        d_d = ((sequence.count("D") * i["D"])) / len(sequence)
        e_e = ((sequence.count("E") * i["E"])) / len(sequence)
        f_f = ((sequence.count("F") * i["F"])) / len(sequence)
        g_g = ((sequence.count("G") * i["G"])) / len(sequence)
        h_h = ((sequence.count("H") * i["H"])) / len(sequence)
        i_i = ((sequence.count("I") * i["I"])) / len(sequence)
        k_k = ((sequence.count("K") * i["K"])) / len(sequence)
        l_l = ((sequence.count("L") * i["L"])) / len(sequence)
        m_m = ((sequence.count("M") * i["M"])) / len(sequence)
        n_n = ((sequence.count("N") * i["N"])) / len(sequence)
        p_p = ((sequence.count("P") * i["P"])) / len(sequence)
        q_q = ((sequence.count("Q") * i["Q"])) / len(sequence)
        r_r = ((sequence.count("R") * i["R"])) / len(sequence)
        s_s = ((sequence.count("S") * i["S"])) / len(sequence)
        t_t = ((sequence.count("T") * i["T"])) / len(sequence)
        v_v = ((sequence.count("V") * i["V"])) / len(sequence)
        w_w = ((sequence.count("W") * i["W"])) / len(sequence)
        y_y = ((sequence.count("Y") * i["Y"])) / len(sequence)

        aaindex_comp = round(((a_a + c_c + d_d + e_e + f_f + g_g + h_h + i_i + k_k + l_l + m_m + n_n + p_p + q_q + r_r + s_s + t_t + v_v + w_w + y_y) / 20),9)

        aaindex_values.append(aaindex_comp)

    aaindex_values = pd.DataFrame(np.array(aaindex_values).reshape(1, -1), columns=aaindex_col)

    return aaindex_values

def toFeatureAAindex(infasta):
    seqlist = fasta2csv(infasta)
    aaindex_feature =[]

    for seq in seqlist["Seq"]:
        seqaaidx = aaindex(seq)
        aaindex_feature.append(seqaaidx)
    return np.vstack(aaindex_feature)

hc_AAI_train_ =toFeatureAAindex(inFastaTrain)
hc_AAI_test_ =toFeatureAAindex(inFastaTest)

"""RF selector"""
#LGBM selector
def LGBM_SelectFeatures(X,y,i):#Light Gradient Boosting Machine Feature Selection

    model = LGBMClassifier(num_leaves=32,n_estimators=888,max_depth=12,learning_rate=0.16,min_child_samples=50,random_state=2020,n_jobs=8)
    model.fit(X, y)
    importantFeatures = model.feature_importances_
    K = importantFeatures.argsort()[::-1][:i]
    LGB_ALL_K=X[:,K]

    return  LGB_ALL_K

scaler = MinMaxScaler()
hc_AAI_train = scaler.fit_transform(hc_AAI_train_)
hc_AAI_test = scaler.transform(hc_AAI_test_)

ADC_train=np.c_[hc_AAC_train,hc_DPC_train,hc_CKS_train]
ADC_test=np.c_[hc_AAC_test,hc_DPC_test,hc_CKS_test]

def pca(X,k):
  X = X - X.mean(axis = 0)
  X_cov = np.cov(X.T, ddof = 0)
  eigenvalues, eigenvectors = eig(X_cov)
  klarge_index = eigenvalues.argsort()[-k:][::-1]
  k_eigenvectors = eigenvectors[klarge_index]
  return np.dot(X,k_eigenvectors.T)


hc_train = np.c_[hc_AAC_train,hc_DPC_train,hc_CKS_train,hc_AAI_train]
hc_test= np.c_[hc_AAC_test,hc_DPC_test,hc_CKS_test,hc_AAI_test]

hc_train=LGBM_SelectFeatures(hc_train ,AOP_y_train_,200)
hc_test=LGBM_SelectFeatures(hc_test ,AOP_y_test_,200)

print("融合")
from sklearn.model_selection import StratifiedKFold

# 定义K折交叉验证的折数
k = 5

# 创建K折交叉验证的拆分器
kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)

# 初始化空列表来存储每个折的训练和验证结果
train_acc_list = []
val_acc_list = []
train_loss_list = []
val_loss_list = []

# 进行K折交叉验证
for i, (train_index, val_index) in enumerate(kfold.split(x_train_, AOP_y_train_)):
    # 根据索引拆分训练集和验证集
    X_train, X_val = x_train_[train_index], x_train_[val_index]
    y_train, y_val = AOP_y_train_[train_index], AOP_y_train_[val_index]
    HC_train, HC_val = hc_train[train_index], hc_train[val_index]

    # 建立了一个基于BiLSTM的模型，结合了新的特征
    """Bilstm + new features"""
    vocab_size = 22
    embedding_dim = 100

    main_input = Input((maxlen,), dtype='int32', name='main_input')
    x = Embedding(vocab_size, embedding_dim, input_length=maxlen, trainable=True)(main_input)
    lstm_out = Bidirectional(LSTM(64))(x)
    aux_len = HC_train.shape[1]
    aux_input = Input((aux_len,), name='aux_input')
    x = layers.concatenate([lstm_out, aux_input])
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    main_output = Dense(1, activation='sigmoid', name='main_output')(x)
    model = Model(inputs=[main_input, aux_input], outputs=main_output)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    # train the model
    hist = model.fit(x={'main_input': X_train, 'aux_input': HC_train}, y=y_train,
                     # class_weight = class_weights_d,
                     epochs=40,
                     verbose=True,
                     validation_data=([X_val, HC_val], y_val),
                     batch_size=32).history

    # 在训练集和验证集上评估模型
    _, train_acc = model.evaluate(x={'main_input': X_train, 'aux_input': HC_train}, y=y_train, verbose=False)
    _, val_acc = model.evaluate(x={'main_input': X_val, 'aux_input': HC_val}, y=y_val, verbose=False)

    # 保存模型
    joblib.dump(model, r'C:\Users\41822\Desktop\DL\moxin\model_fold_{}.joblib'.format(i+1))
    # 将训练集和验证集的准确率添加到列表中
    train_acc_list.append(hist['accuracy'])
    val_acc_list.append(hist['val_accuracy'])
    train_loss_list.append(hist['loss'])
    val_loss_list.append(hist['val_loss'])
# 创建包含所有折的准确率数据的DataFrame
accuracy_data = {'Epoch': range(1, len(train_acc_list[0]) + 1)}
for i in range(k):
    accuracy_data['训练集准确率_{}'.format(i+1)] = train_acc_list[i]
    accuracy_data['验证集准确率_{}'.format(i+1)] = val_acc_list[i]
accuracy_df = pd.DataFrame(accuracy_data)

# 创建包含所有折的损失数据的DataFrame
loss_data = {'Epoch': range(1, len(train_loss_list[0]) + 1)}
for i in range(k):
    loss_data['训练集损失_{}'.format(i+1)] = train_loss_list[i]
    loss_data['验证集损失_{}'.format(i+1)] = val_loss_list[i]
loss_df = pd.DataFrame(loss_data)

# 将准确率数据和损失数据保存到Excel文件
accuracy_df.to_excel(r'C:\Users\41822\Desktop\DL\moxin\accuracy_results.xlsx', index=False)
loss_df.to_excel(r'C:\Users\41822\Desktop\DL\moxin\loss_results.xlsx', index=False)

# 打印每个折的训练集和验证集准确率
for i in range(k):
    print("Fold {}: Training Accuracy: {:.4f}, Validation Accuracy: {:.4f}".format(i+1, train_acc_list[i][-1], val_acc_list[i][-1]))

# 计算平均准确率
avg_train_acc = sum(train_acc_list[i][-1] for i in range(k)) / k
avg_val_acc = sum(val_acc_list[i][-1] for i in range(k)) / k
print("Average Training Accuracy: {:.4f}".format(avg_train_acc))
print("Average Validation Accuracy: {:.4f}".format(avg_val_acc))
# 循环加载和测试模型
for fold in range(1, 6):
 # 加载模型
 file_path = r'C:\Users\41822\Desktop\DL\moxin\model_fold_{}.joblib'.format(fold)
 model1 = joblib.load(filename=file_path)
 #测试集测试
 predictions = model1.predict(x={'main_input': x_test_, 'aux_input': hc_test})
 rounded_predictions = (predictions > 0.5).astype(int)
 #评价指标
 accuracy = accuracy_score(AOP_y_test_, rounded_predictions)
 precision = precision_score(AOP_y_test_, rounded_predictions)
 confusion_mat = confusion_matrix(AOP_y_test_, rounded_predictions)
 tn, fp, fn, tp = confusion_mat.ravel()
 sensitivity = tp / (tp + fn)
 specificity = tn / (tn + fp)
 mcc = matthews_corrcoef(AOP_y_test_, rounded_predictions)
 auc = roc_auc_score(AOP_y_test_, rounded_predictions)
 print("Model {}: ".format(fold))
 print("Testing Accuracy:  {:.4f}".format(accuracy))
 print("Precision: {:.4f}".format(precision))
 print("Sensitivity: {:.4f}".format(sensitivity))
 print("Specificity: {:.4f}".format(specificity))
 print("MCC: {:.4f}".format(mcc))
 print("AUC: {:.4f}".format(auc))
 print()