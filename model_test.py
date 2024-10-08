import numpy as np
from transformers import TFBertModel
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import roc_auc_score
from model import DeepIndel

def calculate_pearson(y_true, y_pred):
    pearson = pearsonr(y_true, y_pred)[0]
    return pearson

def calculate_spearman(y_true, y_pred):
    spearman = spearmanr(y_true, y_pred)[0]
    return spearman

def calculate_ktau(y_true, y_pred):
    ktau = kendalltau(y_true, y_pred)
    return ktau

def calculate_auc(y_true, y_pred):
   Y_test_median = np.median(y_true)
   Y_test_binary = np.where(y_true>=Y_test_median,1.0,y_true)
   Y_test_binary = np.where(y_true<Y_test_median,0.0,y_true)
   auc = roc_auc_score(Y_test_binary,y_pred)
   return auc

model_name = "./bert-base-uncased"
bert_model = TFBertModel.from_pretrained(model_name)
config = bert_model.config
config.vocab_size = 86
model = DeepIndel(bert_model)
model.load_weights('./weight/DeepIndel (K562).h5')

y_pred = model.predict([x_test])
DelF_auc, DelF_scc, DelF_pcc, DelF_ktau = calculate_auc(y_test[:,0],y_pred[:,0]), calculate_spearman(y_test[:,0],y_pred[:,0]), calculate_pearson(y_test[:,0],y_pred[:,0]), calculate_ktau(y_test[:,0],y_pred[:,0])
InsF1_auc, InsF1_scc, InsF1_pcc, InsF1_ktau = calculate_auc(y_test[:,1],y_pred[:,1]), calculate_spearman(y_test[:,1],y_pred[:,1]), calculate_pearson(y_test[:,1],y_pred[:,1]), calculate_ktau(y_test[:,1],y_pred[:,1])
DelF1_auc, DelF1_scc, DelF1_pcc, DelF1_ktau = calculate_auc(y_test[:,2],y_pred[:,2]), calculate_spearman(y_test[:,2],y_pred[:,2]), calculate_pearson(y_test[:,2],y_pred[:,2]), calculate_ktau(y_test[:,2],y_pred[:,2])
FsF1_auc, FsF1_scc, FsF1_pcc, FsF1_ktau = calculate_auc(y_test[:,3],y_pred[:,3]), calculate_spearman(y_test[:,3],y_pred[:,3]), calculate_pearson(y_test[:,3],y_pred[:,3]), calculate_ktau(y_test[:,3],y_pred[:,3])
FsF2_auc, FsF2_scc, FsF2_pcc, FsF2_ktau = calculate_auc(y_test[:,4],y_pred[:,4]), calculate_spearman(y_test[:,4],y_pred[:,4]), calculate_pearson(y_test[:,4],y_pred[:,4]), calculate_ktau(y_test[:,4],y_pred[:,4])
FsF_auc, FsF_scc, FsF_pcc, FsF_ktau = calculate_auc(y_test[:,5],y_pred[:,5]), calculate_spearman(y_test[:,5],y_pred[:,5]), calculate_pearson(y_test[:,5],y_pred[:,5]), calculate_ktau(y_test[:,5],y_pred[:,5])