import PyPluMA
import PyIO
from tqdm import tqdm
import pickle

class SignificantComplexPlugin:
 def input(self, inputfile):
       self.parameters = PyIO.readParameters(inputfile)
 def run(self):
        pass
 def output(self, outputfile):
  stat_feature_attn_dict = dict()

  myAttn = open(PyPluMA.prefix()+"/"+self.parameters["attn"], "rb")
  myNN = open(PyPluMA.prefix()+"/"+self.parameters["nn"], "rb")
  attn_df_test = pickle.load(myAttn)
  score_df = pickle.load(myNN)
  for feature in tqdm(attn_df_test['feature'].unique()):
    stat_feature_attn_dict[feature]=dict()
    tmp_df = attn_df_test[attn_df_test['feature']==feature]
    stat_feature_attn_dict[feature]['mean'] = tmp_df['attn'].mean()
    stat_feature_attn_dict[feature]['std'] = tmp_df['attn'].std()
  stat_feature_attn_dict


# ### Iterate though each complex and plot attention if value is significan
#

# In[32]:


  for index, row in score_df.iterrows():
    ppi_native, ppi_neg = row['PPI_native'], row['PPI_neg']

    signif_feat_pos = []
    signif_feat_neg = []
    attention_dict_pos = {}
    attention_dict_neg = {}
    for feature in attn_df_test['feature'].unique():
        tmp_df = attn_df_test[attn_df_test['feature']==feature]
        tmp_df_neg = tmp_df[tmp_df['PPI']==ppi_neg].reset_index(drop=True)
        tmp_df_pos = tmp_df[tmp_df['PPI']==ppi_native].reset_index(drop=True)

        attn_neg = tmp_df_neg.loc[0,:]['attn']
        attn_pos = tmp_df_pos.loc[0,:]['attn']

        zscore_neg = (attn_neg-stat_feature_attn_dict[feature]['mean'])/stat_feature_attn_dict[feature]['std']
        zscore_pos = (attn_pos-stat_feature_attn_dict[feature]['mean'])/stat_feature_attn_dict[feature]['std']

        if zscore_neg>1.96:
            signif_feat_neg.append(feature)
        if zscore_pos>1.96:
            signif_feat_pos.append(feature)

        attention_dict_pos[feature] = attn_pos#zscore_pos#attn_pos
        attention_dict_neg[feature] = attn_neg#zscore_neg#attn_neg

    score_native = row['PIsToN_native']
    score_incorrect = row['PIsToN_incorrect']

    if len(signif_feat_pos)>0 and len(signif_feat_neg)>0:
        print(f"{row['PPI_native']}: {signif_feat_pos} {row['PPI_neg']}: {signif_feat_neg}; score native: {score_native}; score incorrect: {score_incorrect}")
        print(f"attn pos: {attention_dict_pos}; attn neg: {attention_dict_neg}")
        print("")

