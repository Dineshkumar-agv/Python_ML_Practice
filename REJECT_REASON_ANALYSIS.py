#!/usr/bin/env python
# coding: utf-8

# In[206]:


import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[207]:


os.chdir("D:\DATASETS_MACHINE_LEARNING_PRACTICE")


# In[364]:


df=pd.read_excel("Customer_Master-Frontend_reject_ACCOUNTS.xls")


# In[365]:


df.head()


# In[366]:


df1=df["REJECT_REASON"]
df1.head()


# In[367]:


df1.value_counts()


# In[ ]:





# In[33]:


sns.countplot(x=Cat_var,y=None,data=df1)


# In[368]:


Cat_var=df1.unique()


# In[ ]:





# In[369]:


for i in Cat_var:
    print(i)


# In[214]:


import nltk


# In[371]:


df1=df1.astype(str)


# In[372]:


df1=df1.apply(lambda x:x.lower())


# In[373]:


df1=df1.apply(lambda x:'reject' if x=='r' else x)


# In[374]:


df1=df1.apply(lambda x:'reject' if x=='rej' else x)


# In[375]:


df1.value_counts()


# In[220]:


cat1=df1.unique()


# In[ ]:





# In[221]:


for i in cat1:
    print(i)


# In[222]:


nltk.download()


# In[376]:


improper_words=['re','w','incompledte','incomplete appl','incomplete details','rejct','h','kk','k','l','dfd','dfe','sfgf','sefed','fght','t','f','y','jk','g','jj','uu','n','kj','pp','ok','df','d','fna','x','c','p','rt','u','ffffffffffffffff','j','lk','ntd','a','na','ggyj','e','oj','v','cv','vv','in','recect','s','o','nan','ol','rj','hh','mis','bv','bk','jhk','jl','po','bn','wr','c v','nem','pd','wronwrongttgg w','of','af','mna','invali','inv','b dade','nan s','ook','nr','dfr','hg','np','regt','ujh','nn','recet']


# In[377]:


improper_words


# In[378]:


df1=df1.apply(lambda x:'reject' if x in improper_words else x)


# In[58]:





# In[379]:


tech_error=['er','error','errr','errrr']
df1=df1.apply(lambda x: 'Technical Error' if x in tech_error else x)


# In[380]:


cat1=df1.unique()
for i in cat1:
    print(i)


# In[381]:


df1=df1.apply(lambda x: 'already' if 'alrady' in x else x)
df1=df1.apply(lambda x: 'duplicate' if 'dup' in x else x)
df1=df1.apply(lambda x: 'dob' if 'date of birth' in x else x)


# In[382]:


df1=df1.apply(lambda x: 'error' if 'err' in x else x)


# In[383]:


df1=df1.apply(lambda x: 'reject' if 'rejected' in x else x)
df1=df1.apply(lambda x: 'reject' if 'reje' in x else x)


# In[231]:


cat1=df1.unique()
for i in cat1:
    print(i)
    


# In[232]:


df2=df1


# In[384]:


df1=df1.apply(lambda x: 'Date of birth Issues' if 'dob' in x else x)


# In[385]:


df1=df1.apply(lambda x: 'FormIssues' if 'form' in x else x)


# In[386]:


df1=df1.apply(lambda x: 'Form related Issues' if 'aof' in x else x)


# In[387]:


df1=df1.apply(lambda x: 'Technical Error' if 'error' in x else x)


# In[388]:


df1=df1.apply(lambda x: 'PhotoMismatch' if 'photo' in x else x)


# In[389]:


df1=df1.apply(lambda x: 'Names mismatch' if 'name' in x else x)


# In[390]:


df1=df1.apply(lambda x: 'Title' if 'title' in x else x)
df1=df1.apply(lambda x: 'Signature mismatch' if 'sign' in x else x)


# In[391]:


df1=df1.apply(lambda x: 'AddressProof' if 'add' in x else x)


# In[392]:


df1=df1.apply(lambda x: 'AadhaarCardorNumber' if 'har' in x else x)
df1=df1.apply(lambda x: 'AadhaarCardorNumber' if 'haar' in x else x)


# In[393]:


df1=df1.apply(lambda x: 'IdentificationProofNumber' if 'id' in x else x)


# In[394]:


df1=df1.apply(lambda x: 'PhotoMismatch' if 'pic' in x else x)


# In[395]:


df1=df1.apply(lambda x: 'SignatureMismatch' if 'ature' in x else x)


# In[396]:


df1=df1.apply(lambda x: 'NomineeorWitness' if 'nomin' in x else x)
df1=df1.apply(lambda x: 'NomineeorWitness' if 'wit' in x else x)


# In[397]:


df1=df1.apply(lambda x: 'DocumentImproper' if 'doc' in x else x)


# In[398]:


df1=df1.apply(lambda x: 'PANNumberImproper' if 'pan' in x else x)


# In[399]:


df1=df1.apply(lambda x: 'KYC Issues' if 'kyc' in x else x)


# In[400]:


df1=df1.apply(lambda x: 'KYCIssues' if 'KYC Issues' in x else x)


# In[401]:


df1=df1.apply(lambda x: 'SignatureMismatch' if 'sig' in x else x)


# In[402]:


df1=df1.apply(lambda x: 'FormIssues' if 'Form related Issues' in x else x)
df1=df1.apply(lambda x: 'NamesMismatch' if 'Names mismatch' in x else x)
df1=df1.apply(lambda x: 'NamesMismatch' if 'Names mismatch' in x else x)
df1=df1.apply(lambda x: 'PhotoMismatch' if 'image' in x else x)
df1=df1.apply(lambda x: 'AccountExists' if 'ac' in x else x)
df1=df1.apply(lambda x: 'AccountExists' if 'ead' in x else x)


# In[403]:


df1=df1.apply(lambda x: 'DateOfBirthIssues' if 'Date of birth Issues' in x else x)
df1=df1.apply(lambda x: 'DateOfBirthIssues' if 'dt of' in x else x)


# In[404]:


df1=df1.apply(lambda x: 'TechnicalError' if 'fail' in x else x)
df1=df1.apply(lambda x: 'TechnicalError' if 'Technical Error' in x else x)


# In[405]:


df1=df1.apply(lambda x: 'PhotoMismatch' if 'Photograph related Issues' in x else x)


# In[406]:


df1=df1.apply(lambda x: 'ApplicationNotSubmittedByCSP' if 'ted by csp' in x else x)
df1=df1.apply(lambda x: 'ApplicationNotSubmittedByCSP' if 'application not' in x else x)
df1=df1.apply(lambda x: 'Title' if 'wrong salutation' in x else x)
df1=df1.apply(lambda x: 'Database' if 'db' in x else x)
    


# In[407]:


df1.value_counts()


# In[408]:


cat1=df1.unique()
for i in cat1:
    print(i)


# In[409]:


df1.to_csv("REJECT_REASON_ANALYSIS.csv")#df2 has values
df2=df1


# In[410]:


reason_list=df1.to_list()
reason_list


# In[411]:


dummy_reason_list=reason_list


# In[412]:


my_dict = {i:reason_list.count(i) for i in reason_list}
my_dict


# In[413]:


bes=sorted(my_dict.items(), key=lambda x: x[1], reverse=True)


# In[414]:


bes


# In[415]:


bes1 = []
bes2 = []
for i in bes:
   bes1.append(i[0])
   bes2.append(i[1])


# In[356]:





# In[416]:


df3 = pd.DataFrame(list(zip(bes1, bes2)),columns =['Reject Reason', 'Frequency']) 
df3.to_csv("Final Result.csv") 


# In[417]:


pip install wordcloud


# In[418]:


from wordcloud import WordCloud, STOPWORDS 


# In[419]:


comment_words = '' 
stopwords = set(STOPWORDS) 


# In[420]:


stopwords


# In[422]:


dfc=pd.read_csv("REJECT_REASON_ANALYSIS.csv")
dfc=dfc['REJECT_REASON']
dfc.head()


# In[426]:


for val in reason_list: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
      
    comment_words += " ".join(tokens)+" "


# In[427]:


comment_words


# In[479]:


wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 14,repeat=False,max_words=200,collocations=False).generate(comment_words)
# plot the WordCloud image    
import matplotlib.pyplot as plt1 
plt1.figure(figsize = (8, 7), facecolor = None) 
plt1.imshow(wordcloud) 
plt1.axis("off") 
plt1.tight_layout(pad = 0) 
  
plt1.show() 

