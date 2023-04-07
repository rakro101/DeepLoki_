import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dtl_model_wo_linear_layer import DtlModel
import seaborn as sns
import colorcet as cc
import torch
import os
from PIL import Image
from torchvision import transforms
from loki_datasets import LokiDataModule, LokiTrainValDataset
df_train_val = pd.read_csv('output/update_allcruises_df_validated_5with_zoomie_20230303.csv')
df_test =df_train_val
#df_train_val.to_csv('output/view_all_data03032023.csv', sep=";")
#df_test = pd.read_csv("output/update_wo_artefacts_test_dataset_PS992_03032023.csv")
df_test = df_test[['root_path', 'img_file_name', 'label']]
X_train, X_test, y_train, y_test = train_test_split(df_test, df_test['label'], stratify=df_test['label'], test_size =0.25)
df_test =X_test

dm = LokiDataModule(batch_size=1024)#1024
lrvd = LokiTrainValDataset()
num_classes = lrvd.n_classes
label_encoder = lrvd.label_encoder
model = DtlModel(input_shape=(3,300,300), num_classes=35, arch="resnet_dino450", label_encoder=lrvd)
model.eval()

convert_tensor = transforms.ToTensor()

def get_pred(col1, col2, model=model):
    image_root = col1
    image_path = col2
    img_path = os.path.join(image_root, image_path)
    image = Image.open(img_path).convert('RGB')
    temp_pred = model(convert_tensor(image).unsqueeze(0))
    return temp_pred.detach().numpy().squeeze(0)

df_test['latent_space']= df_test.apply(lambda x: get_pred(x.root_path, x.img_file_name,model), axis=1)
df_test['count']=1
df_sum = df_test.groupby('label').agg(sum)
df_sum =df_sum.sort_values('count', ascending=False)
top_n = 6
top_5_df =df_test[df_test['label'].isin(df_sum.index[:top_n])]
top_5_df = top_5_df[top_5_df['label']!="Detritus"]
from sklearn.manifold import TSNE
V = top_5_df['latent_space'].values.reshape(-1,1)
#V = np.reshape(V, (207,1))
L = np.array([V[i][0] for i in range(len(V))])
print(L.shape)
V_embedded = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=100).fit_transform(L)
print(V_embedded)
print(V_embedded.shape)
top_5_df['x'] =V_embedded[:,0]
top_5_df['y'] =V_embedded[:,1]

sns.set(rc={'figure.figsize':(16,16)})
#palette = sns.color_palette("bright", 40)
palette=sns.color_palette("dark",n_colors=top_n)#sns.color_palette(cc.glasbey, n_colors=top_n)
sns_plot = sns.scatterplot(data=top_5_df,x='x', y='y',  hue='label', legend='full',palette=palette )#palette=sns.color_palette("cubehelix", as_cmap=True)
fig = sns_plot.get_figure()
fig.savefig("tsne_plot_resnet_450.png")


import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
reducer = umap.UMAP()
scaled_data = StandardScaler().fit_transform(L)

U_embedding = reducer.fit_transform(scaled_data)
print(U_embedding.shape)

plt.scatter(U_embedding[:, 0], U_embedding[:, 1], c=top_5_df['labels'].values, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the Loki dataset', fontsize=24)