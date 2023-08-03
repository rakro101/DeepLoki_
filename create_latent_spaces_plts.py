import pandas as pd
import numpy as np

if __name__ == "__main__":
    df_train_val = pd.read_csv('output/update_allcruises_df_validated_5with_zoomie_20230727.csv',sep=";")
    df_test =df_train_val
    #df_train_val.to_csv('output/view_all_data03032023.csv', sep=";")
    df_test = pd.read_csv("output/update_wo_artefacts_test_dataset_PS992_20230727_nicole.csv",sep=";")
    df_test = df_test[['root_path', 'img_file_name', 'label']]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_test, df_test['label'], stratify=df_test['label'], test_size =0.95, random_state=42)
    df_test =X_test

    from torchvision import transforms
    transform = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Resize(size=224),  # paper was 224
                  transforms.CenterCrop(size=224),
                  transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
            ])

    from dtl_model import DtlModel
    import torch
    import os
    from PIL import Image
    from torchvision import transforms
    from loki_datasets import LokiDataModule, LokiTrainValDataset
    dm = LokiDataModule(batch_size=1024)#1024
    lrvd = LokiTrainValDataset()
    num_classes = lrvd.n_classes
    label_encoder = lrvd.label_encoder
    m_arch = "resnet_dino450_latent"
    model = DtlModel(input_shape=(3,300,300), num_classes=num_classes, arch=m_arch, label_encoder=lrvd)
    #c_path ="lightning_logs/version_60/checkpoints/epoch=2-step=912.ckpt" #paperversion
    #c_path = "lightning_logs/version_68/checkpoints/epoch=0-step=608.ckpt"
    #checkpoint = torch.load(c_path)
    #model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    #convert_tensor = transforms.ToTensor()
    convert_tensor = transform
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
    top_n = 10
    top_5_df =df_test[df_test['label'].isin(df_sum.index[:top_n])]
    #top_5_df = top_5_df[top_5_df['label']!="Detritus"]
    top_5_df = top_5_df[top_5_df['label']!="Artefact"]

    from sklearn.manifold import TSNE
    V = top_5_df['latent_space'].values.reshape(-1,1)
    print(V.shape)
    #V = np.reshape(V, (207,1))
    L = np.array([V[i][0] for i in range(len(V))])
    print(L.shape)
    V_embedded = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=100).fit_transform(L)
    V_embedded
    V_embedded.shape

    top_5_df['x_tsne'] =V_embedded[:,0]
    top_5_df['y_tsne'] =V_embedded[:,1]
    import seaborn as sns
    import colorcet as cc
    sns.set(rc={'figure.figsize':(16,16)})
    #palette = sns.color_palette("bright", 40)
    palette=sns.color_palette("bright",n_colors=top_n)#sns.color_palette(cc.glasbey, n_colors=top_n)
    sns_plot_tsne =sns.scatterplot(data=top_5_df,x='x_tsne', y='y_tsne',  hue='label', legend='full',palette=palette )#palette=sns.color_palette("cubehelix", as_cmap=True)
    sns_plot_tsne.tick_params(axis='x', labelsize=32)  # Set x-axis tick label size
    sns_plot_tsne.tick_params(axis='y', labelsize=32)  # Set y-axis tick label size
    sns_plot_tsne.set_xlabel('X-Axis Label', fontsize=32)  # Set x-axis label size
    sns_plot_tsne.set_ylabel('Y-Axis Label', fontsize=32)
    fig = sns_plot_tsne.get_figure()
    fig.savefig(f"tsne_umap_plots/42_A_{m_arch}_{top_n}_testdata95_tsne_plot.png")
    fig.show()
    import umap
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    reducer = umap.UMAP(n_neighbors=15)
    scaled_data = StandardScaler().fit_transform(L)

    U_embedding = reducer.fit_transform(scaled_data)
    print(U_embedding.shape)
    palette=sns.color_palette("bright",n_colors=top_n)

    top_5_df['x_umap'] =U_embedding[:, 0]
    top_5_df['y_umap'] =U_embedding[:, 1]
    sns_plot_umap = sns.scatterplot(data=top_5_df,x='x_umap', y='y_umap',  hue='label', legend='full',palette=palette )
    sns_plot_umap.tick_params(axis='x', labelsize=32)  # Set x-axis tick label size
    sns_plot_umap.tick_params(axis='y', labelsize=32)  # Set y-axis tick label size
    sns_plot_umap.set_xlabel('x_umap', fontsize=32)  # Set x-axis label size
    sns_plot_umap.set_ylabel('y_umap', fontsize=32)
    fig2 = sns_plot_umap.get_figure()
    fig2.savefig(f"tsne_umap_plots/42_A_{m_arch}_{top_n}_testdata95_umap_plot.png")
    fig2.show()
    top_5_df.to_csv("tsne_umap_plots/42_A_tsen_umap_embeddings.csv")
    #plt.scatter(U_embedding[:, 0], U_embedding[:, 1], c=palette, cmap='Spectral', s=5)
    #plt.gca().set_aspect('equal', 'datalim')
    #plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    #plt.title('UMAP projection of the Loki dataset', fontsize=24)