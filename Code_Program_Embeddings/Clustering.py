import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


print("+-+-+-+-+-+-+")
print("Program vectors are clustering!")
print("+-+-+-+-+-+-+")

SVG_COLORS = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
]

def sanitize_data(embeddings, labels):
    norms = np.linalg.norm(embeddings, axis=1)
    valid_indexes = np.abs(norms - np.mean(norms)) <= (np.std(norms) * 2)
    sanitized_embeddings = embeddings[valid_indexes]
    sanitized_labels = labels.iloc[labels.index[valid_indexes]].reset_index(drop=True)
    return (sanitized_embeddings, sanitized_labels)

def reduce_dimensions(embeddings, dimensions=2):
    pca = PCA(n_components=dimensions)
    return pca.fit_transform(embeddings)

def assign_clusters(embeddings, labels, clusters_count):
    clusters = KMeans(n_clusters=clusters_count, max_iter=100000).fit_predict(embeddings)
    labels["Cluster"] = clusters
    print(clusters)

def compute_clusters_count(labels):
    return labels.Cluster.max()

def create_scatter_plot(embeddings_2d, labels, output=None):
    clusters_count = compute_clusters_count(labels)
    fig = plt.figure(figsize=(20, 20))
    axis = fig.add_subplot(111)
    for i in range(clusters_count):
        indexes = labels[labels.Cluster == i].index.values
        embeddings_3d=[]
        label1 = "C-"+str(i+1)
        for j in indexes:
            x=embeddings_2d[j, 0]
            y=embeddings_2d[j, 1]
            a=0
            for item in embeddings_2d:
                if item[0]==x and item[1]==y:
                    a+=1
            embeddings_3d.append([x,y,a-1])
        axis.scatter(embeddings_3d[0][0], embeddings_3d[0][1], s=205,c="C{0}".format(i),label=label1)
        for item in embeddings_3d[1:]:
            if item[2]>100:
                axis.scatter(item[0], item[1], s=5*item[2],c="C{0}".format(i))
            else:
                axis.scatter(item[0], item[1], s=250,c="C{0}".format(i))
    axis.legend(fontsize=30,loc="upper center",ncol=3,columnspacing=1)
    if output:
        fig.savefig(output)
    else:
        plt.show()

def load_labels(labels_path):
    return pd.read_csv(labels_path, sep="\t")

def visualize_clusters(labels_path,model_path,clusters_count):
    labels = load_labels(labels_path)
    #print(labels)
    embeddings=[]
    fr=open(model_path)
    for line in fr.readlines():
        curLine=line.strip().split(' ')
        fltLine=list(map(float,curLine))
        embeddings.append(fltLine)
    embeddings=np.array(embeddings)
    embeddings, labels = sanitize_data(embeddings, labels)
    assign_clusters(embeddings, labels, clusters_count)
    embeddings_2d = reduce_dimensions(embeddings)
    create_scatter_plot(embeddings_2d, labels, "cluster2d")

if __name__ == '__main__':
    asts = open("./AST_generation/asts.txt")
    lines = asts.readlines()
    names=[]
    for item in lines:
        names.append(item.split('/')[-1][:-1])
    #print(names)
    file2 = open("labels.tsv", 'w')
    file2.write("id"+"\t"+"type"+"\n")
    index=0
    for item in names:
        file2.write(str(index)+"\t"+item+"\n")
        index+=1
    file2.close()
    visualize_clusters("labels.tsv","Program_Vector_Embeddings.txt",7)

print("+-+-+-+-+-+-+")
print("success!")
print("+-+-+-+-+-+-+")