import json
import csv
import math
import numpy as np
vocab=[]
vec=[]
results=[]
pro_word=[]
idf={}
td_idf=[]

print("+-+-+-+-+-+-+")
print("Program is embedding!")
print("+-+-+-+-+-+-+")

csv_file=open('./Node_generation/target/scala-2.12/vocabulary-no-ids.tsv')
csv_reader_lines = csv.reader(csv_file)
for line in csv_reader_lines:
    line1=str(line[0]).split('\t')
    vocab.append(line1)

txt=open("./Node_embedding/bin/embeddings.txt")
lines=txt.readlines()
for line in lines:
    vec.append(np.array(line.split(' '),dtype=np.float64))

vocab1=vocab[1:]
with open('./AST_generation/asts.json', 'r', encoding="utf-8") as f:
    for jsonstr in f.readlines():
        jsonstr = json.loads(jsonstr)
        word=[]
        for l in range(len(jsonstr)):
            item=jsonstr[l]
            voc=item["type"]
            if voc not in word:
                word.append(voc)
        pro_word.append(word)


for i in range(len(pro_word)):
    for j in range(len(pro_word[i])):
        if pro_word[i][j] not in idf:
            idf[pro_word[i][j]]=1
        else:
            idf[pro_word[i][j]]+=1

cnt=0
for i in range(len(vocab1)):
    count=vocab1[i][3]
    cnt+=int(count)


for i in range(len(vocab1)):
    tf=int(vocab1[i][3])/cnt
    idf1=math.log(len(pro_word)/(1+idf[vocab1[i][1]]),10)
    r=tf*idf1
    td_idf.append(r)


direct_child={}
all_node={}
dictionary = {}

def node(num):
    if direct_child[num]==0:
        return 1
    else:
        r=0
        for i in range(len(direct_child[num])):
            r+=node(direct_child[num][i])
        return r+1

def embed_ast(num,type):
    if len(direct_child[num])==0:
        return np.tanh(vec[type]*math.exp(td_idf[type]))
    else:
        embed1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        for i in range(len(direct_child[num])):
            embed1+=all_node[direct_child[num][i]]/all_node[num]*embed_ast(direct_child[num][i],dictionary[direct_child[num][i]])
        embed1*=(all_node[num]-1)/all_node[num]
        embed1+=(1/all_node[num])*vec[type]*math.exp(td_idf[type])
        return np.tanh(embed1)

with open('./AST_generation/asts.json', 'r', encoding="utf-8") as f:
    for jsonstr in f.readlines():
        jsonstr = json.loads(jsonstr)
        direct_child = {}
        all_node = {}
        dictionary = {}
        for l in range(len(jsonstr)):
            item=jsonstr[l]
            id = item["id"]
            voc = item["type"]
            for i in range(len(vocab1)):
                if vocab1[i][1] == voc:
                    dictionary[id] = i
            if 'children' in item.keys():
                child = item["children"]
                direct_child[l] = child
            else:
                direct_child[l] = []
        for j in range(len(jsonstr)):
            all_node[j]=node(j)
        for k in range(len(jsonstr)):
            root_embed=embed_ast(k,dictionary[k])
            results.append(root_embed)
            break

with open('./AST_generation/asts.txt', 'r', encoding="utf-8") as fileName:
    with open('Program_Vector_Embeddings.CSV','w',encoding='utf-8-sig',newline='') as csvFile:
        tmp = fileName.readlines()
        fileList=[]
        for _ in tmp:
            fileList.append(_)

        writer = csv.writer(csvFile)
        for i in range(len(fileList)):
            tmp=[]
            tmp.append(fileList[i].split('/')[-1].split('\n')[0])
            for _ in range(len(results[i])):
                tmp.append(results[i][_])
            writer.writerow(tmp)

file2 = open("Program_Vector_Embeddings.txt",'w')
for i in range(len(results)):
    for j in range(len(results[i])):
        file2.write(str(results[i][j])+' ')
    file2.write('\n')

print("+-+-+-+-+-+-+")
print("success!")
print("+-+-+-+-+-+-+")