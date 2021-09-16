# Program Vector Embeddings

The folder is composed of four scripts:

- `AST_gen.sh`: Transform source code into JSON ASTs
- `Node_gen.sh`: Work with JSON ASTs and generate node embeddings
- `Program_embed.py`: Generate program embeddings from node embeddings
- `Clustering.py`: An experiment for clustering program embeddings and visualizing the clustering result


## Implementation

This repository is the `TensorFlow 1.13.1` implementation for generating program vector embeddings, and it should be used with `Python 3, JDK >= 8, sbt >= 1.0`.

Type the following:

```
cd ./AST_generation
pip install .

cd ../Node_generation
sbt assembly

cd ../Node_embedding
pip install .
python setup.py install
```


## Usage

### Step 1. AST generation

Run:

```
./AST_gen.sh
```

**N.B.** you may need to change the mode of this shell by running:
```
chmod 777 AST_gen.sh
```

then `asts.json` will be generated, which contains a JSON formatted AST per line.

### Step 2. Node generation
Run:
```
./Node_gen.sh
```
**N.B.** before running the shell, you may need to change the mode of this shell **and** `bigcode-embeddings` by running:
```
chmod 777 Node_gen.sh
chmod 777 Node_embedding/bin/bigcode-embeddings
```

then `embeddings.txt` will be generated, which contains node embeddings per line.

### Step 3. Program embedding
Run:
```
python Program_embed.py
```

then `program_embedding.txt` will be generated, which contains the program embeddings.

In `Program_embed.py`:

- `node()`: compute the total number of nodes under the root

- `embed_ast()`: compute the node embeddings recursively

### Experiment: Clustering the program vectors
Run:

```
python Clustering.py
```

an image named `cluster2d.png` will be generated, which represents the clustering result.


In `Clustering.py`:

- `assign_clusters()`: kmeans cluster process, which group the program vectors into six categories

- `reduce_dimensions()`: reduce program vector dimensions from 10 to 2 to make visualization available

- `create_scatter_plot()`: create scatter figure which has six colors representing six challenges

- `visualize_clusters()`: make the clustering result displayed in a 2D image
