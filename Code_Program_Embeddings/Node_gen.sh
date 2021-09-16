#!/bin/bash
echo "+-+-+-+-+-+-+-+-+-+-+-+-+-+"
echo "Node vectors are generating..."
echo "+-+-+-+-+-+-+-+-+-+-+-+-+-+"
#  Node_generation
cd Node_generation/target/scala-2.12
./asttools-assembly-0.1.jar generate-vocabulary --strip-identifiers -o vocabulary-no-ids.tsv ../../../AST_generation/asts.json
./asttools-assembly-0.1.jar generate-skipgram-data -v vocabulary-no-ids.tsv --ancestors-window-size 2 --children-window-size 0 --without-siblings -o skipgram-data ../../../AST_generation/asts.json
cd ../../..

# Node_embedding
cd Node_embedding/bin
./bigcode-embeddings train -o simple-embeddings --vocab-size=$(tail -n+2 ../../Node_generation/target/scala-2.12/vocabulary-no-ids.tsv | wc -l) --emb-size=10 --optimizer=gradient-descent --batch-size=20 --epochs=100 ../../Node_generation/target/scala-2.12/skipgram-data*

# get the MAX num of 'embeddings.*.index
MAX="0"
for file in ./simple-embeddings/*
do
	if [[ $file == *".index" ]]; then
		index=${file##*-}
		if [ ${index%.*} -gt $MAX ]; then
			MAX=${index%.*}
		fi
	fi
done

./bigcode-embeddings export -o embeddings.txt simple-embeddings/embeddings.bin-$MAX
cd ../..
echo "+-+-+-+-+-+-+"
echo "success!"
echo "+-+-+-+-+-+-+"
