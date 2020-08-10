# code-embeddings
The ability to generate natural language from source code is an open research topic and has gained an increasing popularity in recent years. Due to the nature of open research topics, there is no silverbullet in solving this problem, meaning there are different promising approaches being explored by the research community.

This specific work is heavily inspired by [Uri Alon](http://urialon.cswp.cs.technion.ac.il), [Shaked Brody](http://www.cs.technion.ac.il/people/shakedbr/), [Omer Levy](https://levyomer.wordpress.com) and [Eran Yahav](http://www.cs.technion.ac.il/~yahave/), "code2seq: Generating Sequences from Structured Representations of Code" [[PDF]](https://openreview.net/pdf?id=H1gKYo09tX) and relies partly on their unofficial implementation on GitHub [[Repository]](https://github.com/Kolkir/code2seq).

Please see the wiki for a more detailed explanation on the motivation, the design decisions and the evaluation of this project.

Table of Contents
=================
  * [Dataset](#dataset)
  * [Model](#model)

## Dataset
Please follow these steps to download and preprocess the py150 dataset.

1. Dowload and unarchive the parsed AST paths

```bash
wget http://files.srl.inf.ethz.ch/data/py150.tar.gz
tar -xzvf py150.tar.gz
```

2. Clone the code2seq repository

```bash
git clone https://github.com/Kolkir/code2seq.git
cd code2seq/Python150kExtractor
```

3. Extract the data

```bash
python extract.py --data_dir=<PATH_TO_PY150_FOLDER> --output_dir=<PATH_TO_EXTRACTED_FOLDER> --seed=239
```

4. Preprocess the data for training

```bash
./preprocess.sh <PATH_TO_EXTRACTED_FOLDER>
```

## Model
Once you have downloaded and preprocessed the dataset go back this repository.

1. Build and run the docker image in a container

```bash
docker build -t code-embeddings .
docker run --gpus all --rm -it -v <PATH_TO_PREPROCESSED_FOLDER>:/tmp/py150 -p 6006:6006 code-embeddings /bin/bash
```

2. Run the training script

```bash
python ./src/train.py \
--dict <PATH_TO_PREPROCESSED_DICT> \
--train <PATH_TO_PREPROCESSED_TRAIN> \
--test <PATH_TO_PREPROCESSED_TEST>
```

3. (Optional) Run tensorboard for better analysis of the training run

```bash
tensorboard --logs ./logs
```

4. Evaluate the script with some example functions

```bash
python ./src/train.py \
--dict <PATH_TO_PREPROCESSED_DICT> \
--data <PATH_TO_EXAMPLES> \
```
