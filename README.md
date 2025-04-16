# Tagging Knowledge Concepts for Math Problems Based on Multi-label Text Classification
The paper is published on [Expert Systems with Applications](https://www.sciencedirect.com/science/article/abs/pii/S0957417424030999?via%3Dihub).
The supplementary materials include: the dataset of the experiment, the code for model training, 
and the data of experimental results.

## Dataset
We conduct experiments on four datasets. 

Three of them are open-source:
+ `TAL-SAQ7K-CN`: https://ai4ed.cc/competitions/aaai2024competition.
+ `CCNU`: https://github.com/ccnu-edm/PQSCT.
+ `GeoQA`: https://github.com/chen-judge/GeoQA.

We also establish a real-world Chinese dataset named DA-20k from the website http://tiku.zujuan.com/.
The website is a professional K12 Chinese test questions website, the data and labels have been verified by human experts. DA-20k consists of 22,498 high school math problems including text 
information and mathematical expressions, as well as 427 corresponding categories of knowledge concepts which can be 
organized into 16 parent categories of knowledge concepts. We save the data on an anonymous website: figshare.com.
The original data are saved on website: [DA-20k(original)](https://figshare.com/s/2be2eb2c06d00a9e4349) (https://figshare.com/s/2be2eb2c06d00a9e4349). 

We preprocess the data and save them on webset: [Preprocessed data for automatic tagging of knowledge concepts](https://figshare.com/s/16c87fb0e3b73037035a) (https://figshare.com/s/16c87fb0e3b73037035a).

You can download the Dataset and save them to the `/file_data` folder. An ideal example of directory `file_data` is like:

```text
.
├── file_data
│   ├── DA-20k
│   │   ├── label2index.pkl
│   │   ├── math_data_roberta.h5
│   │   ├── math_data_roberta_latex.h5
│   │   ├── math_data_roberta_delete.h5
│   ├── GeoQA
│   │   ├── label2index.pkl
│   │   ├── math_data_roberta.h5
│   ├── TAL-SAQ7K-CN
│   │   ├── ...
│   ├── CCNU
│   │   ├── ...
```

There are two kinds of files in each folder: `math_data.h5` and `label2index.pkl`.

+ `math_data.h5` saves a dictionary.
```Python
{
    'input_ids': [[1,12,233,10002,0,...], [...], ...], # The input_ids of questions, each item is a word_index sequence of the question, the sequence length is 150
    'attention_mask': [[1,1,1,1,1,0,...], [...], ...], # The attention_mask of questions
    'label_list': [[1,1,0,1,0,0,...], [...], ...], # A set of multi-hot coding, each item is the knowledge concepts corresponding to the question
}
```
+ `label2index.pkl` saves the dictionary of label_id to label_index in multi-hot coding.

Questions in DA-20k, TAL-SAQ7K-CN and CCNU include complex formulas. There are three different formula preprocessing methods corresponding to three `math_data.h5` files:

+ `math_data_roberta.h5`: Formulas are transformed into plain text and cut as general texts. (Formula-C)
+ `math_data_roberta_latex.h5`: Formulas are transformed into LaTeX format. (Formula-L)
+ `math_data_roberta_delete.h5`: Formulas are deleted. (Formula-D)

## Requirements
You can see the requirements in `requirements.txt`.

## Run the experiments

You can run the codes below to run the experiment.
```bash
cd MathByte
python classification.py -h # review the command parameters
python classification.py
python classification.py --use_lcm=True # use label smoothing only, LBS model
python classification.py --use_att=True # use label attention only, LHAB model
python classification.py --use_lcm=True --use_att=True # use ls & la, LHABS model
python classification.py --config=config/config_DA20k_cut.yml # specify the configuration file, DA-20k with Formula-C, can be replaced by TAL-SAQ7K-CN or CCNU
python classification.py --config=config/config_DA20k_latex.yml # specify the configuration file, DA-20k with Formula-L, can be replaced by TAL-SAQ7K-CN or CCNU
python classification.py --config=config/config_DA20k_delete.yml # specify the configuration file, DA-20k with Formula-D, can be replaced by TAL-SAQ7K-CN or CCNU
python classification.py --config=config/config_GeoQA.yml # specify the configuration file, GeoQA
```
