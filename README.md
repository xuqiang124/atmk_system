# Tagging Knowledge Concepts for Math Problems Based on Multi-label Text Classification
The supplementary materials include: the dataset of the experiment, the code for model training, 
and the data of experimental results.

## Dataset
We conduct experiments on two datasets. 

The first one is an open-source dataset named GeoQA: https://github.com/chen-judge/GeoQA.

We also establish a real-world Chinese dataset named DA-20k from the website http://tiku.zujuan.com/.
The website is a professional K12 Chinese test questions website, the data and labels have been verified by human experts. DA-20k consists of 22,498 high school math problems including text 
information and mathematical expressions, as well as 427 corresponding categories of knowledge concepts which can be 
organized into 15 parent categories of knowledge concepts. We save the data on an anonymous website: figshare.com.
The original data are saved on website: [DA-20k(original)](https://figshare.com/s/2be2eb2c06d00a9e4349) (https://figshare.com/s/2be2eb2c06d00a9e4349). 

We preprocess the data and save them on webset: [Preprocessed data for automatic tagging of knowledge concepts](https://figshare.com/s/7257c04bdf3ba689b126) (https://figshare.com/s/7257c04bdf3ba689b126).

You can download the Dataset and save them to the `/file_data` folder. An ideal example of directory `file_data` is like:

```text
.
├── file_data
│   ├── DA-20k
│   │   ├── embeddings.pkl
│   │   ├── vocab_label.pkl
│   │   ├── math_data_cut.h5
│   │   ├── math_data_delete.h5
│   │   ├── math_data_latex.h5
│   │   ├── math_data_delete.h5
│   ├── GeoQA
│   │   ├── embeddings.pkl
│   │   ├── vocab_label.pkl
│   │   ├── math_data.h5
```

There are three kinds of files in each folder: `math_data.h5` , `vocab_label.pkl` and `embeddings.pkl`.

+ `math_data.h5` saves a dictionary.
```Python
{
    'train_X': [[1,12,233,10002,0,...]], # A set of questions, each item is a word_index sequence of the question, the sequence length is 120
    'train_Y': [[1,1,0,1,0,0,...]], # A set of multi-hot coding, each item is the knowledge concepts corresponding to the question
    'vaild_X': [[]], 
    'valid_Y': [[]], 
    'test_X': [[]],
    'test_Y': [[]]
}
```
+ `vocab_label.pkl` saves the tuple`(word2index, label2index)`.
+ `embeddings.pkl` saves two-dimensional arrays, the first dimension is word index, and the second dimension is corresponding embedding.

Questions in DA-20k include complex formulas. There are four different formula preprocessing methods corresponding to four `math_data.h5` files in `/DA-20k`:

`math_data_cut.h5`: Formulas are transformed into plain text and cut as general texts. (Formula-C; Our final method)

`math_data_parse.h5`: Formulas are parsed into tuples using TangenCFT method, and representation vectors are obtained using FastText. (Formula-P)

`math_data_latex.h5`: Formulas are transformed into LaTeX format and cut with special vocabulary like "$\backslash frac$" ,  "$\backslash sum$", etc. (Formula-L)

`math_data_delete.h5`: Formulas are deleted. (Formula-D)

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
python classification.py --config=config/config_cut.yml # specify the configuration file, DA-20k with Formula-C
python classification.py --config=config/config_parse.yml # specify the configuration file, DA-20k with Formula-P
python classification.py --config=config/config_latex.yml # specify the configuration file, DA-20k with Formula-L
python classification.py --config=config/config_delete.yml # specify the configuration file, DA-20k with Formula-D
python classification.py --config=config/config_GeoQA.yml # specify the configuration file, GeoQA
```

## Experiment Results and Analysis
### Experiment Results(RQ1)
The results in Section 5.1 are the average values of lots of experiments. You can run experiments on each dataset to get new results. 
```bash
cd MathByte
# for dataset DA-20k, we use Formula-C as default.
python classification.py --config=config/config_cut.yml
python classification.py --use_lcm=True --config=config/config_cut.yml
python classification.py --use_att=True --config=config/config_cut.yml 
python classification.py --use_lcm=True --use_att=True --config=config/config_cut.yml
# for dataset GeoQA
python classification.py --config=config/config_GeoQA.yml
python classification.py --use_lcm=True --config=config/config_GeoQA.yml
python classification.py --use_att=True --config=config/config_GeoQA.yml 
python classification.py --use_lcm=True --use_att=True --config=config/config_GeoQA.yml
```
You can also run `run.ipynb` to get new results, and the new model will be saved in `MathByte/logs` directory.
Our experimental results are saved to the `/results` folder.
The learning curve of four models are generated by `Section_5_1.ipynb`

### Experiment Results with GPT(RQ2)
We randomly selected 100 questions from each of our two datasets and used GPT-3.5 and GPT-4 zero-shot inference to tag the problems. The results are save to the `GPT_results(RQ2).xlsx` in the `/results` folder.

There are four sheets in the `GPT_results(RQ2).xlsx`:

- **DA-20k-4.0**: Results of tagging problems from DA-20k using GPT-4
- **DA-20k-3.5**: Results of tagging problems from DA-20k using GPT-3.5
- **GeoQA-4.0**: Results of tagging problems from GeoQA using GPT-4
- **GeoQA-3.5**: Results of tagging problems from GeoQA using GPT-3.5


| Column Name  | Description                               |
|--------------|-------------------------------------------|
| `question`   | Represents the text of the math problem   |
| `label_list` | Represents the true labels of the problem  |
| `knowledge`  | Represents the true knowlege concepts of the problem    |
| `GPT`        | Represents the predicted labels of the problem using GPT (None means labels which are not in our label list and generated by GPT) |
| `GPT_knowledge_points`  | Represents the predicted knowlege concepts of the problem using GPT |
| `right`  | Represents whether the prediction is right or not |

### Analysis of the influence of LA and LS(RQ3 & RQ4)
The codes for analysis in Section 5.3 & Section 5.4 are saved in `analysis on DA-20k.ipynb` and `analysis on GeoQA.ipynb`. The steps in them include loading the model, selecting test data, predicting results, attention visualization, and drawing the predicted or simulated results.

We provide models that have finished training on Figshare: https://figshare.com/s/8200c24498514bea91aa. 
You can download the models and load them directly. Make sure they are under the `/MathByte/logs` folder. 

An ideal example for the directory `MathByte/logs` is like :
```text
.
├── MathByte
│   ├── logs
│   │   ├── DA-20k
│   │   │   ├── cut
│   │   │   │   ├── LHABS
│   │   │   │   │   ├── {model_name}
│   │   │   │   │   │   ├── model
│   │   │   │   │   │   │   ├── checkpoint_labs.h5
│   │   │   │   │   │   ├── fit
│   │   │   │   │   │   ├── training.csv
│   │   │   │   ├── LHAB
│   │   │   │   │   ├── {model_name}
│   │   │   │   │   │   ├── ...
│   │   │   │   ├── LBS
│   │   │   │   │   ├── {model_name}
│   │   │   │   │   │   ├── ...
│   │   │   │   ├── Basic
│   │   │   │   │   ├── {model_name}
│   │   │   │   │   │   ├── ...
│   │   │   ├── parse
│   │   │   │   ├── LHABS
│   │   │   │   │   ├── {model_name}
│   │   │   │   │   │   ├── ...
│   │   │   ├── latex
│   │   │   │   ├── LHABS
│   │   │   │   │   ├── {model_name}
│   │   │   │   │   │   ├── ...
│   │   │   ├── delete
│   │   │   │   ├── LHABS
│   │   │   │   │   ├── {model_name}
│   │   │   │   │   │   ├── ...
│   │   ├── GeoQA
│   │   │   ├── LHABS
│   │   │   │   ├── {model_name}
│   │   │   │   │   ├── ...
│   │   │   ├── LHAB
│   │   │   │   ├── ...
│   │   │   ├── LBS
│   │   │   │   ├── ...
│   │   │   ├── Basic
│   │   │   │   ├── ...
```

### Comparison of different preprocessing methods of mathematical formulas(RQ5)
To compare three different methods, train LHABS model using Formula-C, Formula-P, Formula-L and Formula-D and compare the `Precision@1`, `Recall@3`, `F1@2` and `Accuracy@3` (the highest values for each metric).

```bash
cd MathByte
python classification.py --use_lcm=True --use_att=True --config=config/config_parse.yml
python classification.py --use_lcm=True --use_att=True --config=config/config_latex.yml
python classification.py --use_lcm=True --use_att=True --config=config/config_delete.yml
```