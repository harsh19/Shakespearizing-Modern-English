# Shakespearizing-Modern-English
Code for "Jhamtani H., Gangal V., Hovy E. and Nyberg E. Shakespearizing Modern Language Using Copy-Enriched Sequence to Sequence Models"  Workshop on Stylistic Variation, EMNLP 2017

Link to paper: https://arxiv.org/abs/1707.01161

### Requirements
- Python 2.7
- Tensorflow 1.1.0

### Instructions to run:

#### Preprocessing: 
- Change working directory to code/main/
- Create a new directory named 'tmp'
- Run: </br>
`python mt_main.py preprocessing` </br>

#### Pointer model: 
- First run pre-processing
- Change working directory to code/main/
- `python mt_main.py train 10 pointer_model` </br>
For inference: </br>
- Change working directory to code/main/
- `python mt_main.py inference tmp/pointer_model7.ckpt greedy` </br>

#### Normal seq2seq model: 
- First run pre-processing
- Change working directory to code/main/
- Run: </br>
`python mt_main.py train 10 seq2seq` </br>
For inference: </br>
- Change working directory to code/main/
- Run: </br>
`python mt_main.py inference tmp/seq2seq5.ckpt greedy` </br>

#### Post-Processing:
There are two post-processing actions which one may be interested in performing:
1. Visualizing attention matrices
2. Replacing UNKS in hypothesis with their highest-aligned (attention) input tokens.
For both of these actions, refer to the running instructions in code/main/post_process.py (comments commencing the file). The file can be run in two modes, to perform 1 (write) and 2 (postProcess) respectively*.
*Not elaborated on here to preserve conciseness and clarity. </br>
Note that the path to test file is hard-coded in the post_process.py file, so to try with a new file,one will have to make corresponding changes.


#### Baseline (Dictionary):
- Change working directory to code/baselines/
- Run: </br>
`python dictionary_baseline.py ../../data/shakespeare.dict ../../data/test.modern.nltktok ../../data/test.dictBaseline`
- The test.dictBaseline file contains the output (Shakespearean) of the dictionary baseline.
- To evaluate BLEU: 
  - Change working directory to code/main/
  - Run: </br>
  `perl multi-bleu.perl -lc ../../data/test.original.nltktok < ../../data/test.dictBaseline`

#### Baseline (statistical MT)
- Please follow instructions in "Wei Xu, Alan Ritter, William B Dolan, Ralph Grish- man, and Colin Cherry. 2012. Paraphrasing for style. In 24th International Conference on Computational Linguistics, COLING 2012."




### Citation
If you use this code or the processed data, please consider citing our work:
```
@article{jhamtani2017shakespearizing,
  title={Shakespearizing Modern Language Using Copy-Enriched Sequence-to-Sequence Models},
  author={Jhamtani, Harsh and Gangal, Varun and Hovy, Eduard and Nyberg, Eric},
  journal={EMNLP 2017},
  volume={6},
  pages={10},
  year={2017}
}
```

Additionally, if you use the data, please consder citing "Wei Xu, Alan Ritter, William B Dolan, Ralph Grish- man, and Colin Cherry. 2012. Paraphrasing for style. In 24th International Conference on Computational Linguistics, COLING 2012."
