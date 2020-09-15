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


#### Normal seq2seq model: 
- First run pre-processing
- Change working directory to code/main/
- Run: </br>
`python mt_main.py train 10 seq2seq` </br>
For inference: </br>
- Change working directory to code/main/
- Run: </br>
`python mt_main.py inference tmp/seq2seq5.ckpt greedy` </br>



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
