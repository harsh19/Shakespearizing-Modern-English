# Shakespearizing-Modern-English
Code for "Jhamtani H., Gangal V., Hovy E. and Nyberg E. Shakespearizing Modern Language Using Copy-Enriched Sequence to Sequence Models"  Workshop on Stylistic Variation, EMNLP 2017

Code has been tested with Tensorflow version 1.1.0
- If you use this code or the processed data, please consider citing Jhamtani H., Gangal V., Hovy E. and Nyberg E. Shakespearizing Modern Language Using Copy-Enriched Sequence to Sequence Models"  Workshop on Stylistic Variation, EMNLP 2017
- If you use the data, please consder citing "Wei Xu, Alan Ritter, William B Dolan, Ralph Grish- man, and Colin Cherry. 2012. Paraphrasing for style. In 24th International Conference on Computational Linguistics, COLING 2012."

Instructions to run: </br>
Preprocessing: 
- Change working directory to code/main/
- Create a new directory named 'tmp'
- python mt_main.py pre-processing </br>

Pointer model: 
- First run pre-processing
- Change working directory to code/main/
- python mt_main.py train 10 pointer_model </br>
For inference: </br>
- Change working directory to code/main/
- python mt_main.py inference tmp/pointer_model7.ckpt greedy </br>

Normal seq2seq model: 
- First run pre-processing
- Change working directory to code/main/
- python mt_main.py train 10 seq2seq </br>
For inference: </br>
- Change working directory to code/main/
- python mt_main.py inference tmp/seq2seq5.ckpt greedy </br>

Baseline (Dictionary):
- Change working directory to code/baselines/
- TODO </br>

Baseline (statistical MT)
- Please follow instructions in "Wei Xu, Alan Ritter, William B Dolan, Ralph Grish- man, and Colin Cherry. 2012. Paraphrasing for style. In 24th International Conference on Computational Linguistics, COLING 2012."
