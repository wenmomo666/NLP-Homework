Yiwen Zhang
Yz3310
Hw4 

Using packages: tensorflow, keras,numpy.
.
|____trees
| |____dev.conll
| |____test.conll
| |____train.conll
|____models
| |____model1 (trained model of question1)
| |____model3 (trained model of question2)
| |____model2 (trained model of question3)
|____readme.txt (my documentation/observation)
|____data (data folder)
| |____dev.data
| |____vocabs.pos
| |____vocabs.word
| |____vocabs.actions
| |____train_with_indices.data
| |____vocabs.labels
| |____train.data
|____outputs (6 outputs file, which will be separately upload on canvas)
| |____test_part3.conll
| |____test_part1.conll
| |____dev_part3.conll
| |____dev_part1.conll
| |____test_part2.conll
| |____dev_part2.conll
|____src (I created train_1.py trian_2.py and train_3.py to train models; depModel1.py, depModel2.py and depModel3.py are the decoder for question 1,2,3; utils_1.py contains some helper functions. All others are given files.)
| |____train_2.py
| |____decoder.py
| |____configuration.py
| |____depModel2.py
| |____depModel3.py
| |____gen.py
| |____train_3.py
| |______init__.py
| |______pycache__
| | |____decoder.cpython-36.pyc
| | |____utils_1.cpython-36.pyc
| | |____aux.cpython-36.pyc
| | |____configuration.cpython-36.pyc
| | |____utils.cpython-36.pyc
| |____utils.py
| |____unit_tests.py
| |____eval.py
| |____utils_1.py
| |____train_1.py
| |____gen_vocab.py
| |____depModel1.py


Instructions: 
Train the models: 
Q1: python src/train_1.py
Q2: python src/train_2.py
Q3: python src/train_3.py
Validate the models:
Q1:python src/depModel1.py trees/dev.conll outputs/dev_part1.conll
   python src/eval.py trees/dev.conll outputs/dev_part1.conll

Q2:python src/depModel2.py trees/dev.conll outputs/dev_part2.conll
   python src/eval.py trees/dev.conll outputs/dev_part2.conll

Q3:python src/depModel3.py trees/dev.conll outputs/dev_part3.conll
   python src/eval.py trees/dev.conll outputs/dev_part3.conll

Test the modes:
Q1: python src/depModel1.py trees/test.conll outputs/test_part1.conll
Q2: python src/depModel2.py trees/test.conll outputs/test_part2.conll
Q3: python src/depModel3.py trees/test.conll outputs/test_part3.conll


Results:
Q1:
yiwenzhang$ python src/eval.py trees/dev.conll outputs/dev_part1.conll
Unlabeled attachment score 82.08
Labeled attachment score 78.79Q2:yiwenzhang$ python src/eval.py trees/dev.conll outputs/dev_part2.conllUnlabeled attachment score 83.09
Labeled attachment score 79.84Q3:yiwenzhang$ python src/eval.py trees/dev.conll outputs/dev_part3.conllUnlabeled attachment score 84.34
Labeled attachment score 81.26

Observations:
Q1: The result of Q1 is the baseline.
Q2:The accuracy of Q2 is slightly higher than question1, because the 2nd model uses 400 units in the hidden layers, which could capture more complex features. However, we could notice that it doesn't increase much, which caused by overfitting.

Q3: 
1) I set dropout=0.3 on our baseline model(all other parameter unchanged).(Unlabeled attachment score 84.1;Labeled attachment score 80.96). Got higher score than Q1 and Q2, which proved dropout method helps.
2) I set dropout=0.3 and epochs changed to 15 on our baseline model(Unlabeled attachment score 83.66;Labeled attachment score 80.51) This model got lower score than 1), may because overfit, which lead by larger epochs.
3) I set dropout=0.3 and change both hidden layer unites to 500 and number of epochs=15.(Unlabeled attachment score 84.34;Labeled attachment score 81.26). Here got the highest score, not only because dropout method helps and also more hidden unites could capture more details, which improved the performance of the model.
	
Finally we got the highest score on 3), which got the best model I saved.

