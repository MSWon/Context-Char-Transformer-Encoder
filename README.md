# Context-Char-Transformer-Encoder
- Representing words with balancing between context and character information
- We train how much contextual information and character information to use
## 1. Model 
- The model consists of a Transformer that learns contextual information and a Feed-Forward-Network that learns character information
- The figure below is a structure of the model
- The model learns how much to focus on between contextual information and character information by using gate

![alt_text](https://github.com/MSWon/Context-Char-Transformer-Encoder/blob/master/images/model.png "Model")

![alt_text](https://github.com/MSWon/Context-Char-Transformer-Encoder/blob/master/images/equation.png "Equation")

## 2. Experiments
- We have tested on 4 tasks and the results show that our proposed model converges faster compared to word, char level transformers
### 2-1. Named-Entity-Recognition 
![alt_text](https://github.com/MSWon/Context-Char-Transformer-Encoder/blob/master/images/ner_task.png "NER")
### 2-2. Part-Of-Speech Tagging
![alt_text](https://github.com/MSWon/Context-Char-Transformer-Encoder/blob/master/images/pos_task.png "POS Tagging")
### 2-3. AG's news text classification
![alt_text](https://github.com/MSWon/Context-Char-Transformer-Encoder/blob/master/images/classification_task.png "Classification")
### 2-4. Vietnamese to English
![alt_text](https://github.com/MSWon/Context-Char-Transformer-Encoder/blob/master/images/translation_task.png "Translation")
