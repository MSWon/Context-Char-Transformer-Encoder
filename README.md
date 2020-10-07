# CCTE
- Context Char Transformer Encoder

## 1. Install

```
$ git clone https://github.com/MSWon/Context-Char-Transformer-Encoder.git
$ cd Context-Char-Transformer-Encoder
$ python setup.py install
```

## 2. Pretrain

- Download 1-billion-word benchmark data
- Pretrain using [config.yaml](https://github.com/MSWon/Context-Char-Transformer-Encoder/blob/master/CCTE/config.yaml) file

```
$ cd CCTE
$ python main.py -c config.yaml
```

## 3. Downstream task (NER)

- Download pretrained model
  ```
  $ cd CCTE
  $ sh download_data.sh
  ```
- Fine-tune with [ner_config.yaml](https://github.com/MSWon/Context-Char-Transformer-Encoder/blob/master/CCTE/ner_config.yaml) file

```
$ cd CCTE
$ python main.py -c config.yaml
```
