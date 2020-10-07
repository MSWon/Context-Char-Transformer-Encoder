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
- `config.yaml`
  ```python
  
  ```
- Pretrain using `config.yaml` file

```
$ cd CCTE
$ python main.py -c config.yaml
```

## 3. Downstream task (NER)

- Download pretrained model
- `ner_config.yaml`
  ```python
  # Discriminator
  D_num_layers: 6
  D_num_heads: 8
  D_hidden_dim: 512
  D_linear_key_dim: 512
  D_linear_value_dim: 512
  D_ffn_dim: 2048
  D_dropout: 0.1
  D_activation: gelu
  char_emb_dim: 16
  kernel_width: [1, 2, 3, 4, 5, 6, 7]
  kernel_depth: [32, 32, 64, 128, 256, 512, 1024]
  highway_layers: 2
  # data config
  char_vocab_path: data/char.en.vocab
  # training config
  max_word_len: 150
  max_char_len: 50
  batch_size: 32
  data_path: CoNLL # (CoNLL, wnut17, bio_ner) 중 택1
  pretrained_model_path: saved_model/char_electra.model
  training_epochs: 3
  train_type: org
  test_type: org
  # vocab config
  char_vocab_size: 251
  ```
- Fine-tune with `ner_config.yaml` file

```
$ cd CCTE
$ python main.py -c config.yaml
```
