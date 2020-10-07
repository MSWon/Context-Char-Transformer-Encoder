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
  # Generator
  G_num_layers: 6
  G_num_heads: 4
  G_hidden_dim: 256
  G_linear_key_dim: 256
  G_linear_value_dim: 256
  G_ffn_dim: 1024
  G_dropout: 0.1
  G_activation: gelu
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
  # Model weight
  G_weight: 1.0
  D_weight: 50.0
  temperature: 1.0
  # data config
  corpus_path: data/train.tok.en
  word_vocab_path: data/bpe.en.vocab
  char_vocab_path: data/char.en.vocab
  model_path: char_electra.model
  # training config
  max_word_len: 150
  max_char_len: 50
  batch_size: 1024
  training_steps: 300000
  warmup_step: 10000
  n_gpus: 8
  # vocab config
  mask_idx: 2
  vocab_size: 30001
  char_vocab_size: 251  
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
