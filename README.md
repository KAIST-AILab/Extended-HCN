## Extensions to Hybrid Code Networks for FAIR Dialog Dataset
Jiyeon Ham, Soohyun Lim, Kyeng-Hun Lee, Kee-Eung Kim

### Getting started
#### Version info

* Python 2.7
* Keras 2.1.4

#### Prerequisite

* Download dataset from https://github.com/perezjln/dstc6-goal-oriented-end-to-end
* Make `data` directory and unzip the dataset
* Make `weight` directory

Then tree veiw should be shown as:

```
├─data
│  ├─dataset-E2E-goal-oriented
│  └─dataset-E2E-goal-oriented-test-v1.0
│      ├─tst1
│      ├─tst2
│      ├─tst3
│      └─tst4
├─scripts
└─weight
```

#### Training

run 	`scripts/main.py` with the following arguments:

* `-t`: train
* `-et`: entity tracking module
* `-as`: action selector module
* `-eo`: entity output module
* `-ts`: task number to train (only used for action selector module)

Train entity tracking module

```bash
$ python scripts/main.py -t -et
```

Train action selector module for task 1

```bash
$ python scripts/main.py -t -as -ts 1
```

Train entity output module

```bash
$ python scripts/main.py -t -eo
```

#### Predict

run `scripts/main.py` with the following arguments:

* `-us`: test data with unseen slot
* `-oov`: test data with out-of-vocabulary knowledge base
* `-ts`: task number to predict

Predict for task 1 with unseen slot and out-of-vocabulary

```bash
$ python main.py -us -oov -ts 1
```

