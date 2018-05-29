## Extensions to Hybrid Code Networks for FAIR Dialog Dataset
Jiyeon Ham, Soohyun Lim, Kyeng-Hun Lee, Kee-Eung Kim

### Getting started
**Version info**
* Python 2.7
* Keras 2.1.4

**Train entity tracking module**

    $ python main.py -t -et

* -t: train
* -et: entity tracking module

**Train action selector module**

    $ python main.py -t -as -ts 1

* -t: train
* -as: action selector module
* -ts: task number to train

**Train entity output module**

    $ python main.py -t -eo

* -t: train
* -eo: entity output module

**Predict**

    $ python main.py -us -oov -ts 1

* -ts: task number to predict
* -us: test data with unseen slot
* -oov: test data with out-of-vocabulary