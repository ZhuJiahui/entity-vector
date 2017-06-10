entity-vector
=============

## Introduction

This tool provides a Python implementation of learning embeddings of words and Wikipedia entities.

The embedding can be directly built using a Wikipedia dump retrieved from [Wikimedia Downloads](http://dumps.wikimedia.org/).

For further details, please refer to our [paper](https://arxiv.org/abs/1601.01343).

## Installing package

```
% pip install Cython numpy
% pip install -r requirements.txt
% python setup.py develop
```

## Basic usage

The pretrained embedding can be downloaded from the following links.
Please note that these files must be placed in the same directory.

* [enwiki_entity_vector_500_20151026.pickle](http://entity-vector.s3.amazonaws.com/pub/enwiki_entity_vector_500_20151026.pickle)
* [enwiki_entity_vector_500_20151026_syn0.npy](http://entity-vector.s3.amazonaws.com/pub/enwiki_entity_vector_500_20151026_syn0.npy)


```python
>>> from entity_vector import EntityVector
>>> entvec = EntityVector.load('enwiki_entity_vector_500_20151026.pickle')
>>> word = entvec.get_word(u'c-3po')
>>> entvec[word]
memmap([ 0.05961042,  0.24534572,  0.42090839, -0.01455959,  0.11772038,
        0.55437287, -0.62508648, -0.24478671,  0.07838536,  0.27331885,
        0.35184374,  0.34113087,  0.11718472, -0.14086614, -0.00730115,
...
>>> entvec.most_similar(word)
[(<Word c-3po>, 1.0000000000000002),
 (<Entity C-3PO>, 0.8855517572211461),
 (<Word r2-d2>, 0.85768096183067088),
 (<Entity R2-D2>, 0.81842535257607718),
 (<Word chewbacca>, 0.7771232783769505),
 (<Entity Chewbacca>, 0.77412692204846856),
...
>>> entity = entvec.get_entity(u'C-3PO')
>>> entvec[entity]
memmap([ -3.51071961e-03,   4.82281654e-01,   6.72443198e-01,
         2.41103170e-01,   1.43198542e-01,   6.44051048e-01,
        -5.48925964e-01,  -4.64934616e-01,  -2.48444133e-01,
...
>>> entvec.most_similar(entity)
[(<Entity C-3PO>, 1.0),
 (<Entity R2-D2>, 0.90188752966007535),
 (<Word c-3po>, 0.88555175722114643),
 (<Entity Chewbacca>, 0.8304708994223623),
 (<Word r2-d2>, 0.82777910810169675),
 (<Entity Han Solo>, 0.80912814689071744),
 ...
>>> entvec.get_similarity(word, entity)
0.90466782126690559
```

## Reference

If you use the code or the pretrained embedding in your research, please cite the following paper:

```
@InProceedings{yamada-EtAl:2016:CoNLL,
author    = {Yamada, Ikuya  and  Shindo, Hiroyuki  and  Takeda, Hideaki  and  Takefuji, Yoshiyasu},
title     = {Joint Learning of the Embedding of Words and Entities for Named Entity Disambiguation},
booktitle = {Proceedings of The 20th SIGNLL Conference on Computational Natural Language Learning},
month     = {August},
year      = {2016},
address   = {Berlin, Germany},
publisher = {Association for Computational Linguistics},
pages     = {250--259}
}
```

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
