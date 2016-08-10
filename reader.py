import numpy
import os
import pickle

labels = ['O', 'B-PLACE', 'I-PLACE', 'B-PERSON', 'I-PERSON', 'B-ORGANIZATION',
    'I-ORGANIZATION', 'B-OTHERPROPNOUN', 'I-OTHERPROPNOUN']
id_to_labels = dict(enumerate(labels))
labels_to_id = dict((v,i) for i, v in enumerate(labels))

def load(data_file = "ner_data2.pkl"):
  if os.path.exists(data_file):
    return pickle.load(open(data_file, 'rb'))

  word_set = set()
  def load_file(filename, build_dict=False):
    X = []
    Y = []
    with open(filename, 'r') as fin:
      _x = []
      _y = []
      for r in fin.readlines():
        if len(r.strip()) == 0:
          if len(_x) >0:
            X.append(_x)
            Y.append(numpy.asarray(_y))
            _x = []
            _y = []
        else:
          cells = r.split()
          _x.append(cells[5])
          _y.append(labels_to_id[cells[0]])
          if build_dict:
            word_set.add(cells[5])
    return X, Y

  train_x, train_y = load_file('/home/hadoop/data/ner/train_new', build_dict=True)
  test_x, test_y = load_file('/home/hadoop/data/ner/test_new')
  word_to_id = dict((w, i) for i, w in enumerate(word_set))

  print('total words: ', len(word_to_id))
  print('train set size: ', len(train_x))
  print('test set size: ', len(test_x))
  

  def convert(X): 
    Xid = []
    for _x in X:
      Xid.append([word_to_id[v] if v in word_to_id else -1 for v in _x])
    return Xid

  trainX = numpy.asarray(convert(train_x))
  trainY = numpy.asarray(train_y)
  testX = numpy.asarray(convert(test_x))
  testY = numpy.asarray(test_y)
  with open(data_file, 'wb') as fout:
    pickle.dump((trainX, trainY, testX, testY, word_to_id, labels_to_id), fout)

  return trainX, trainY, testX, testY, word_to_id, labels_to_id
