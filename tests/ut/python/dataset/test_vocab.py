import mindspore.dataset as ds
import mindspore.dataset.text as text

# this file contains "home is behind the world head" each word is 1 line
DATA_FILE = "../data/dataset/testVocab/words.txt"
VOCAB_FILE = "../data/dataset/testVocab/vocab_list.txt"


def test_from_list():
    vocab = text.Vocab.from_list("home IS behind the world ahead !".split(" "))
    lookup = text.Lookup(vocab)
    data = ds.TextFileDataset(DATA_FILE, shuffle=False)
    data = data.map(input_columns=["text"], operations=lookup)
    ind = 0
    res = [2, 1, 4, 5, 6, 7]
    for d in data.create_dict_iterator():
        assert d["text"] == res[ind], ind
        ind += 1


def test_from_file():
    vocab = text.Vocab.from_file(VOCAB_FILE, ",")
    lookup = text.Lookup(vocab)
    data = ds.TextFileDataset(DATA_FILE, shuffle=False)
    data = data.map(input_columns=["text"], operations=lookup)
    ind = 0
    res = [10, 11, 12, 15, 13, 14]
    for d in data.create_dict_iterator():
        assert d["text"] == res[ind], ind
        ind += 1


def test_from_dict():
    vocab = text.Vocab.from_dict({"home": 3, "behind": 2, "the": 4, "world": 5, "<unk>": 6})
    lookup = text.Lookup(vocab, 6)  # default value is -1
    data = ds.TextFileDataset(DATA_FILE, shuffle=False)
    data = data.map(input_columns=["text"], operations=lookup)
    res = [3, 6, 2, 4, 5, 6]
    ind = 0
    for d in data.create_dict_iterator():
        assert d["text"] == res[ind], ind
        ind += 1

if __name__ == '__main__':
    test_from_list()
    test_from_file()
    test_from_dict()
