import pickle
def save_pkl(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
