import os, pickle

def check_path_exist(path):
    return False

def save_object(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pickle.dump(obj, open(path, 'wb'))

def load_object(path):
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    return None

def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)