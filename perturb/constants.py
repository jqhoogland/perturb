import torch as t

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")