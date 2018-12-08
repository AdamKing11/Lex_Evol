import string, random, sys

try:
	from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
	from matplotlib.figure import Figure
	import matplotlib.animation as animation
except:
	print('please install `matplotlib`!')
	sys.exit()

try:
	import tkinter as tk
	from tkinter import HORIZONTAL, END
	from tkinter.ttk import Progressbar
except:
	print('please install `tkinter`')
	sys.exit()

try:
	import numpy as np
except:
	print('please install `numpy`!')
	sys.exit()

def log2(p):
	return np.log(p) / np.log(2)

def exp_h(p):
	return p * -log2(p)

def H(ps):
	H = 0
	for p in ps:
		H += exp_h(p)
	return H

def dict_to_p_dist(d, E = 1):
	sum_vals = sum([v ** E for v in d.values()])
	return dict((k, (v ** E) / sum_vals) for k, v in d.items())

def p_dist_to_lists(p_dist, sort_by_keys = False):
	if sort_by_keys:
		ks, ps = zip(*[(c, p / sum(p_dist.values())) for c, p in sorted(p_dist.items(), key = lambda x : x[0])])
	else:
		ks, ps = zip(*[(c, p / sum(p_dist.values())) for c, p in p_dist.items()])
	return ks, ps

def sample_from_p_dict(p_dist, n = 1):
	ks, ps = p_dist_to_lists(p_dist)
	return ''.join(np.random.choice(ks, n, replace = True, p = ps))

if __name__ == '__main__':
	n_words = 1000
	l = Lexicon(n_words)
	EvolGUI(l)
