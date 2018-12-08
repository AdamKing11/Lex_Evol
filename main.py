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


from Lexicon import *
from GUI import *

if __name__ == '__main__':
	print('\n\n\n')
	n_words = 1000
	n_phones = 5
	#phones = {'a' : 10, 'b' : 5, 'c' : 5, 'd' : 1}
	l = Lexicon(n_words, phones = n_phones, frequency_groups = 2, hard_start_length = 7)
	"""
	word_list = sorted(l.words, key = lambda w : w.frequency)

	for i in range(100):
		z = l.sample_zipf()
		zp = l.frequency_to_p(z)
		zl = l.word_p_to_length([zp])
		print(z, zp, zl)
	"""
	EvolGUI(l)
