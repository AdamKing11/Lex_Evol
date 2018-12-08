import string, random, sys

try:
	import numpy as np
except:
	print('please install `numpy`')
	sys.exit()

try:
	from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
	from matplotlib.figure import Figure
	import matplotlib.animation as animation
except:
	print('please install `matplotlib`')
	sys.exit()

try:
	import tkinter as tk
	from tkinter import HORIZONTAL, END, IntVar
except:
	print('please install `tkinter`')
	sys.exit()
try:
	from scipy import stats
except:
	print('please install `scipy`')
	sys.exit()

from Lexicon import *

_colors = ['blue', 'orange', 'green', 'purple', 'black', 'yellow']

class EvolGUI():
	def __init__(self, lexicon):
		self.lexicon = lexicon

		self.root = tk.Tk()
		self.root.title("Bear down")

		self.fig = Figure()
		self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
		self.canvas.get_tk_widget().grid(row=0, column=0)

		self.fig2 = Figure()
		self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.root)
		self.canvas2.get_tk_widget().grid(row=0, column=2)

		self.fig3 = Figure()
		self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.root)
		self.canvas3.get_tk_widget().grid(row=1, column=0)

		self.fig4 = Figure()
		self.canvas4 = FigureCanvasTkAgg(self.fig4, master=self.root)
		self.canvas4.get_tk_widget().grid(row=1, column=2)

		
		button_frame = tk.Frame(self.root)
		button_frame.grid(row=0, column=1)

		self.evolution_steps = IntVar(0)
		tk.Label(button_frame, text = 'evolution steps:').grid(row=0, column=0)
		tk.Label(button_frame, textvariable = self.evolution_steps).grid(row=0, column=1)

		tk.Label(button_frame, text = 'lexicon size').grid(row=1, column=0)
		tk.Label(button_frame, text = 'n. symbols').grid(row=1, column=1)
		self.last_lexicon_size = 1000
		self.last_n_symbols = 5
		self.lexicon_size_text = tk.Entry(button_frame, width = 6)
		self.lexicon_size_text.grid(row=2, column=0)		
		self.lexicon_size_text.insert(0, str(self.last_lexicon_size))
		self.n_symbols_text = tk.Entry(button_frame, width = 4)
		self.n_symbols_text.grid(row=2, column=1)		
		self.n_symbols_text.insert(0, str(self.last_n_symbols))

		# prepare the buttons
		tk.Button(button_frame,text="One Step",command=self.step).grid(row=3, column=0)
		tk.Button(button_frame,text="20 Steps",command=lambda : self.step(20)).grid(row=3, column=1)
		tk.Button(button_frame,text="Reset Lexicon",command=self.reset_lex).grid(row=4, column=0)
		tk.Button(button_frame,text="Quit",command=sys.exit).grid(row=4, column=1)

		slider_frame = tk.Frame(self.root)
		slider_frame.grid(row=1, column=1)
		self.merger_p_slider = tk.Scale(slider_frame, from_=0, to=100, orient = HORIZONTAL, label = 'merger prob.')
		self.merger_p_slider.grid(row=0, column=0)
		self.merger_p_slider.set(50)
		
		tk.Label(slider_frame, text = 'phone. dist. E').grid(row=1, column=0)
		self.last_symbol_E = 1.
		self.symbol_E_text = tk.Entry(slider_frame, width = 4)
		self.symbol_E_text.grid(row=2, column=0)		
		self.symbol_E_text.insert(0, str(self.last_symbol_E))

		# prepare the line graph
		self.plot_1 = self.fig.subplots()

		max_si = 0
		self.avg_si_lines = []
		for i in range(self.lexicon.frequency_groups):
			x = np.arange(self.lexicon.hard_max_length) + 1
			avg_si = lexicon.avg_segmental_info(which_group = i + 1)
			new_line, = self.plot_1.plot(x, avg_si, color = _colors[i], label='group {0}'.format(i + 1))
			self.avg_si_lines.append(new_line)
			max_si = max(max_si, max(avg_si))

		self.plot_1.set_xlim(1, self.lexicon.hard_max_length)
		for y_lim in range(5, 50, 5):
			if y_lim > max_si:	break
		self.plot_1.set_ylim(-.5, y_lim)

		self.plot_1.legend(handles = self.avg_si_lines)
		self.plot_1.set_title('avg. seg info')

		# prep the word length histogram
		self.plot_2 = self.fig2.subplots()

		hist_data = [self.lexicon.word_lengths(i + 1) for i in range(self.lexicon.frequency_groups)]
		self.wl_hist = self.plot_2.hist(hist_data, range = (1, self.lexicon.hard_max_length), 
			stacked=False, color = _colors[:self.lexicon.frequency_groups])
		self.plot_2.set_title('word lengths')
		
		# zipf!
		sorted_unig = sorted([w.unigram for w in self.lexicon.words])
		plot_3 = self.fig3.subplots()
		plot_3.set_xlim(0, self.lexicon.hard_max_length)
		plot_3.set_ylim(sorted_unig[0] - 1, sorted_unig[-1] + 1)
		plot_3.set_title('word length and unigram word information')

		lengths, unigrams = zip(*self.lexicon.lengths_and_unigrams())
		slope, intercept, r_value, p_value, std_err = stats.linregress(lengths, unigrams)
		unig_pred = intercept + (slope * np.arange(self.lexicon.hard_max_length))

		self.zipf_scatter,  = plot_3.plot(lengths, unigrams, 'o')
		self.zipf_line, = plot_3.plot(np.arange(self.lexicon.hard_max_length), unig_pred)

		# phoneme distribution
		self.plot_4 = self.fig4.subplots(2)
		ks, ps =  p_dist_to_lists(self.lexicon.seg_ps, sort_by_keys = True)
		self.phoneme_dist_bars = self.plot_4[0].bar(np.arange(len(ks)), ps, color = _colors[-3])
		self.plot_4[0].set_ylim(0,.75 if max(ps) < .75 else 1)
		self.plot_4[0].set_xticks(np.arange(len(ks)))
		self.plot_4[0].set_xticklabels(ks)
		self.plot_4[0].set_title('seg. distribution in lexicon')

		# segmental entropy
		self.edge_ent_bars = []
		for i, edge_ent in enumerate(self.lexicon.edge_entropies()):
			new_bar = self.plot_4[1].bar((i * .5) + (1.5 * np.arange(self.lexicon.frequency_groups)), 
				edge_ent, color = _colors[-2+i], width = .5)
			self.edge_ent_bars.append(new_bar)
		
		self.plot_4[1].set_xticks(.25 + np.arange(self.lexicon.frequency_groups) * 1.5)
		self.plot_4[1].set_xticklabels(['group {0}'.format(i + 1) for i in range(self.lexicon.frequency_groups)])
		self.plot_4[1].set_title('seg. entropy - first/last segment')		
		self.plot_4[1].legend(labels = ['first', 'last'])

		tk.mainloop()

	def update(self):
		# update the plot with new data
		max_si = 0
		for i, line in enumerate(self.avg_si_lines):
			avg_si = self.lexicon.avg_segmental_info(which_group = i + 1)
			line.set_ydata(avg_si)
			max_si = max(max_si, max(avg_si))

		for y_lim in range(5, 50, 5):
			if y_lim > max_si:	break
		self.plot_1.set_ylim(-.5, y_lim)
		
		self.canvas.draw()
		
		# word length histogram
		self.plot_2.cla()
		hist_data = [self.lexicon.word_lengths(i + 1) for i in range(self.lexicon.frequency_groups)]	
		self.wl_hist = self.plot_2.hist(hist_data, range = (1, self.lexicon.hard_max_length), 
			stacked=False, color = _colors[:self.lexicon.frequency_groups])
		self.plot_2.set_title('word lengths')
		self.canvas2.draw()		

		# zipf scatter
		lengths, unigrams = zip(*self.lexicon.lengths_and_unigrams())
		slope, intercept, r_value, p_value, std_err = stats.linregress(lengths, unigrams)
		unig_pred = intercept + (slope * np.arange(self.lexicon.hard_max_length))

		self.zipf_scatter.set_xdata(lengths)
		self.zipf_line.set_ydata(unig_pred)

		self.canvas3.draw()

		ks, ps =  p_dist_to_lists(self.lexicon.seg_ps, sort_by_keys = True)
		self.plot_4[0].cla()
		self.phoneme_dist_bars = self.plot_4[0].bar(np.arange(len(ks)), ps, color = _colors[-3])
		self.plot_4[0].set_ylim(0,.75 if max(ps) < .75 else 1)
		self.plot_4[0].set_xticks(np.arange(len(ks)))
		self.plot_4[0].set_xticklabels(ks)
		self.plot_4[0].set_title('seg. distribution in lexicon')

		# first/last seg info
		self.plot_4[1].cla()
		for i, edge_ent in enumerate(self.lexicon.edge_entropies()):
			new_bar = self.plot_4[1].bar((i * .5) + (1.5 * np.arange(self.lexicon.frequency_groups)), 
				edge_ent, color = _colors[-2+i], width = .5)
			self.edge_ent_bars.append(new_bar)
		self.plot_4[1].set_xticks(.25 + np.arange(self.lexicon.frequency_groups) * 1.5)
		self.plot_4[1].set_xticklabels(['group {0}'.format(i + 1) for i in range(self.lexicon.frequency_groups)])
		self.plot_4[1].set_title('avg. information - first/last segment')
		self.plot_4[1].legend(labels = ['first', 'last'])
		self.canvas4.draw()
	
	def step(self, n_steps = 1):
		for i in range(n_steps):
			self.lexicon.change_segs(symbol_E = self.symbol_E(), merger_p = self.merger_p())
			print('step {0} - {1}'.format(i + 1, self.lexicon.entropy))
			if i % 4 == 0:
				self.update()
			self.evolution_steps.set(self.evolution_steps.get() + 1)				
		self.update()
		
	def reset_lex(self):
		self.evolution_steps.set(0)
		self.lexicon = Lexicon(self.lexicon_size(), phones = self.n_symbols(), 
			frequency_groups = self.lexicon.frequency_groups,
			hard_max_length = self.lexicon.hard_max_length, hard_start_length = self.lexicon.hard_start_length) 
		self.update()

	def merger_p(self):
		return self.merger_p_slider.get() / 100

	def symbol_E(self):
		try:
			symbol_E = float(self.symbol_E_text.get())
		except:
			symbol_E = self.last_symbol_E
		self.last_symbol_E = symbol_E
		return symbol_E

	def lexicon_size(self):
		try:
			lexicon_size = int(self.lexicon_size_text.get())
		except:
			symbol_E = self.last_lexicon_size
		self.last_lexicon_size = lexicon_size
		return lexicon_size

	def n_symbols(self):
		try:
			n_symbols = int(self.n_symbols_text.get())
		except:
			n_symbols = self.last_n_symbols
		self.last_n_symbols = n_symbols
		return n_symbols