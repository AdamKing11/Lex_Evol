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
except:
	print('please install `tkinter`')
	sys.exit()
try:
	from scipy import stats
except:
	print('please install `scipy`')
	sys.exit()

from Lexicon import *

_colors = ['blue', 'orange', 'green', 'purple', 'black', 'red']

class EvolGUI():
	def __init__(self, lexicon):
		self.lexicon = lexicon

		# each `root` object is a window
		# open one for each figure and all the buttons/sliders
		self.root = tk.Tk()
		self.root.title("Segmental Information by position")
		self.fig = Figure()
		self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
		self.canvas.get_tk_widget().grid(row=0, column=0)

		self.root2 = tk.Tk()
		self.root2.title("Word Length Distribution")
		self.fig2 = Figure()
		self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.root2)
		self.canvas2.get_tk_widget().grid(row=0, column=0)

		self.root3 = tk.Tk()
		self.root3.title("Word Length and -log Word Probability")
		self.fig3 = Figure()
		self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.root3)
		self.canvas3.get_tk_widget().grid(row=0, column=0)

		self.root4 = tk.Tk()
		self.root4.title("Lexical Phoneme distribution")
		self.fig4 = Figure()
		self.canvas4 = FigureCanvasTkAgg(self.fig4, master=self.root4)
		self.canvas4.get_tk_widget().grid(row=0, column=0)

		self.root5 = tk.Tk()
		self.root5.title("Positional Entropy")
		self.fig5 = Figure()
		self.canvas5 = FigureCanvasTkAgg(self.fig5, master=self.root5)
		self.canvas5.get_tk_widget().grid(row=0, column=0)

		self.interaction_root = tk.Tk()
		self.interaction_root.title("Bear down")
		button_frame = tk.Frame(self.interaction_root)
		button_frame.grid(row=0, column=0)

		self.evolution_steps = 0
		tk.Label(button_frame, text = 'evolution steps:').grid(row=0, column=0)
		self.evolution_steps_label = tk.Label(button_frame, text = self.evolution_steps)
		self.evolution_steps_label.grid(row=0, column=1)

		tk.Label(button_frame, text = 'lexicon size').grid(row=1, column=0)
		tk.Label(button_frame, text = 'n. symbols').grid(row=1, column=1)
		self.last_lexicon_size = len(self.lexicon)
		self.last_n_symbols = len(self.lexicon.phones)
		self.lexicon_size_text = tk.Entry(button_frame, width = 6)
		self.lexicon_size_text.grid(row=2, column=0)		
		self.lexicon_size_text.insert(0, str(self.last_lexicon_size))
		self.n_symbols_text = tk.Entry(button_frame, width = 4)
		self.n_symbols_text.grid(row=2, column=1)		
		self.n_symbols_text.insert(0, str(self.last_n_symbols))

		# figure out minimum length needed for whole lexicon
		total_possible_forms = 0
		for i in range(1, self.lexicon.hard_max_length):
			total_possible_forms += self.n_symbols() ** i
			if total_possible_forms > len(self.lexicon):
				break

		self.min_len_needed = tk.Label(button_frame, text = 'max length needed for lexicon: {0}'.format(i))
		self.min_len_needed.grid(row=3, column=0)
		self.possible_forms = tk.Label(button_frame, text = 'possible forms at length: {0}'.format(total_possible_forms))
		self.possible_forms.grid(row=3, column=1)
		
		tk.Label(button_frame, text = 'starting word length (-1 for Zipfian)').grid(row=4, column=0)
		self.last_hard_word_length = 7
		self.hard_word_length_text = tk.Entry(button_frame, width = 4)
		self.hard_word_length_text.grid(row=4, column=1)
		self.hard_word_length_text.insert(0, str(self.last_hard_word_length))
		
		# prepare the buttons
		tk.Button(button_frame,text="One Step",command=self.step).grid(row=5, column=0)
		tk.Button(button_frame,text="20 Steps",command=lambda : self.step(20)).grid(row=5, column=1)
		tk.Button(button_frame,text="Reset Lexicon",command=self.reset_lex).grid(row=6, column=0)
		tk.Button(button_frame,text="Quit",command=sys.exit).grid(row=6, column=1)

		slider_frame = tk.Frame(self.interaction_root)
		slider_frame.grid(row=0, column=1)
		self.merger_p_slider = tk.Scale(slider_frame, from_=0, to=100, orient = tk.HORIZONTAL, label = 'merger prob.')
		self.merger_p_slider.grid(row=0, column=1)
		self.merger_p_slider.set(50)
		
		# determine skew in distribution for merger
		tk.Label(slider_frame, text = 'phone. dist. E').grid(row=1, column=1)
		self.last_symbol_E = 1.
		self.symbol_E_text = tk.Entry(slider_frame, width = 4)
		self.symbol_E_text.grid(row=2, column=1)		
		self.symbol_E_text.insert(0, str(self.last_symbol_E))

		# skew in distribution for chosing word to undergo merger/deletion
		tk.Label(slider_frame, text = 'word E').grid(row=1, column=0)
		self.last_word_E = 1.
		self.word_E_text = tk.Entry(slider_frame, width = 4)
		self.word_E_text.grid(row=2, column=0)		
		self.word_E_text.insert(0, str(self.last_word_E))

		# skew in distribution for chosing segment in word which will undergo merger/deletion
		tk.Label(slider_frame, text = 'segment E').grid(row=1, column=2)
		self.last_segment_E = 1.
		self.segment_E_text = tk.Entry(slider_frame, width = 4)
		self.segment_E_text.grid(row=2, column=2)		
		self.segment_E_text.insert(0, str(self.last_segment_E))		

		# prepare the line graph
		self.plot_1 = self.fig.subplots()

		max_si = 0
		self.avg_si_lines = []
		if self.lexicon.frequency_groups == 2:
			line_labels = ['high frequency', 'low frequency']
		else:
			line_labels = ['group {0}'.format(i + 1) for i in range(self.lexicon.frequency_groups)]
		
		for i in range(self.lexicon.frequency_groups):
			x = np.arange(self.lexicon.hard_max_length) + 1
			avg_si = self.lexicon.avg_segmental_info(which_group = i + 1)
			new_line, = self.plot_1.plot(x, avg_si, color = _colors[i], label=line_labels[i])
			self.avg_si_lines.append(new_line)
			max_si = max(max_si, max(avg_si))

		self.plot_1.set_xlim(1, self.lexicon.hard_max_length)
		for y_lim in range(5, 50, 5):
			if y_lim > max_si:	break
		self.plot_1.set_ylim(-.5, y_lim)

		self.plot_1.legend(handles = self.avg_si_lines)
		#self.plot_1.set_title('avg. seg info')

		# prep the word length histogram
		self.plot_2 = self.fig2.subplots()

		hist_data = [self.lexicon.word_lengths(i + 1) for i in range(self.lexicon.frequency_groups)]
		self.wl_hist = self.plot_2.hist(hist_data, range = (1, self.lexicon.hard_max_length), 
			stacked=False, color = _colors[:self.lexicon.frequency_groups])
		#self.plot_2.set_title('word lengths')
		
		# zipf!
		sorted_unig = sorted([w.unigram for w in self.lexicon.words])
		self.plot_3 = self.fig3.subplots()
		self.plot_3.set_xlim(0, self.lexicon.hard_max_length)
		self.plot_3.set_ylim(sorted_unig[0] - 1, sorted_unig[-1] + 1)
		self.plot_3.set_title('word length and unigram word information')

		lengths, unigrams = zip(*self.lexicon.lengths_and_unigrams())
		slope, intercept, r_value, p_value, std_err = stats.linregress(lengths, unigrams)
		unig_pred = intercept + (slope * np.arange(self.lexicon.hard_max_length))

		self.zipf_scatter,  = self.plot_3.plot(lengths, unigrams, 'o')
		self.zipf_line, = self.plot_3.plot(np.arange(self.lexicon.hard_max_length), unig_pred)

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
		if self.lexicon.frequency_groups == 2:
			self.plot_4[1].set_xticklabels(['high frequency', 'low frequency'])
		else:
			self.plot_4[1].set_xticklabels(['group {0}'.format(i + 1) for i in range(self.lexicon.frequency_groups)])
		self.plot_4[1].set_title('seg. entropy - first/last segment')		
		self.plot_4[1].legend(labels = ['first', 'last'])

		self.plot_5 = self.fig5.subplots()

		self.pos_ent_lines = []
		max_pe = 0
		# could use this from above but this makes it easier to read...
		if self.lexicon.frequency_groups == 2:
			line_labels = ['high frequency', 'low frequency']
		else:
			line_labels = ['group {0}'.format(i + 1) for i in range(self.lexicon.frequency_groups)]

		for i in range(self.lexicon.frequency_groups):
			x = np.arange(self.lexicon.hard_max_length) + 1
			pos_ent = self.lexicon.positional_entropy(which_group = i + 1)
			new_line, = self.plot_5.plot(x, pos_ent, color = _colors[i], label=line_labels[i])
			self.pos_ent_lines.append(new_line)
			max_pe = max(max_pe, max(pos_ent))

		self.plot_5.set_xlim(1, self.lexicon.hard_max_length)
		for y_lim in range(5, 30, 3):
			if y_lim > max_pe:	break
		self.plot_5.set_ylim(-.5, y_lim)

		self.plot_5.legend(handles = self.pos_ent_lines)
		
		tk.mainloop()

	def update(self):
		# put all the updating in try/catch in case someone closes a window	
		try:
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
		except:
			pass
			#print('unable to update plot 1')
		
		# word length histogram
		try:
			self.plot_2.cla()
			hist_data = [self.lexicon.word_lengths(i + 1) for i in range(self.lexicon.frequency_groups)]	
			self.wl_hist = self.plot_2.hist(hist_data, range = (1, self.lexicon.hard_max_length), 
				stacked=False, color = _colors[:self.lexicon.frequency_groups])
			self.plot_2.set_title('word lengths')
			self.canvas2.draw()		
		except:
			pass
			#print('unable to update plot 2')

		try:
			# zipf scatter
			lengths, unigrams = zip(*self.lexicon.lengths_and_unigrams())
			slope, intercept, r_value, p_value, std_err = stats.linregress(lengths, unigrams)
			unig_pred = intercept + (slope * np.arange(self.lexicon.hard_max_length))

			sorted_unig = sorted([w.unigram for w in self.lexicon.words])
			self.plot_3.set_ylim(sorted_unig[0] - 1, sorted_unig[-1] + 1)
			self.zipf_scatter.set_xdata(lengths)
			self.zipf_line.set_ydata(unig_pred)
			self.canvas3.draw()
		except:
			pass
			#print('unable to update plot 3')

		try:
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
			if self.lexicon.frequency_groups == 2:
				self.plot_4[1].set_xticklabels(['high frequency', 'low frequency'])
			else:
				self.plot_4[1].set_xticklabels(['group {0}'.format(i + 1) for i in range(self.lexicon.frequency_groups)])
			self.plot_4[1].set_title('avg. information - first/last segment')
			self.plot_4[1].legend(labels = ['first', 'last'])
			self.canvas4.draw()
		except:
			pass
			#print('unable to update plot 4')

		try:
			max_pe = 0
			for i, line in enumerate(self.pos_ent_lines):
				pos_ent = self.lexicon.positional_entropy(which_group = i + 1)
				line.set_ydata(pos_ent)
				max_si = max(max_pe, max(pos_ent))

			for y_lim in range(5, 30, 3):
				if y_lim > max_pe:	break
			self.plot_5.set_ylim(-.5, y_lim)
			self.canvas5.draw()

		except:
			pass
			#print('unable to update plot 5')

	def step(self, n_steps = 1):
		for i in range(n_steps):
			self.evolution_steps +=  1
			self.evolution_steps_label['text'] = self.evolution_steps
			self.lexicon.change_segs(word_E = self.word_E(), symbol_E = self.symbol_E(), merger_p = self.merger_p())
			print('step: {0} - total steps: {1}'.format(i + 1, self.evolution_steps))
			if i % 4 == 0:
				self.update()
							
		self.update()
		
	def reset_lex(self):
		self.evolution_steps = 0
		self.lexicon = Lexicon(self.lexicon_size(), phones = self.n_symbols(), 
			frequency_groups = self.lexicon.frequency_groups,
			hard_max_length = self.lexicon.hard_max_length, hard_start_length = self.hard_word_length()) 

		# figure out minimum length needed for whole lexicon
		total_possible_forms = 0
		for i in range(1, self.lexicon.hard_max_length):
			total_possible_forms += self.n_symbols() ** i
			if total_possible_forms > len(self.lexicon):
				break
		self.min_len_needed['text'] = 'max length needed for lexicon: {0}'.format(i)
		self.possible_forms['text'] = 'possible forms at length: {0}'.format(total_possible_forms)
		
		# zipf
		self.plot_3.cla()
		sorted_unig = sorted([w.unigram for w in self.lexicon.words])
		self.plot_3.set_xlim(0, self.lexicon.hard_max_length)
		self.plot_3.set_ylim(sorted_unig[0] - 1, sorted_unig[-1] + 1)
		self.plot_3.set_title('word length and unigram word information')

		lengths, unigrams = zip(*self.lexicon.lengths_and_unigrams())
		slope, intercept, r_value, p_value, std_err = stats.linregress(lengths, unigrams)
		unig_pred = intercept + (slope * np.arange(self.lexicon.hard_max_length))

		self.zipf_scatter,  = self.plot_3.plot(lengths, unigrams, 'o')
		self.zipf_line, = self.plot_3.plot(np.arange(self.lexicon.hard_max_length), unig_pred)

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

	def word_E(self):
		try:
			word_E = int(self.word_E_text.get())
		except:
			word_E = self.last_word_E
		self.last_word_E = word_E
		return word_E

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

	def hard_word_length(self):
		try:
			hard_word_length = int(self.hard_word_length_text.get())
		except:
			hard_word_length = self.last_hard_word_length
		self.last_hard_word_length = hard_word_length
		if hard_word_length < 0: 
			hard_word_length = None
		return hard_word_length
