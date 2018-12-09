import string, random, sys

from pprint import pprint
from utils import *
try:
	import numpy as np
except:
	print('please install `numpy`!')
	sys.exit()


class Word:
	# word object, pretty straight forward
	def __init__(self, w, f, p):
		self.word = w
		self.frequency = f
		self.prob = p
		self.unigram = -log2(p)
		self.group = 1

		self.reset_si()

	def __repr__(self):
		return self.word

	def __str__(self):
		return self.word

	def __len__(self):
		return len(self.word)

	def __getitem__(self, i):
		return self.word[i]

	def reset_si(self):
		self.si = [0. for _ in range(len(self.word))]

	def sub_char(self, i, c):
		self.word = self.word[:i] + c + self.word[i+1:]
		self.reset_si()

	def remove_char(self, i):
		self.word = self.word[:i] + self.word[i+1:]
		self.reset_si()

	def add_prefix(self, prefix, hard_max_length):
		self.word = prefix + self.word
		self.word = self.word[:hard_max_length]
		self.reset_si()

	def rebuild_word(self, new_word):
		self.word = new_word
		self.reset_si()

class Lexicon:
	# lexicon object, contains all words and methods for calculating seginfo and removing segments
	def __init__(self, N, phones = 6, z = 2., hard_start_length = None, hard_max_length = 12, frequency_groups = 1):
		
		self.zipf_constant = z
		self.hard_start_length = hard_start_length
		self.hard_max_length = hard_max_length
		self.frequency_groups = frequency_groups
		self.phones = phones
		# generate a zipfian dist for word frequencies
		self.frequencies = self.sample_zipf(N)
		self.total_frequency = np.sum(self.frequencies)
		# determine unigram word prob
		p = self.frequencies / np.sum(self.frequencies)
		
		self.lengths = self.word_p_to_length(p)

		# build words
		# not important now, but allows to assigning different probabilities for differench symbols, e.g. `a` is more likely than `b`, etc
		if type(phones) == int:
			self.phones = dict((c, 1) for c in string.ascii_lowercase[:phones])
		elif type(phones) == set:
			self.phones = dict((c, 1) for c in phones)

		# get two lists: possible phones and their probabilities
		
		self.words = []
		unique_forms = set([''])
		phone_seq = ''
		for freq, l in zip(self.frequencies, self.lengths):
			# generate strings until we get one that isn't already in the lexicon
			while phone_seq in unique_forms:
				# make word as string of random symbols, determined by the symbol prior (at this point, equiprobable)
				phone_seq = sample_from_p_dict(self.phones, l)
		
			unique_forms.add(phone_seq)

			
			w = Word(phone_seq, freq, self.frequency_to_p(freq))
			self.words.append(w)

		self.words = sorted(self.words, key = lambda w : w.unigram)
		if frequency_groups > 1:
			split_by_frequency = np.array_split(self.words, frequency_groups)
			for i, group in enumerate(split_by_frequency):
				for word in group:
					word.group = i + 1
		# one lexicon is complete, calculate seg info for lexicon
		self.calc_segmental_info(total_recount = True)

	def __len__(self):
		return len(self.words)

	def frequency_to_p(self, frequency):
		return frequency / self.total_frequency

	def sample_zipf(self, N = 1):
		if N == 1:
			return np.random.zipf(self.zipf_constant, N)[0]
		else:
			return np.random.zipf(self.zipf_constant, N)

	def word_p_to_length(self, p):
		if self.hard_start_length:
			# set all lengths to single value
			lengths = np.ones(len(p), dtype = 'int') * self.hard_start_length
			lengths += np.random.randint(2,size = len(lengths))
		else:
			# set length to be a logarithmic function of the Zipfian distribution
			lengths = np.array(np.minimum(np.floor(-np.log(p)) + 1, self.hard_max_length), dtype = 'int')
		
		if len(lengths) == 1:
			return lengths[0]
		else:
			return lengths

	def add_word_to_cohorts(self, word):
		for i in range(1, len(word) + 1):
			prefix = word[:i]
			symbol = word[i-1]
			self.cohort_sizes[prefix] = self.cohort_sizes.get(prefix, 0) + word.frequency
			self.symbol_counts[symbol] = self.symbol_counts.get(symbol, 0) + word.frequency

	def delete_word_from_cohorts(self, word):
		for i in range(1, len(word) + 1):
			prefix = word[:i]
			symbol = word[i-1]
			self.cohort_sizes[prefix] = self.cohort_sizes.get(prefix, word.frequency) - word.frequency
			self.symbol_counts[symbol] = self.symbol_counts.get(symbol, word.frequency) - word.frequency

	def calc_segmental_info(self, total_recount = False):
		self.entropy = 0
		self.max_si = 0
		# count size of cohorts
		if total_recount:
			self.symbol_counts = {}
			self.cohort_sizes = {}
			for w in self.words:
				self.add_word_to_cohorts(w)
			self.seg_ps = dict_to_p_dist(self.symbol_counts)

		for w in self.words:
		# determine seg info as -log2 (count(s) / count(cohort))
			for i, c in enumerate(w):
				if i == 0:
					previous_cohort = self.total_frequency
				else:
					previous_cohort = self.cohort_sizes[w[:i]]
				
				current_cohort = self.cohort_sizes[w[:i+1]]
				
				if current_cohort == w.frequency:
					freq_mod = 0
				else:
					freq_mod = w.frequency

				seg_p = (current_cohort - freq_mod) / (previous_cohort - freq_mod)
				seg_h = -log2(seg_p)
				
				w.si[i] = seg_h
				self.max_si = max(self.max_si, seg_h)

				total_p = seg_p * w.prob
				self.entropy += exp_h(total_p)

	def word_lengths(self, which_group = None):
		word_lengths = []
		if which_group:
			words_for_lengths = [w for w in self.words if w.group == which_group]
		else:
			words_for_lengths = [w for w in self.words]

		for w in words_for_lengths:
			word_lengths.append(len(w))
		return word_lengths

	def lengths_and_unigrams(self, which_group = None):
		length_unigram_tuples = []
		if which_group:
			words_to = [w for w in self.words if w.group == which_group]
		else:
			words_to = [w for w in self.words]

		for w in words_to:
			length_unigram_tuples.append((len(w), w.unigram))

		return length_unigram_tuples
		
	def avg_segmental_info(self, which_group = None):
		if which_group:
			words_to_avg = [w for w in self.words if w.group == which_group]
		else:
			words_to_avg = [w for w in self.words]

		si_matrix = np.zeros((len(words_to_avg), self.hard_max_length))
		word_lens = np.zeros(self.hard_max_length)
		for i, word in enumerate(words_to_avg):
			for j, seg_info in enumerate(word.si):
				si_matrix[i,j] = seg_info
				word_lens[j] += 1
		return np.sum(si_matrix, axis = 0) / word_lens

	def positional_entropy(self, position, which_group = None):
		if which_group:
			words_to = [w for w in self.words if w.group == which_group]
		else:
			words_to = [w for w in self.words]

		counts = {}
		for i, word in enumerate(words_to):
			seg = word[position]
			counts[seg] = counts.get(seg, 0) + word.frequency
		ps = dict_to_p_dist(counts)
		return H(ps.values()) 

	def edge_entropies(self):

		firsts, lasts = [], []
		for i in range(self.frequency_groups):
			#f, l = self.first_last_avg_information(i + 1)
			f = self.positional_entropy(0, which_group = i + 1)
			l = self.positional_entropy(-1, which_group = i + 1)
			firsts.append(f)
			lasts.append(l)


		return firsts, lasts

	def change_segs(self, word_E = 1.5, seg_E = 1.5, symbol_E = 1., merger_p = .5):
		
		# where I get creative, a million ways to do this 
		infos = [w.unigram for w in self.words]
		
		unique_forms = set([w.word for w in self.words])
		# determine the max h**2 for the whole lexicon, the most information segment
		max_h2 = max([h ** word_E for h in infos])
		sum_h2 = sum([h ** word_E for h in infos])

		n_merged, n_removed = 0, 0
		for i, word in enumerate(self.words):
			# pick random number, N, between 0-1
			# if N is greater than h**2 / max_h2, remove
			R = np.random.rand()
			alter_word = R > max(min(.95, (word.unigram ** word_E) / max_h2),.01)
			
			### bonus
			# only remove 1/5 of segments that pass above
			#alter_word = alter_word and np.random.rand() >= .8
			
			if alter_word:
				# remove the old word from the count of cohorts
				self.delete_word_from_cohorts(word)
				unique_forms.remove(word.word)
				made_change = False
				# choose segment to change
				max_si2 = max([h ** seg_E for h in word.si])

				# get the segments of the word and their segmental info value
				# randomly shuffle them and go through them
				seg_pos_and_h = list(enumerate(word.si))
				random.shuffle(seg_pos_and_h)
				for j, h in seg_pos_and_h:
					R = np.random.rand()
					alter_seg = R > max(min(.99, (h ** seg_E) / max_si2),.05) 
					
					if alter_seg:
						merger = np.random.rand() < merger_p
						if merger:
							made_change = True
							n_merged += 1
							word.sub_char(j, sample_from_p_dict(self.seg_ps, 1))
						else:
							made_change = True
							n_removed += 1
							word.remove_char(j)
							if len(word) == 0:
								new_length = self.word_p_to_length([word.prob])
								new_word = sample_from_p_dict(self.seg_ps, new_length)
								word.rebuild_word(new_word)
					# if we change a segment, stop looping through segments
					if made_change:	
						break
				# if the new word is something already in the lexicon
				while made_change and word.word in unique_forms:
					short_enough_words = [w for w in self.words if self.hard_max_length >= len(w) + len(word)]
					
					for i in range(1,2):
						similar_freq_words = [w for w in short_enough_words if w.unigram <= word.unigram + i and w.unigram >= word.unigram - i]
						if len(similar_freq_words) > 0:
							short_enough_words = similar_freq_words
							break
					
					if len(short_enough_words) > 0:
						shortest_enough_word = min([len(w) for w in short_enough_words])
						short_enough_words = [w for w in short_enough_words if len(w) == shortest_enough_word]
						prefix = random.choice(short_enough_words).word
					else:
						prefix = sample_from_p_dict(self.seg_ps, 1)

					# add a prefix, built of another random word	
					word.add_prefix(prefix, self.hard_max_length)

				unique_forms.add(word.word)
				# add the new word to the cohort counts
				self.add_word_to_cohorts(word)

		self.seg_ps = dict_to_p_dist(self.symbol_counts, E = symbol_E)
		self.calc_segmental_info()