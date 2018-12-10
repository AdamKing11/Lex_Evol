from Lexicon import *
from GUI import *

if __name__ == '__main__':
	print('\n\n\n')
	n_words = 1000
	n_phones = 4
	#phones = {'a' : 10, 'b' : 5, 'c' : 5, 'd' : 1}
	l = Lexicon(n_words, phones = n_phones, frequency_groups = 2, hard_start_length = 7)
	
	EvolGUI(l)
