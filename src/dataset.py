# Most of the codes in this file are copied from here: <https://github.com/gyunggyung/KoGPT2-FineTuning>
# Origianl Author: gyunggyung <https://github.com/gyunggyung>
#
# Modified by: Yeonwoo Sung

from torch.utils.data import Dataset
from gluonnlp.data import SentencepieceTokenizer
import gluonnlp
import numpy as np

from prepare_pytorch_kogpt2 import load_kogpt2_model
from utils import download, tokenizer, get_tokenizer



class KoLyricsDataset(Dataset):
	"""Korean Lyrics dataset"""

	def __init__(self, file_path, vocab, tokenizer):
		self.file_path = file_path
		self.data = []
		self.vocab = vocab
		self.tokenizer = tokenizer

		# open text file
		file = open(self.file_path, 'r', encoding='utf-8')

		lines = file.read()
		lines = lines.split("\n")

		datasets = []
		now = ""
		for i, line in enumerate(lines):
			if i % 30 == 0 and i != 0:
				datasets.append(now)
				now = ""
				continue
			now = now + "\n" + line


		print("tokenizer ending")

		# use for loop to iterate array of lines
		for line in datasets:
			if not line:
				break
			if len(line) < 3:
				continue

			toeknized_line = tokenizer(line[:-1])

			index_of_words = [vocab[vocab.bos_token], ] + vocab[toeknized_line] + [vocab[vocab.eos_token]]
			self.data.append(index_of_words)

		print(np.shape(self.data))

		file.close()

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		item = self.data[index]
		return item



if __name__ == "__main__":
    tok_path = get_tokenizer()
    tok = SentencepieceTokenizer(tok_path,  num_best=0, alpha=0)
    
    # load model and vocabularies
    model, vocab = load_kogpt2_model()

    # create KoLyricsDataset instance
    dataset = KoLyricsDataset('../dataset/lyrics_dataset.txt', vocab, tok)
