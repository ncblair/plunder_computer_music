import os
from math import ceil
class Params:
	def __init__(self):
		# self.genres= ["Electronic", "Hip-Hop", "Instrumental", "Rock", "Pop", "Folk", "Experimental", "International"]
		self.genres = ["Pop"]
		self.cuda = True
		self.segsize=.3
		self.low_feather_size = .1
		self.high_feather_size = .05
		self.samplerate=44100
		self.stride=3
		self.nb_samples = 1321967
		self.num_frequencies=5

		self.cliplength=30
		self.maxlength = int(self.nb_samples / self.samplerate / self.segsize // self.stride) + 1


		self.FMA_DIR = os.environ.get('FMA_DIR')
		self.meta_dir = f"{self.FMA_DIR}/fma_metadata"
		self.audio_dir = f"{self.FMA_DIR}/fma_small"

		testname = "dance_song"
		# self.testfilename = f"{self.audio_dir}/013/{testname}.mp3"
		self.savefilename = f"results/{testname}_{str(self.segsize).replace('.', '')}.wav"
		self.testfilename=f"data/{testname}.wav"
		self.datafilename = f"data/{''.join(self.genres)}_{str(self.segsize).replace('.', '')}_{self.stride}"
		self.idxfilename = f"data/idx{''.join(self.genres)}_{str(self.segsize).replace('.', '')}_{self.stride}"

p = Params()
