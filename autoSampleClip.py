import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.cuda.empty_cache()
from tqdm import tqdm as tqdm
import utils
from params import p
import matplotlib.pyplot as plt
from data_handling import load_audio, decompose, save_audio



def autosample(audiofilename, genres, num_frequencies=3, segsize=.5, stride=3, low_feather_size=.1, high_feather_size=.001):
	audio = load_audio(audiofilename)
	segment_length = int(segsize*p.samplerate)
	# audio = audio[:-(len(audio)%segment_length)]
	audio_split = torch.split(audio, segment_length)
	fft_split = [torch.rfft(seg, 1) for idx, seg in enumerate(audio_split)]
	partitions = [decompose(seg, num_frequencies) for seg in fft_split]
	recreation_data = torch.zeros(len(audio_split), num_frequencies, 4) # genre, song_id, seg_idx, score


	tracks = utils.load(f'{p.meta_dir}/tracks.csv')

	small = tracks['set', 'subset'] <= 'small'
	train = tracks['set', 'split'] == 'training'

	for g, genre in enumerate(tqdm(genres)):
		dataset = torch.load(f"data/{genre}_{str(segsize).replace('.', '')}_{stride}")
		idx_to_id = torch.load(f"data/idx{genre}_{str(segsize).replace('.', '')}_{stride}")

		for i, (part, seg) in enumerate(zip(partitions, fft_split)):
			differences = torch.norm(dataset-seg, dim=3)
			per_freq_differences = [torch.norm(differences[:, :, part[f]:part[f+1]], dim=2) for f in range(len(part)-1)]
			best_per_freq = [torch.argmin(diff) for diff in per_freq_differences]
			best_per_freq = [[k//dataset.shape[1], k%dataset.shape[1]] for k in best_per_freq]

			for j, (song_idx, seg_idx) in enumerate(best_per_freq):
				value = per_freq_differences[j][best_per_freq[j][0], best_per_freq[j][1]]
				if recreation_data[i, j, 0] == 0 or recreation_data[i, j, 3] > value:
					song_id = idx_to_id[song_idx.cpu().numpy()]
					recreation_data[i, j] = torch.Tensor([g, song_id, seg_idx, value])

	result = torch.zeros_like(audio)
	if num_frequencies != 1:
		feather_sizes = [low_feather_size - i * (low_feather_size - high_feather_size) \
											/ (num_frequencies - 1) for i in range(num_frequencies)]
	else:
		feather_sizes = [low_feather_size]

	feather_length = int(low_feather_size * p.samplerate)

	ramps = []
	flens = []
	for fsize in feather_sizes:
		flen = int(fsize*p.samplerate//2*2)
		ramps.append(torch.cat([torch.arange(flen).float()/flen, torch.ones(segment_length-flen).float(), \
						torch.arange(flen, 0, -1).float()/flen]))
		flens.append(flen)

	for i, (segment, part) in enumerate(tqdm(zip(recreation_data, partitions), total=len(recreation_data))):
		res_start = i*segment_length - feather_length//2
		res_start_t = max(res_start, 0)
		res_end = (i+1)*segment_length + feather_length//2
		res_end_t = min(res_end, result.shape[0])
		for j, (freq_seg, ramp, flen) in enumerate(zip(segment, ramps, flens)):

			#load audio
			song_id = int(freq_seg[1])
			filepath = utils.get_audio_path(p.audio_dir, song_id)
			tosample = load_audio(filepath)

			# get relevant region
			loc_start = int(freq_seg[2])*segment_length - flen//2
			loc_end = loc_start + segment_length + flen
			loc_start_t = int(max(loc_start, 0))
			loc_end_t = int(min(loc_end, tosample.shape[0]))
			to_sample_segment = torch.zeros(segment_length + feather_length)

			s = loc_start_t - loc_start + feather_length//2 - flen//2
			e = to_sample_segment.shape[0] - (loc_end-loc_end_t) - (feather_length//2 - flen//2)
			to_sample_segment[s:e] = tosample[loc_start_t:loc_end_t]

			#band pass filter
			to_sample_fourier = torch.rfft(to_sample_segment, 1)
			to_sample_fourier[:part[j]] = 0
			to_sample_fourier[part[j+1]:] = 0
			filtered_segment = torch.irfft(to_sample_fourier, 1, signal_sizes=to_sample_segment.shape)

			#feather edges
			s1 = feather_length//2 - flen//2
			e1 = filtered_segment.shape[0] - s1
			filtered_segment[s1:e1] *= ramp

			# add to result
			result[res_start_t:res_end_t] += filtered_segment[res_start_t - res_start:filtered_segment.shape[0] - (res_end - res_end_t)]

	return result, recreation_data



if __name__ == "__main__":
	result, recreation_data = autosample(p.testfilename, p.genres, num_frequencies=p.num_frequencies, segsize=p.segsize, stride=p.stride, 
						low_feather_size=p.low_feather_size, high_feather_size=p.high_feather_size)
	save_audio(result, p.savefilename)


