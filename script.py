import os
import librosa
import math
import json
SAMPLE_RATE=22_050
DURATION=30
SAMPLES_PER_TRACK=SAMPLE_RATE*DURATION
dataset_path='genres'
json_path='data1.json'
def save_mfcc(dataset_path,json_path,n_mfcc=13,n_fft=2048,hop_length=512,num_segments=5):
	#dict
	data={"mapping":[],
			"mfcc":[],
			"labels":[]

			}	
	num_samples_per_segment=int(SAMPLES_PER_TRACK/num_segments)
	expected_num_mfcc_vectors_per_segment=math.ceil(num_samples_per_segment/hop_length)

	for i,(dirpath,dirname,filenames) in enumerate(os.walk(dataset_path)):
		if dirpath is not dataset_path:
			dirpath_components=dirpath.split("/")
			semantic_label=dirpath_components[-1]
			data["mapping"].append(semantic_label)
			print("\n Processing {}".format(semantic_label))

			for f in filenames:
				file_path=os.path.join(dirpath,f)
				signal,sr=librosa.load(file_path,sr=SAMPLE_RATE)

				for s in range(num_segments):
					start_sample=num_samples_per_segment*s
					finish_sample=start_sample+num_samples_per_segment


					


					mfcc=librosa.feature.mfcc(signal[start_sample:finish_sample],sr=sr,n_fft=n_fft,n_mfcc=n_mfcc,hop_length=hop_length)

					mfcc=mfcc.T
					if len(mfcc)==expected_num_mfcc_vectors_per_segment:
						data["mfcc"].append(mfcc.tolist())
						data["labels"].append(i-1)
						print("{},segment:{}".format(file_path,s))


	with open(json_path,"w") as fp:
		json.dump(data,fp,indent=4)


if __name__=="__main__":
	save_mfcc(dataset_path,json_path,num_segments=10)


