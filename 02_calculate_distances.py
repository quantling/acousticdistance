import functools
import os
import pickle

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import soundfile as sf
from tqdm import tqdm

tqdm.pandas()

# local package
import acousticdistance


data = pd.read_pickle('audios_essv2025.pickle')


# do some inspections / visualization of the data
# ===============================================

print("subjects:")
print(data.subject_id.unique())

# -1 as there is one "Reference" speaker, who does not count
print(f'number of subjects: {len(data.subject_id.unique()) - 1}')


if not os.path.exists("audios"):
    os.mkdir("audios")

for ii, row in data.iterrows():
    name = f"audios/{row.label}_{row.subject_id}.flac"
    sf.write(name, row.sig, row.sample_rate)


# On the label side there are some labels with a followin "(f)" suffix.
# create clean labels:

print("Unclean labels:")
print(data['label'].unique())


data['label'] = data.label.str.removesuffix("(f)").str.strip()

print("Clean labels:")
print(data['label'].unique())



# calculate distances
# ===================

avg_dist = acousticdistance.avg_dist
dtw_dist = acousticdistance.dtw_dist

features = ("wave2vec2_sliced", "wave2vec2", "mfcc_edd")
#features = ("whisper", "wave2vec2_sliced", "wave2vec2", "mfcc_edd")
#features = ("mfcc_edd",)

baselines = dict()
mean_baselines = dict()

for feature in features:
    if feature == "whisper":
        feature_extractor = acousticdistance.whisper_features
    elif feature == "wave2vec2":
        feature_extractor = acousticdistance.wav2vec2_features
    elif feature == "wave2vec2_sliced":
        feature_extractor = functools.partial(acousticdistance.wav2vec2_features, slice_audio=True)
    elif feature == "mfcc_edd":
        feature_extractor = acousticdistance.mfcc_edd
    else:
        raise ValueError(f"You have to define a `feature_extractor` for {feature}.")

    print(f"extracting {feature}")
    data[feature] = data.progress_apply(lambda row: feature_extractor(row.sig, row.sample_rate), axis=1)
    # channel_normalize_features
    # do a global normalization
    global_mean_ch = np.stack(data[feature].apply(lambda feature: feature.mean(axis=0))).mean(axis=0)
    global_std_ch = np.stack(data[feature].apply(lambda feature: feature.std(axis=0))).mean(axis=0)
    data[feature + "_ch_norm"] = data.apply(lambda row: acousticdistance.channel_normalize(row[feature], means=global_mean_ch, stds=global_std_ch), axis=1)

    reference_features = dict()
    for ii, row in data[data.subject_id == 'Reference'].iterrows():
        reference_features[row.label] = row[feature]
    print(f"({feature}) calculate dtw distances")
    data[f"dtw_dist_{feature}"] = data.progress_apply(lambda row: dtw_dist(row[feature],
                                                                           reference_features[row.label]),
                                                      axis=1)
    data[f"avg_dist_{feature}"] = data.apply(lambda row: avg_dist(row[feature],
                                                                  reference_features[row.label]),
                                             axis=1)

    reference_features = dict()
    for ii, row in data[data.subject_id == 'Reference'].iterrows():
        reference_features[row.label] = row[feature + "_ch_norm"]
    print(f"({feature}) calculate dtw distances for ch_norm")
    data[f"dtw_dist_{feature}_ch_norm"] = data.progress_apply(lambda row: dtw_dist(row[feature + "_ch_norm"],
                                                                                   reference_features[row.label]),
                                                              axis=1)
    data[f"avg_dist_{feature}_ch_norm"] = data.apply(lambda row: avg_dist(row[feature + "_ch_norm"],
                                                                          reference_features[row.label]),
                                                     axis=1)



    baseline_strs = (f"dtw_dist_{feature}",
                     f"avg_dist_{feature}",
                     f"dtw_dist_{feature}_ch_norm",
                     f"avg_dist_{feature}_ch_norm")

    for baseline_str in baseline_strs:
        baselines[baseline_str] = list()
        if "dtw_dist" in baseline_str:
            distance = dtw_dist
        elif "avg_dist" in baseline_str:
            distance = avg_dist
        else:
            raise ValueError("Distance string not defined.")

        with tqdm(total=1000) as pbar:
            while len(baselines[baseline_str]) < 1000:
                row1 = data.sample().iloc[0]
                row2 = data.sample().iloc[0]

                if row1.label.split(".")[0] == row2.label.split(".")[0]:
                    continue

                pbar.update(1)

                feature_str = baseline_str.removeprefix("dtw_dist_").removeprefix("avg_dist_")
                baselines[baseline_str].append(distance(row1[feature_str], row2[feature_str]))
                del feature_str

        mean_baselines[baseline_str] = np.mean(baselines[baseline_str])
        data[f'mean_baseline_{baseline_str}'] = mean_baselines[baseline_str]


data.to_pickle('audios_essv2025_dists.pickle')

with open("baseline_data.pkl", "wb") as pfile:
     pickle.dump(baselines, pfile)


