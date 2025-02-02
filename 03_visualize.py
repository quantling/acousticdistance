import pandas as pd
from matplotlib import pyplot as plt

import seaborn as sns
sns.set_theme(style="ticks", palette="pastel")



data = pd.read_pickle('audios_essv2025_dists.pickle')

features = ("wave2vec2_sliced",
            "wave2vec2",
            "mfcc_edd",
            #"whisper",
            "wave2vec2_sliced_ch_norm",
            "wave2vec2_ch_norm",
            "mfcc_edd_ch_norm",
            #"whisper_ch_norm",
            )
#features = ("mfcc_edd",)

fig_labels = [f"{sub}_{label}" for sub, label in zip(data.subject_id, data.label)]

for feature in features:
    dists_dtw_to_ref = data[f"dtw_dist_{feature}"] / data[f"mean_baseline_dtw_dist_{feature}"]
    dists_avg_to_ref = data[f"avg_dist_{feature}"] / data[f"mean_baseline_avg_dist_{feature}"]
    plt.plot(dists_dtw_to_ref, label=f'dtw {feature}')
    plt.plot(dists_avg_to_ref, label=f'avg {feature}')

    # normalize to 1
plt.plot((0, len(dists_dtw_to_ref)), (1.0, 1.0), label='baseline')
    #plt.plot((0, len(dists_dtw_to_ref)), (mean_baselines_dtw[FEATURE], mean_baselines_dtw[FEATURE]), label='mean_dtw baseline')
    #plt.plot((0, len(dists_avg_to_ref)), (mean_baselines_avg[FEATURE], mean_baselines_avg[FEATURE]), label='mean_avg baseline')
plt.title("distances to reference")
plt.xticks(range(len(fig_labels)), fig_labels, rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig(f"dist_to_ref.pdf")
plt.clf()


# baseline plot

# create long format
long_norm_dist_dfs = list()
for feature in features:
    tmp_df = pd.DataFrame({"dist": data[f"dtw_dist_{feature}"] / data[f"mean_baseline_dtw_dist_{feature}"]})
    tmp_df['feature'] = feature
    tmp_df['dist type'] = 'DTW'
    long_norm_dist_dfs.append(tmp_df)
    tmp_df = pd.DataFrame({"dist": data[f"avg_dist_{feature}"] / data[f"mean_baseline_avg_dist_{feature}"]})
    tmp_df['feature'] = feature
    tmp_df['dist type'] = 'AVG'
    long_norm_dist_dfs.append(tmp_df)
del tmp_df
long_norm_dists = pd.concat(long_norm_dist_dfs)
del long_norm_dist_dfs


plt.figure(figsize=(5, 5))
sns.boxplot(x="feature", y='dist',
            hue="dist type", palette=["m", "g"],
            data=long_norm_dists)
sns.despine(offset=10, trim=True) 
features_txt = ("W2V2 sliced", "W2V2", "MFCC", "W2V2 sl.norm", "W2V2 norm", "MFCC norm")
plt.xticks(range(len(features)), features_txt, rotation=45)
plt.xlabel("")
plt.ylabel("normalised distance")
plt.plot((-0.5, 5.5), (1.0, 1.0), label='baseline', color='red', linestyle='dashed', linewidth=2)
plt.tight_layout()
plt.savefig(f"plots/baseline.pdf")
plt.clf()





trial_pseudoword = ('1', '2', '3', '4', '5', '6')
trial_l3 = ('7.1', '7.2', '8.1', '8.2', '9.1', '9.2', '10.1', '10.2', '11.1',
        '11.2', '12.1', '12.2', '13.1', '13.2')

data['condition'] = None
data.loc[data.label.isin(trial_pseudoword), 'condition'] = 'pseudoword'
data.loc[data.label.isin(trial_l3), 'condition'] = 'Slovak (l3)'


multilingual = (30., 40., 50., 60., 70.)
monolingual = (0., 10., 20., 80., 90., 100.)
data['group'] = None
data.loc[data.speak_l1.isin(multilingual), 'group'] = 'multilingual'
data.loc[data.speak_l1.isin(monolingual), 'group'] = 'monolingual'


response_variables = ('dtw_dist_wave2vec2_sliced',
                      'avg_dist_wave2vec2_sliced',
                      'dtw_dist_wave2vec2',
                      'avg_dist_wave2vec2',
                      'dtw_dist_mfcc_edd',
                      'avg_dist_mfcc_edd',
                      #'dtw_dist_whisper',
                      #'avg_dist_whisper',
                      'dtw_dist_wave2vec2_sliced_ch_norm',
                      'avg_dist_wave2vec2_sliced_ch_norm',
                      'dtw_dist_wave2vec2_ch_norm',
                      'avg_dist_wave2vec2_ch_norm',
                      'dtw_dist_mfcc_edd_ch_norm',
                      'avg_dist_mfcc_edd_ch_norm',
                      #'dtw_dist_whisper_ch_norm',
                      #'avg_dist_whisper_ch_norm',
                      )

for response_var in response_variables:
    sns.boxplot(x="condition", y=response_var,
                hue="group", palette=["m", "g"],
                data=data)
    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    plt.savefig(f"plots/boxplot_{response_var}.pdf")
    plt.clf()

