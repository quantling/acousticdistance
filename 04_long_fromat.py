import pandas as pd


data = pd.read_pickle('audios_essv2025_dists.pickle')

features = ("wave2vec2_sliced",
            "wave2vec2",
            "mfcc_edd",
            "wave2vec2_sliced_ch_norm",
            "wave2vec2_ch_norm",
            "mfcc_edd_ch_norm",
            )

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

# create long format
tmp_df_id_labels = data[['subject_id', 'label', 'speak_l1', 'condition', 'group']]
long_norm_dist_dfs = list()
for feature in features:
    tmp_df = pd.DataFrame({"dist": data[f"dtw_dist_{feature}"] / data[f"mean_baseline_dtw_dist_{feature}"]})
    tmp_df['feature'] = feature
    tmp_df['dist type'] = 'DTW'
    tmp_df = pd.concat((tmp_df_id_labels, tmp_df), axis=1)
    long_norm_dist_dfs.append(tmp_df)
    tmp_df = pd.DataFrame({"dist": data[f"avg_dist_{feature}"] / data[f"mean_baseline_avg_dist_{feature}"]})
    tmp_df['feature'] = feature
    tmp_df['dist type'] = 'AVG'
    tmp_df = pd.concat((tmp_df_id_labels, tmp_df), axis=1)
    long_norm_dist_dfs.append(tmp_df)
del tmp_df, tmp_df_id_labels
long_norm_dists = pd.concat(long_norm_dist_dfs)
del long_norm_dist_dfs

long_norm_dists.to_csv("dists_long_format.csv", index=False)

