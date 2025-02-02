import os
import shutil

import pandas as pd
import soundfile as sf
import praatio.textgrid


#files = [f for f in os.listdir('.') if f.endswith("Textfile")]
#for name in files:
#    shutil.move(name, name.removesuffix(" - Textfile") + ".TextGrid")



# read in and split audio

AUDIO_PATH = "../Für Tino in Tübingen"

textgrids = [name_ for name_ in os.listdir(AUDIO_PATH) if name_.endswith(".TextGrid")]

subject_ids = list()
labels = list()
sigs = list()
sample_rates = list()

for textgrid in textgrids:

    subject_id = textgrid.split("_")[3]

    try:
        path_audio = f"{AUDIO_PATH}/{textgrid.removesuffix('.TextGrid')}.wav" 
        full_audio, sr = sf.read(path_audio)
    except sf.LibsndfileError:
        path_audio = f"{AUDIO_PATH}/{textgrid.removesuffix('.TextGrid')}.WAV" 
        full_audio, sr = sf.read(path_audio)

    tg = praatio.textgrid.openTextgrid(f"{AUDIO_PATH}/{textgrid}", includeEmptyIntervals=False)

    try:
        tier = tg.getTier("word")
    except KeyError:
        tier = tg.getTier("substitution")
    for interval in tier:
        label = interval.label
        if not label:
            continue
        start_idx = int(interval.start * sr)
        end_idx = int(interval.end * sr)
        sig = full_audio[start_idx:(end_idx + 1)]

        subject_ids.append(subject_id)
        labels.append(label)
        sigs.append(sig)
        sample_rates.append(sr)

data = pd.DataFrame({"subject_id": subject_ids,
                     "label": labels,
                     "sig": sigs,
                     "sample_rate": sample_rates})

data.loc[data.subject_id == 'Wörtern.TextGrid', "subject_id"] = "Reference"


data.to_pickle('audios_essv2025.pickle')


# merge subject data into data
subject_data = pd.read_csv("../Für Tino in Tübingen/Tino2.csv")

percentage_speak_l1 = list()
for ii, row in data.iterrows():
    try:
        mask = subject_data['Subject.ID'] == int(row['subject_id'])
        speak_l1 = subject_data[mask]['speak_l1'].iloc[0]
    except ValueError:
        speak_l1 = None
    percentage_speak_l1.append(speak_l1)

data['speak_l1'] = percentage_speak_l1

data.to_pickle('audios_essv2025.pickle')



