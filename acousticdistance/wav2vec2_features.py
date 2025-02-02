import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor


MODEL_STR = "facebook/wav2vec2-large-960h-lv60-self"

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DTYPE = torch.float16

TARGET_SR = 16000

# https://huggingface.co/docs/transformers/model_doc/wav2vec2
# pip install -U flash-attn --no-build-isolation

model = Wav2Vec2Model.from_pretrained(MODEL_STR, torch_dtype=DTYPE).to(DEVICE)
processor = Wav2Vec2Processor.from_pretrained(MODEL_STR)


def wav2vec2_features(sig, sr, *, n_hidden_layer=9, slice_audio=False):
    """

    Returns
    -------
    features : (sequence, channels)

    """
    # resample to 16000 Hz
    if sr != TARGET_SR:
        sig_ = librosa.resample(sig, orig_sr=sr, target_sr=TARGET_SR)

    if slice_audio:
        assert len(sig_) >= 400
        sig_ = slice_(sig_)

    if slice_audio:
        input_w2v2 = processor.feature_extractor(sig_, sampling_rate=TARGET_SR)
    else:
        input_w2v2 = processor.feature_extractor([sig_,], sampling_rate=TARGET_SR)

    input_ = torch.tensor(np.array(input_w2v2['input_values']))
    input_ = input_.to(DTYPE).to(DEVICE)

    # is it batch first?
    output = model(input_, output_hidden_states=True)

    hidden_state = output[2]
    target_layer = hidden_state[n_hidden_layer]  # 0-based counting

    if slice_audio:
        features = target_layer.squeeze(1).detach().cpu().numpy()
    else:
        features = target_layer.squeeze(0).detach().cpu().numpy()
    return features


def slice_(wav):
    """
    slices the wav signal into chuncks of 400 smaples with a hop_size of 160.

    Assumes a sampling rate of 16000.

    """
    slices = list()
    hop_size = 160
    window_size = 400
    index = 0
    length = len(wav)
    while (index * hop_size + 400) <= length:
        t0 = (index * hop_size)
        t1 = t0 + window_size
        slices.append(wav[t0:t1])
        index += 1
    return slices

