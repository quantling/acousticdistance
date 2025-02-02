import librosa
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

TARGET_SR = 16000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the model and processor
MODEL_STR = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_STR, torch_dtype=DTYPE, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(DEVICE)

processor = AutoProcessor.from_pretrained(MODEL_STR)



def whisper_features(sig, sr, *, n_hidden_layer=9):
    """

    Returns
    -------
    features : (sequence, channels)

    """
    print("WARNING: this code seems to still give incorrect / useless features.")
    # resample to 16000 Hz
    if sr != TARGET_SR:
        sig_ = librosa.resample(sig, orig_sr=sr, target_sr=TARGET_SR)


    inputs = processor(sig_, return_tensors="pt", sampling_rate=TARGET_SR)


    input_ = torch.tensor(np.array(inputs['input_features']))
    input_ = input_.to(DTYPE).to(DEVICE)


    # Ensure model is in evaluation mode
    model.eval()

    # Provide a dummy tensor for decoder_input_ids
    # The shape should match that of the encoder inputs, but we'll just pass a tensor of zeros
    dummy_decoder_input = torch.zeros(input_.shape[0], input_.shape[1], dtype=torch.long).to(DEVICE)


    with torch.no_grad():
        output = model(input_, decoder_input_ids=dummy_decoder_input, output_hidden_states=True)

    # odict_keys(['logits', 'past_key_values', 'decoder_hidden_states', 'encoder_last_hidden_state', 'encoder_hidden_states'])
    hidden_state = output['encoder_last_hidden_state']
    #target_layer = hidden_state[n_hidden_layer]  # 0-based counting

    features = hidden_state.squeeze(0).detach().cpu().numpy()
    return features

