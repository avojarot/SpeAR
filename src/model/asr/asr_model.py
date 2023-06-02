from math import ceil

import nemo.collections.asr as nemo_asr
import numpy as np
import torch

sr = 16_000
buffer_len = 1.6
chunk_len = 0.8
total_buffer = round(buffer_len * sr)
overhead_len = round((buffer_len - chunk_len) * sr)
model_stride = 4


class AsrModel:
    def __init__(self, device):
        asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
            "theodotus/stt_uk_squeezeformer_ctc_ml", map_location=device
        )

        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0
        asr_model.eval()
        asr_model.encoder.freeze()
        asr_model.decoder.freeze()
        self.asr_model = asr_model
        self.device = device

        self.model_stride_in_secs = (
            asr_model.cfg.preprocessor.window_stride * model_stride
        )
        self.tokens_per_chunk = ceil(chunk_len / self.model_stride_in_secs)
        self.mid_delay = ceil(
            (chunk_len + (buffer_len - chunk_len) / 2) / self.model_stride_in_secs
        )

    def model(self, audio_16k):
        logits, logits_len, greedy_predictions = self.asr_model.forward(
            input_signal=torch.tensor([audio_16k]).to(self.device),
            input_signal_length=torch.tensor([len(audio_16k)]).to(self.device),
        )
        return logits.cpu()

    def decode_predictions(self, logits_list):
        logits_len = logits_list[0].shape[1]
        # cut overhead
        cutted_logits = []
        for idx in range(len(logits_list)):
            start_cut = 0 if (idx == 0) else logits_len - 1 - self.mid_delay
            end_cut = (
                -1
                if (idx == len(logits_list) - 1)
                else logits_len - 1 - self.mid_delay + self.tokens_per_chunk
            )
            logits = logits_list[idx][:, start_cut:end_cut]
            cutted_logits.append(logits)

        # join
        logits = torch.cat(cutted_logits, axis=1)
        logits_len = torch.tensor([logits.shape[1]])
        (
            current_hypotheses,
            all_hyp,
        ) = self.asr_model.decoding.ctc_decoder_predictions_tensor(
            logits,
            decoder_lengths=logits_len,
            return_hypotheses=False,
        )

        return current_hypotheses[0]

    def transcribe(self, audio, state):
        if state is None:
            state = [np.array([], dtype=np.float32), []]

        # join to audio sequence
        state[0] = np.concatenate([state[0], audio])

        while len(state[0]) > total_buffer:
            buffer = state[0][:total_buffer]
            state[0] = state[0][total_buffer - overhead_len :]
            # run model
            logits = self.model(buffer)
            # add logits
            state[1].append(logits)

        if len(state[1]) == 0:
            text = ""
        else:
            text = self.decode_predictions(state[1])
        return text
