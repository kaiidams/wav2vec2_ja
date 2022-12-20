from dataclasses import dataclass
from torchaudio.pipelines import Wav2Vec2ASRBundle
from torchaudio._internal import load_state_dict_from_url


def _get_ja_labels():
    return (
        '_',
        '_',
        '_',
        'A',
        'I',
        'O',
        'K',
        'U',
        'E',
        'T',
        'N',
        'R',
        'M',
        'S',
        'NN',
        '|',
        'SH',
        'D',
        'G',
        'O:',
        'W',
        'Q',
        'Y',
        'B',
        'H',
        'TS',
        'J',
        'CH',
        'E:',
        'U:',
        'Z',
        'P',
        'F',
        'I:',
        'KY',
        'A:',
        'RY',
        'HY',
        'GY',
        'NY',
        'MY',
        'BY',
        'PY',
        'DY'
    )


@dataclass
class MyWav2Vec2ASRBundle(Wav2Vec2ASRBundle):
    def _get_state_dict(self, dl_kwargs):
        url = f"https://github.com/kaiidams/wav2vec2_ja/releases/download/{self._path}"
        dl_kwargs = {} if dl_kwargs is None else dl_kwargs
        state_dict = load_state_dict_from_url(url, **dl_kwargs)
        return state_dict


WAV2VEC2_ASR_BASE_CV12JA = MyWav2Vec2ASRBundle(
    "v0.1/wav2vec2_fairseq_base_ls960_asr_cv12ja.pth",
    {
        "extractor_mode": "group_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": False,
        "encoder_embed_dim": 768,
        "encoder_projection_dropout": 0.1,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 12,
        "encoder_num_heads": 12,
        "encoder_attention_dropout": 0.1,
        "encoder_ff_interm_features": 3072,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.1,
        "encoder_layer_norm_first": False,
        "encoder_layer_drop": 0.05,
        "aux_num_out": 45,
    },
    _labels=_get_ja_labels(),
    _sample_rate=16000,
)