"""
paper1: Benchmarking Multimodal Sentiment Analysis
paper2: Recognizing Emotions in Video Using Multimodal DNN Feature Fusion
From: https://github.com/rhoposit/MultimodalDNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..subNets.Conformer import conformer
from ..subNets.Conformer_cat import conformer_cat
from ..subNets import BertTextEncoder
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
# from speechbrain.lobes.models.ECAPA_TDNN import BatchNorm1d
from ..subNets import AlignSubNet

__all__ = ['MAVMFA_Conformer']




class MAVMFA_Conformer(nn.Module):
    """

    """

    def __init__(self, args):
        super(MAVMFA_Conformer, self).__init__()
        text_in, audio_in, video_in = args.feature_dims
        # in_size = text_in + audio_in + video_in

        input_len = args.seq_lens
        fusion_hidden_size, text_hidden_size, audio_hidden_size, video_hidden_size = args.hidden_dims
        num_layers = args.num_layers
        fusion_dropout, text_dropout, audio_dropout, video_dropout = args.dropouts
        output_dim = args.num_classes if args.train_mode == "classification" else 1

        self.bertmodel = BertTextEncoder(use_finetune=False, transformers='bert', pretrained='bert-base-chinese')
        self.alignNet = AlignSubNet(args, 'avg_pool')

        self.aConformer = conformer_cat(n_mels=audio_in, num_blocks=6, output_size=audio_hidden_size, embedding_dim=64,
                                        input_layer="conv2d", pos_enc_layer_type="rel_pos")
        self.vConformer = conformer_cat(n_mels=video_in, num_blocks=6, output_size=video_hidden_size, embedding_dim=64,
                                        input_layer="conv2d", pos_enc_layer_type="rel_pos")


        #
        a_encoder_layer = nn.TransformerEncoderLayer(d_model=audio_hidden_size * 6, nhead=2, batch_first=True)
        self.a_transformer_encoder = nn.TransformerEncoder(a_encoder_layer, num_layers=num_layers)

        v_encoder_layer = nn.TransformerEncoderLayer(d_model=video_hidden_size * 6, nhead=2, batch_first=True)
        self.v_transformer_encoder = nn.TransformerEncoder(v_encoder_layer, num_layers=num_layers)


        self.audio_pooling = AttentiveStatisticsPooling(audio_hidden_size * 6)
        self.audio_dropout = nn.Dropout(audio_dropout)
        self.audio_linear = nn.Linear(2 * audio_hidden_size * 6, audio_hidden_size)
        self.audio_out = nn.Linear(audio_hidden_size, output_dim)


        self.video_pooling = AttentiveStatisticsPooling(video_hidden_size * 6)
        self.video_dropout = nn.Dropout(video_dropout)
        self.video_linear = nn.Linear(2 * video_hidden_size * 6, video_hidden_size)
        self.video_out = nn.Linear(video_hidden_size, output_dim)


        self.text_pooling = AttentiveStatisticsPooling(text_in)
        self.text_dropout = nn.Dropout(text_dropout)
        self.text_linear = nn.Linear(text_in * 2, text_hidden_size)
        self.text_out = nn.Linear(text_hidden_size, output_dim)


        self.fusion_pooling = AttentiveStatisticsPooling(2 * fusion_hidden_size * 6 + 768)
        self.fusion_dropout = nn.Dropout(fusion_dropout)
        self.fusion_linear = nn.Linear((2 * fusion_hidden_size * 6 + 768) * 2, fusion_hidden_size)
        self.fusion_out = nn.Linear(fusion_hidden_size, output_dim)


    def forward(self, text_x, audio_x, video_x, text_o, audio_o, video_o):
        _, a = self.aConformer(audio_o)
        _, v = self.vConformer(video_o)
        t = self.bertmodel(text_o)

        #仅text输出
        x_t = t.permute(0, 2, 1)
        x_t = self.text_pooling(x_t)
        x_t = x_t.permute(0, 2, 1)
        x_t = x_t.squeeze(1)
        x_t = F.gelu(self.text_linear(x_t))
        x_t = self.text_dropout(x_t)
        output_text = self.text_out(x_t)


        # audio拼接+SA
        a = torch.cat([a[0], a[1], a[2], a[3], a[4], a[5]], dim=-1)
        a = self.a_transformer_encoder(a)

        #仅audio输出
        x_a = a.permute(0, 2, 1)
        x_a = self.audio_pooling(x_a)
        x_a = x_a.permute(0, 2, 1)
        x_a = x_a.squeeze(1)
        x_a = F.gelu(self.audio_linear(x_a))
        x_a = self.audio_dropout(x_a)
        output_audio = self.audio_out(x_a)

        #video拼接+SA
        v = torch.cat([v[0], v[1], v[2], v[3], v[4], v[5]], dim=-1)
        v = self.v_transformer_encoder(v)

        #仅video输出
        x_v = v.permute(0, 2, 1)
        x_v = self.video_pooling(x_v)
        x_v = x_v.permute(0, 2, 1)
        x_v = x_v.squeeze(1)
        x_v = F.gelu(self.video_linear(x_v))
        x_v = self.video_dropout(x_v)
        output_video = self.video_out(x_v)

        #fusion输出
        t, a, v = self.alignNet(t, a, v)
        x = torch.cat([a, v, t], dim=-1)
        x = x.permute(0, 2, 1)
        x = self.fusion_pooling(x)
        x = x.permute(0, 2, 1)
        x = x.squeeze(1)

        # x = self.layernorm(x)
        x = self.fusion_dropout(x)
        x = F.gelu(self.fusion_linear(x))
        x = self.fusion_dropout(x)
        output_fusion = self.fusion_out(x)


        res = {
            'M': output_fusion,
            'A': output_audio,
            'V': output_video,
            'T': output_text
        }
        return res

