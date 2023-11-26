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
#from speechbrain.lobes.models.ECAPA_TDNN import BatchNorm1d
from ..subNets import AlignSubNet

__all__ = ['AVMFA_Conformer']


class AVMFA_Conformer(nn.Module):
    """

    """

    def __init__(self, args):
        super(AVMFA_Conformer, self).__init__()
        text_in, audio_in, video_in = args.feature_dims
        #in_size = text_in + audio_in + video_in

        av_in_size = audio_in + video_in

        input_len = args.seq_lens
        hidden_size = args.hidden_dims
        num_layers = args.num_layers
        dropout = args.dropout
        output_dim = args.num_classes if args.train_mode == "classification" else 1

        self.bertmodel = BertTextEncoder(use_finetune=False, transformers='bert', pretrained='bert-base-uncased')
        self.alignNet = AlignSubNet(args, 'avg_pool')


        self.aConformer = conformer_cat(n_mels=audio_in, num_blocks=6, output_size=64, embedding_dim=64,
                                    input_layer="conv2d", pos_enc_layer_type="rel_pos")
        self.vConformer = conformer_cat(n_mels=video_in, num_blocks=6, output_size=64, embedding_dim=64,
                                    input_layer="conv2d", pos_enc_layer_type="rel_pos")

        #encoder_layer = nn.TransformerEncoderLayer(d_model=2 * hidden_size * 6 + 768, nhead=4, batch_first=True)
        #self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        a_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size * 6, nhead=2, batch_first=True)
        self.a_transformer_encoder = nn.TransformerEncoder(a_encoder_layer, num_layers=1)

        v_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size * 6, nhead=2, batch_first=True)
        self.v_transformer_encoder = nn.TransformerEncoder(v_encoder_layer, num_layers=1)

        self.pooling = AttentiveStatisticsPooling(2 * hidden_size * 6 + 768)
        #self.batchnorm = nn.BatchNorm1d((2 * hidden_size * 6 + 768)* 2)
        #self.layernorm = torch.nn.LayerNorm((2 * hidden_size * 6 + 768)* 2, eps=1e-12)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear((2 * hidden_size * 6 + 768) * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_dim)

    def forward(self, text_x, audio_x, video_x, text_o, audio_o, video_o):

        _, a = self.aConformer(audio_o)
        _, v = self.vConformer(video_o)
        t = self.bertmodel(text_o)


        #av各自多尺度融合后执行平均池化对齐文本，再av-拼接
        a = torch.cat([a[0], a[1], a[2], a[3], a[4], a[5]], dim=-1)
        a = self.a_transformer_encoder(a)
        v = torch.cat([v[0], v[1], v[2], v[3], v[4], v[5]], dim=-1)
        v = self.v_transformer_encoder(v)
        t, a, v = self.alignNet(t, a, v)
        x = torch.cat([a, v, t], dim=-1)


        """
        #av先执行平均池化对齐文本后av-拼接，再多尺度融合
        for i in range(6):
            t, a[i], v[i] = self.alignNet(t, a[i], v[i])


        out = []

        for i in range(6):
            x = torch.cat([a[i], v[i]], dim=-1)
            out.append(x)
        
        x = torch.cat([out[0], out[1], out[2], out[3], out[4], out[5], t], dim=-1)#64, 50, 128*6+768
        """
        """
        #av先执行平均池化对齐文本后多尺度融合，再av-拼接
        for i in range(6):
            t, a[i], v[i] = self.alignNet(t, a[i], v[i])

        a = torch.cat([a[0], a[1], a[2], a[3], a[4], a[5]], dim=-1)
        v = torch.cat([v[0], v[1], v[2], v[3], v[4], v[5]], dim=-1)

        x = torch.cat([a, v, t], dim=-1)
        """


        x = x.permute(0, 2, 1)
        x = self.pooling(x)
        # x = self.batchnorm(x)
        x = x.permute(0, 2, 1)
        x = x.squeeze(1)

        #x = self.layernorm(x)
        x = self.dropout(x)
        x = F.gelu(self.linear(x))
        x = self.dropout(x)
        output = self.out(x)
        res = {
            'M': output
        }
        return res

