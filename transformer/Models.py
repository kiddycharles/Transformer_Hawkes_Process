import math

import torch
import torch.nn as nn

import transformer.Constants as Constants
from transformer.Layers import EncoderLayer


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    # seq.ne  Computes input ≠ other input element-wise.
    # A boolean tensor that is True where input is not equal to other and False elsewhere
    # torch.unsequence(dim)
    # Return a new tensor with a dimension of size one inserted at the specified position.
    # The returned tensor shares the same underlying data with this tensor
    # A dim value within the range [-input.dim() - 1, input.dim() + 1) can be used. Negative dim will
    # unsqueeze() applied at dim = dim + input.dim() + 1
    # the index at which to insert the singleton dimension
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


# print(get_non_pad_mask(torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])))
# print(torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 0]]).ne(Constants.PAD))
# print(torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 0]]).dim())
# print(torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 0]]).shape)
# print(torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 0]]).unsqueeze(-1))
# print(torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 0]]).unsqueeze(-1).shape)

def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    print("len_q", len_q)
    padding_mask = seq_k.eq(Constants.PAD)
    print("seq_k", seq_k)
    print("seq_q", seq_q)
    print("padding_mask", padding_mask)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    print("sz_b, len_s", sz_b, len_s)
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    print("subsequent_mask", subsequent_mask)
    print(subsequent_mask.shape)
    print(subsequent_mask.unsqueeze(0))
    print(subsequent_mask.unsqueeze(0).shape)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    print("New subsequent_mask", subsequent_mask)
    return subsequent_mask


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # x = x.permute(0, 2, 1)
        # x: [Batch Time Variate]
        # x: [Batch Variate Time]
        x = x.unsqueeze(-1)
        print(x.shape)
        print(x_mark.shape)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)

class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cpu'))

        # print(self.position_vec)
        # print(self.position_vec.shape)
        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        # Time: N_type * train_len e.g., [2, 1798]
        # time.unsuqeeze(-1) -> [2, 1798, 1]
        result = time.unsqueeze(-1) / self.position_vec
        # print(result)
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        # print(result)
        return result * non_pad_mask

    # def temporal_enc(self, time, non_pad_mask):
    #     data_embed = DataEmbedding_inverted(1, 1798)
    #     result = data_embed(time, non_pad_mask)
    #     return result

    def forward(self, event_type, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """
        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        # print(tem_enc.shape)
        enc_output = self.event_emb(event_type)
        # print(enc_output.shape)
        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        out = out * non_pad_mask
        return out


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1):
        super().__init__()

        self.encoder = Encoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.num_types = num_types

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)

        # prediction of next time stamp
        self.time_predictor = Predictor(d_model, 1)

        # prediction of next event type
        self.type_predictor = Predictor(d_model, num_types)

    def forward(self, event_type, event_time):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """
        print("\nevent_type", event_type.shape)
        print("event_type", event_type)
        print("event_time", event_time.shape)
        print("event_time", event_time)
        non_pad_mask = get_non_pad_mask(event_type)
        print(non_pad_mask.shape)
        print(non_pad_mask)
        # Access the last axis using indexing
        last_axis = non_pad_mask[:, :, -1]

        print("last_axis", last_axis)
        # Verify if all elements are ones
        are_all_ones = torch.all(last_axis == 1)

        print("are_all_ones", are_all_ones)
        # print("x", event_type.shape)  # [2, 1798]
        # print("y", non_pad_mask.shape)  # [2, 1798, 1]
        enc_output = self.encoder(event_type, event_time, non_pad_mask)
        # print("T", enc_output.shape)
        enc_output = self.rnn(enc_output, non_pad_mask)

        time_prediction = self.time_predictor(enc_output, non_pad_mask)

        type_prediction = self.type_predictor(enc_output, non_pad_mask)

        return enc_output, (type_prediction, time_prediction)
