import torch
import torch.nn as nn
import math

def create_padding_mask(x, pad_token=0):
    mask = (x == pad_token).unsqueeze(1).unsqueeze(2)
    return mask

def create_causal_mask(seq_len):
    mask = torch.triu(torch.ones((1, 1, seq_len, seq_len)), diagonal=1).bool()
    return mask  # Shape: [1, 1, seq_len, seq_len]

def combine_masks(padding_mask, causal_mask):
    return padding_mask | causal_mask  # Combine using logical OR


class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(self.d_model)
    def forward(self, x):
        embedding = self.embedding(x) * self.scale
        return embedding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        position = torch.arange(0, max_len).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # Shape: [d_model/2]

        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pos_encoding[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices

        pos_encoding = pos_encoding.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pos_enc', pos_encoding)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.size()
        # Slice the positional encoding to match the sequence length
        pos_enc = self.pos_enc[:, :seq_len, :].to(x.device)  # Shape: [1, seq_len, d_model]
        # Expand pos_enc to match the batch size
        pos_enc = pos_enc.expand(batch_size, seq_len, self.d_model)  # Shape: [batch_size, seq_len, d_model]
        x = x + pos_enc
        return self.dropout(x)

        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, dropout, heads):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        
        assert d_model % heads == 0
        
        self.head_dim = d_model // heads
        self.queryWeight = nn.Linear(d_model, d_model, bias=False)
        self.keyWeight = nn.Linear(d_model, d_model, bias=False)
        self.valueWeight = nn.Linear(d_model, d_model, bias=False)
        self.outputWeight = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, Q, K, V , mask=None):
        query = self.queryWeight(Q)
        key = self.keyWeight(K)
        value = self.valueWeight(V)
        
        query = query.view(query.shape[0], -1, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(key.shape[0], -1, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(value.shape[0], -1, self.heads, self.head_dim).transpose(1, 2)
        scale = math.sqrt(self.head_dim)
        energy = torch.matmul(query, key.transpose(-2, -1)) / scale

        if mask is not None:
            # mask shape: [batch_size, 1, 1, seq_len]
            # Expand to [batch_size, heads, seq_len_q, seq_len_k]
            mask = mask.expand(-1, self.heads, query.size(-2), key.size(-2))
            energy = energy.masked_fill(mask, float('-inf'))

        attn_weights = nn.functional.softmax(energy, dim=-1)
         
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, value)
        output = output.transpose(1, 2).contiguous().view(output.shape[0], -1, self.d_model)
        output = self.outputWeight(output)
        
        return output

class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, dff):
        super().__init__()
        self.Linear1 = nn.Linear(d_model, dff)
        self.relu = nn.ReLU()
        self.Linear2 = nn.Linear(dff, d_model)
    def forward(self, x):
        return self.Linear2(self.relu(self.Linear1(x)))

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False) ####### .std caused TPU error
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class ResidualBlock(nn.Module):
    def __init__(self, d_model, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, d_model, dff, dropout, heads, eps=float(10**-6)):
        super().__init__()
        self.d_model = d_model
        self.dff = dff
        self.dropout = dropout
        self.heads = heads
        self.eps = eps
        
        self.MultiHeadAttention = MultiHeadAttention(d_model, dropout, heads)
        self.residual_mha = ResidualBlock(d_model, dropout)
        self.residual_ffn = ResidualBlock(d_model, dropout)
        self.PositionWiseFFN = PositionWiseFFN(d_model, dff)
    def forward(self, x, padding_mask):
        x = self.residual_mha(x, lambda z: self.MultiHeadAttention(z, z, z, padding_mask))
        x = self.residual_ffn(x, self.PositionWiseFFN)
        return x
    
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, dff, dropout, heads, eps=float(10**-6)):
        super().__init__()
        self.num_layers = num_layers
        self.stacked_layers = nn.ModuleList([
            EncoderBlock(d_model, dff, dropout, heads, eps) for _ in range(num_layers)
            ])
    def forward(self, x, padding_mask):
        for i in range(self.num_layers):
            x = self.stacked_layers[i](x, padding_mask)
        return x
        
class DecoderBlock(nn.Module):
    def __init__(self, d_model, dff, dropout, heads, eps=float(10**-6)):
        super().__init__()
        self.MultiHeadAttention = MultiHeadAttention(d_model, dropout, heads)
        self.residual_mha1 = ResidualBlock(d_model, dropout)
        self.residual_mha2 = ResidualBlock(d_model, dropout)
        self.residual_ffn = ResidualBlock(d_model, dropout)
        self.PositionWiseFFN = PositionWiseFFN(d_model, dff)
    def forward(self, x, encoder_output, padding_mask, combined_mask):
        x = self.residual_mha1(x, lambda z: self.MultiHeadAttention(z, z, z, combined_mask))
        x = self.residual_mha2(x, lambda z: self.MultiHeadAttention(z, encoder_output, encoder_output, padding_mask))
        x = self.residual_ffn(x, self.PositionWiseFFN)                       
        return x
    
class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, dff, dropout, heads, eps=float(10**-6)):
        super().__init__()
        self.num_layers = num_layers
        self.stacked_layers = nn.ModuleList([
            DecoderBlock(d_model, dff, dropout, heads, eps) for _ in range(num_layers)
            ])
    def forward(self, x, encoder_output, padding_mask, combined_mask):
        padding_mask = padding_mask.expand(-1, -1, x.size(1), -1)
        for i in range(self.num_layers):
            x = self.stacked_layers[i](x, encoder_output, padding_mask, combined_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, dff, dropout, heads, src_vocab_size, tgt_vocab_size, max_len, eps=float(10**-6)):
        super().__init__()
        self.input_embedding = Embedding(d_model, src_vocab_size)
        self.output_embedding = Embedding(d_model, tgt_vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.encoder = Encoder(num_layers, d_model, dff, dropout, heads, eps)
        self.decoder = Decoder(num_layers, d_model, dff, dropout, heads, eps)
        
        self.linear = nn.Linear(d_model, tgt_vocab_size)
    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask, combined_mask):        
        enc_input = self.input_embedding(src)
        enc_input = self.positional_encoding(enc_input)
        enc_output = self.encoder(enc_input, src_padding_mask)
        
        dec_input = self.output_embedding(tgt)
        dec_input = self.positional_encoding(dec_input)
        dec_output = self.decoder(dec_input, enc_output, src_padding_mask, combined_mask)
        linear = self.linear(dec_output)
        return linear