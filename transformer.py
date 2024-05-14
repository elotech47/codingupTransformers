import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32)) # scale by sqrt(d_model) as per the paper (from the original transformer paper)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # positional encoding formula - PE(pos, 2i) = sin(pos/10000^(2i/d_model)) and PE(pos, 2i+1) = cos(pos/10000^(2i/d_model)) where pos is the position and i is the dimension of the embedding
        # create a vector of shape (seq_len) with values from 0 to seq_len-1
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # shape (seq_len, 1)    
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)) # shape (d_model/2)
        # applyy the sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # shape (1, seq_len, d_model)
        # register the positional encoding as a buffer so that it is saved in the state_dict
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False) # add the positional encoding to the input
        return self.dropout(x)
    
    
class LayerNormalization(nn.Module):
    def __init__(self, esp: float = 1e-6):
        super(LayerNormalization, self).__init__()
        self.esp = esp
        self.alpha = nn.Parameter(torch.tensor(1.)) # Multiplicative parameter
        self.bias = nn.Parameter(torch.tensor(1.)) # Additive parameter
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.esp) + self.bias
    
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super(FeedForwardBlock, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff) # W1 and b1
        self.linear2 = nn.Linear(d_ff, d_model) # W2 and b2
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model should be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.d_k = d_model // num_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_o = nn.Linear(d_model, d_model)
        
        
    @staticmethod
    def attention(q, k, v, d_k, dropout:nn.Dropout, mask=None):
        d_k = q.size(-1)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = dropout(attention_scores)
        output = torch.matmul(attention_scores, v)
        return output, attention_scores
        
    def forward(self, q, k, v, mask=None):
        # q, k, v have shape (batch_size, seq_len, d_model)
        batch_size = q.size(0)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) # shape (batch_size, num_heads, seq_len, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # attention calculation
        output, _ = MultiHeadAttention.attention(q, k, v, self.d_k, self.dropout, mask) # shape (batch_size, num_heads, seq_len, d_k)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) # shape (batch_size, seq_len, d_model)
        output = self.linear_o(output)
        return output
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttention, feed_forward: FeedForwardBlock, dropout: float):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connection1 = ResidualConnection(dropout)
        self.residual_connection2 = ResidualConnection(dropout)
        
    def forward(self, x, mask):
        x = self.residual_connection1(x, lambda x: self.self_attention(x, x, x, mask))
        x = self.residual_connection2(x, self.feed_forward)
        return x
    
class Encoder(nn.Module):
    def __init__(self, encoder_layers: nn.ModuleList):
        super(Encoder, self).__init__()
        self.encoder_layers = encoder_layers
        self.norm_layer = LayerNormalization()
        
    def forward(self, x, mask):
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return self.norm_layer(x)
    
class DecoderBlock(nn.Module):
    
    def __init__(self, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, feed_forward: FeedForwardBlock, dropout: float):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connection1 = ResidualConnection(dropout)
        self.residual_connection2 = ResidualConnection(dropout)
        self.residual_connection3 = ResidualConnection(dropout)
        
    def forward(self, x, encoder_output, src_mask, tgt_mask): 
        x = self.residual_connection1(x, lambda x: self.self_attention(x, x, x, tgt_mask)) # decoder self attention
        x = self.residual_connection2(x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask)) # encoder-decoder attention
        x = self.residual_connection3(x, self.feed_forward) # feed forward
        return x
    
class Decoder(nn.Module):
    def __init__(self, decoder_layers: nn.ModuleList):
        super(Decoder, self).__init__()
        self.decoder_layers = decoder_layers
        self.norm_layer = LayerNormalization()
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm_layer(x)
    
class Generator(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return torch.log_softmax(self.linear(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_positional_encoding: PositionalEncoding, tgt_positional_encoding: PositionalEncoding, generator: Generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_positional_encoding = src_positional_encoding
        self.tgt_positional_encoding = tgt_positional_encoding
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        src = self.src_positional_encoding(self.src_embed(src))
        tgt = self.tgt_positional_encoding(self.tgt_embed(tgt))
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return self.generator(decoder_output)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_positional_encoding(self.src_embed(src)), src_mask)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        return self.decoder(self.tgt_positional_encoding(self.tgt_embed(tgt)), encoder_output, src_mask, tgt_mask)
    
    
def transformer_builder(src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512, src_seq_len: int = 100, tgt_seq_len: int = 100, num_layers: int = 6, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1) -> Transformer:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    src_positional_encoding = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_positional_encoding = PositionalEncoding(d_model, tgt_seq_len, dropout)
   
    # create the encoder layers
    encoder_blocks = nn.ModuleList([EncoderBlock(MultiHeadAttention(d_model, num_heads, dropout), FeedForwardBlock(d_model, d_ff, dropout), dropout) for _ in range(num_layers)])
    
    # create the decoder layers
    decoder_blocks = nn.ModuleList([DecoderBlock(MultiHeadAttention(d_model, num_heads, dropout), MultiHeadAttention(d_model, num_heads, dropout), FeedForwardBlock(d_model, d_ff, dropout), dropout) for _ in range(num_layers)])
    
    encoder = Encoder(encoder_blocks)
    decoder = Decoder(decoder_blocks)
    
    generator = Generator(d_model, tgt_vocab_size)
    
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_positional_encoding, tgt_positional_encoding, generator)
    
    # initialize the weights
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer
            
            
if __name__ == '__main__':
    src_vocab_size = 100000
    tgt_vocab_size = 100000
    d_model = 512
    src_seq_len = 100
    tgt_seq_len = 100
    num_layers = 6
    num_heads = 8
    d_ff = 2048
    dropout = 0.1
    transformer = transformer_builder(src_vocab_size, tgt_vocab_size, d_model, src_seq_len, tgt_seq_len, num_layers, num_heads, d_ff, dropout)