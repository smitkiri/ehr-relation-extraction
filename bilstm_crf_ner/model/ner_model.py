#from fastai.text import *
from .core import *

class NERModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_elmo = config.use_elmo

        if not self.use_elmo:
            self.emb = nn.Embedding(self.config.nwords, self.config.dim_word, padding_idx=0)
            self.char_embeddings = nn.Embedding(self.config.nchars, self.config.dim_char, padding_idx=0)
            self.char_lstm = nn.LSTM(self.config.dim_char, self.config.hidden_size_char, bidirectional=True)

        self.dropout = nn.Dropout(p=self.config.dropout)
        self.word_lstm = nn.LSTM(self.config.dim_elmo if self.use_elmo else self.config.dim_word+2*self.config.hidden_size_char,
                                 self.config.hidden_size_lstm, bidirectional=True)

        self.linear = LinearClassifier(self.config, layers=[self.config.hidden_size_lstm*2, self.config.ntags], drops=[0.5])


    def forward(self, input):
        # Word_dim = (batch_size x sent_length)
        # char_dim = (batch_size x sent_length x word_length)

        if self.use_elmo:
            word_emb = self.dropout(input.transpose(0,1))

        else:
            word_input, char_input = input[0], input[1]
            word_input.transpose_(0,1)

            # Word Embedding
            word_emb = self.emb(word_input) #shape= S*B*wnh

            # Char LSTM
            char_emb = self.char_embeddings(char_input.view(-1, char_input.size(2))) #https://stackoverflow.com/questions/47205762/embedding-3d-data-in-pytorch
            char_emb = char_emb.view(*char_input.size(), -1) #dim = BxSxWxE

            _, (h, c) = self.char_lstm(char_emb.view(-1, char_emb.size(2), char_emb.size(3)).transpose(0,1)) #(num_layers * num_directions, batch, hidden_size) = 2*BS*cnh
            char_output = torch.cat((h[0], h[1]), 1) #shape = BS*2cnh
            char_output = char_output.view(char_emb.size(0), char_emb.size(1), -1).transpose(0,1) #shape = S*B*2cnh

            # Concat char output and word output
            word_emb = torch.cat((word_emb, char_output), 2) #shape = S*B*(wnh+2cnh)
            word_emb = self.dropout(word_emb)

        output, (h, c) = self.word_lstm(word_emb) #shape = S*B*hidden_size_lstm
        output = self.dropout(output)

        output = self.linear(output)
        return output #shape = S*B*ntags

class LinearBlock(nn.Module):
    def __init__(self, ni, nf, drop):
        super().__init__()
        self.lin = nn.Linear(ni, nf)
        self.drop = nn.Dropout(drop)
        self.bn = nn.BatchNorm1d(ni)

    def forward(self, x):
        return self.lin(self.drop(self.bn(x)))


class LinearClassifier(nn.Module):
    def __init__(self, config, layers, drops):
        self.config = config
        super().__init__()
        self.layers = nn.ModuleList([
            LinearBlock(layers[i], layers[i + 1], drops[i]) for i in range(len(layers) - 1)])

    def forward(self, input):
        output = input
        sl,bs,_ = output.size()
        x = output.view(-1, 2*self.config.hidden_size_lstm)

        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return l_x.view(sl, bs, self.config.ntags)
