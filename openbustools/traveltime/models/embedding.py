from torch import nn

EMBED_DICT = {
    't_min_of_day': {
        'vocab_dim': 1440,
        'embed_dim': 48
    },
    't_day_of_week': {
        'vocab_dim': 7,
        'embed_dim': 4
    }
}


class MinuteEmbedding(nn.Module):
    def __init__(self):
        super(MinuteEmbedding, self).__init__()
        self.vocab_dim = EMBED_DICT['t_min_of_day']['vocab_dim']
        self.embed_dim = EMBED_DICT['t_min_of_day']['embed_dim']
        self.em = nn.Embedding(self.vocab_dim, self.embed_dim)
    def forward(self, x):
        return self.em(x)


class DayEmbedding(nn.Module):
    def __init__(self):
        super(DayEmbedding, self).__init__()
        self.vocab_dim = EMBED_DICT['t_day_of_week']['vocab_dim']
        self.embed_dim = EMBED_DICT['t_day_of_week']['embed_dim']
        self.em = nn.Embedding(self.vocab_dim, self.embed_dim)
    def forward(self, x):
        return self.em(x)