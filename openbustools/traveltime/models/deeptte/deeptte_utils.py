import torch


def get_local_seq(full_seq, kernel_size):
    seq_len = full_seq.size()[0]
    if torch.cuda.is_available():
        indices = torch.cuda.LongTensor(seq_len)
    else:
        indices = torch.LongTensor(seq_len)
    torch.arange(0, seq_len, out=indices)
    indices = torch.autograd.Variable(indices, requires_grad=False)
    first_seq = torch.index_select(full_seq, dim=1, index=indices[kernel_size - 1:])
    second_seq = torch.index_select(full_seq, dim=1, index=indices[:-kernel_size + 1])
    local_seq = first_seq - second_seq
    return local_seq