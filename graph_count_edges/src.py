"""Задача подсчета количества ребер графа"""
import os
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm


def create_graph(n, m):
    """
    функция создания полно связного графа

    Args:
        n: количество узлов графа
        m: количество ребер графа

    Returns:
        adj - матрица смежности
    """
    assert m + 1 >= n
    adj = torch.zeros((n, n))

    ij = set()
    for i in range(n-1):
        j = torch.randint(i+1, n, (1, ))
        j = int(j)
        adj[i, j] = 1
        adj[j, i] = 1
        ij.add(tuple(sorted([i, j])))

    while len(ij) < m:
        i, j = torch.randint(0, n, (2,))
        i, j = int(i), int(j)
        i, j = sorted([i, j])
        if i != j and (i, j) not in ij:
            adj[i, j] = 1
            adj[j, i] = 1
            ij.add((i, j))
    return adj


def create_batch(n, m0, m1, batch_size):
    nodes = torch.ones((batch_size, n), dtype=torch.long)
    adj = [create_graph(n, torch.randint(m0, m1, (1, ))[0])[None, ...]
           for _ in range(batch_size)]
    adj = torch.cat(adj)
    return nodes, adj


class GCN(nn.Module):
    def __init__(self, na=2, hidden_dim=8, pad_idx=0):
        super().__init__()
        self.na = na
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(
            num_embeddings=na,
            embedding_dim=hidden_dim,
            padding_idx=pad_idx)
        self.w = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.act = nn.Sigmoid()
        self.count = nn.Linear(hidden_dim, 1)

    def forward(self, nodes, adj):
        h = self.embedding(nodes)
        wh = self.w(h)
        awh = self.act(wh)
        h = h + adj @ awh
        c = self.count(h)
        c = c[..., 0]
        c = torch.sum(c, dim=1)
        return c

    def init_1(self):
        self.embedding.weight.data.fill_(0)
        self.count.weight.data.fill_(1./self.hidden_dim)
        self.count.bias.data.fill_(0)

    def init_2(self):
        self.embedding.weight.data.fill_(1)
        self.w.weight.data.fill_(1000)
        self.count.weight.data.fill_(0.5/self.hidden_dim)
        self.count.bias.data.fill_(-0.5)


def main():
    n0, n1 = 5, 20

    batch_size = 256
    num_step = 1000

    mdl = GCN()
    loss_fnc = nn.MSELoss()
    opt = torch.optim.Adam(mdl.parameters(), lr=1e-2)
    opt.zero_grad()

    history = []
    with tqdm(total=num_step) as pbar:
        for cnt in range(num_step):
            n = torch.randint(n0, n1 + 1, (1, ))[0]
            m0, m1 = int(n - 1), int(n * (n - 1) / 2)
            nodes, adj = create_batch(n, m0, m1, batch_size)
            num_edges = torch.sum(torch.sum(adj, dim=-1), dim=-1) / 2
            c = mdl(nodes, adj)
            loss = loss_fnc(num_edges, c.view(-1))
            loss.backward()

            opt.step()
            opt.zero_grad()

            if cnt % 10 == 0:
                pbar.set_description(f'loss: {loss}')
            history.append(loss.item())
            pbar.update(1)

    with open(os.path.join('.', 'data', 'history.pkl'), 'wb') as fp:
        pickle.dump(history, fp)

    with open(os.path.join('.', 'data', 'mdl.bin'), 'wb') as fp:
        torch.save(mdl.state_dict(), fp)


if __name__ == '__main__':
    main()
