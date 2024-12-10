import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, dims, dropout=0.5, is_training=True):

        super(Encoder, self).__init__()

        self.dims = dims

        temp_dims = dims[:-1] + [dims[-1] * 2]

        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(temp_dims[:-1], temp_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(temp_dims[:-1]) - 1:
                mlp_modules.append(nn.Tanh())
        self.layers = nn.Sequential(*mlp_modules)

        self.dropout = nn.Dropout(dropout)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):

        h = F.normalize(x)
        h = self.dropout(h)
        h = self.layers(h)

        mu = h[:, :self.dims[-1]]
        logvar = h[:, self.dims[-1]:]

        z = self.reparameterize(mu, logvar)

        return z, mu, logvar


class Decoder(nn.Module):

    def __init__(self, dims):

        super(Decoder, self).__init__()

        self.dims = dims

        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(dims[:-1]) - 1:
                mlp_modules.append(nn.Tanh())
        self.layers = nn.Sequential(*mlp_modules)

    def forward(self, z):

        h = self.layers(z)

        # for i, layer in enumerate(self.layers):
        #     h = layer(h)
        #     if i != len(self.layers) - 1:
        #         # h = nn.Tanh(h)
        #         # h = F.tanh(h)
        #         h = torch.tanh(h)

        return h


class GatingNetwork(torch.nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0., latent_dim=128):
        super(GatingNetwork, self).__init__()

        self.fc1 = nn.Linear(in_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        weights = F.softmax(x, dim=1)
        return weights


# class GatingNetwork(torch.nn.Module):
#
#     def __init__(self, in_dim, out_dim, dropout=0., latent_dim=128):
#         super(GatingNetwork, self).__init__()
#
#         self.query = nn.Linear(in_dim, latent_dim)
#         self.key = nn.Linear(in_dim, latent_dim)
#         self.value = nn.Linear(in_dim, latent_dim)
#
#         self.fc = nn.Linear(latent_dim, out_dim)
#
#     def forward(self, x):
#         Q = self.query(x)
#         K = self.key(x)
#         V = self.value(x)
#         attention = torch.softmax(Q @ K.transpose(-2, -1) / (128 ** 0.5), dim=-1)
#         gates = self.fc(attention @ V)
#         weights = F.softmax(gates, dim=1)
#         return weights


class MAE(nn.Module):

    def __init__(self, args):
        super(MAE, self).__init__()

        self.config = args
        self.num_items = args.num_items

        # Automatic calculation of input and output dimensions based on the number of layers
        self.p_dims = sorted([args.latent_dim * (l + 1) for l in range(self.config.num_layers)])
        self.p_dims.append(self.num_items)
        # In and Out dimensions must equal to each other
        if args.q_dims:
            assert args.q_dims[0] == self.p_dims[-1], "In and Out dimensions must equal to each other"
            assert args.q_dims[-1] == self.p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = args.q_dims
        else:
            self.q_dims = self.p_dims[::-1]

        self.gate = GatingNetwork(args.num_items, 2, args.dropout, latent_dim=args.latent_dim)
        self.ql = Encoder(self.q_dims)
        self.qg = Encoder(self.q_dims)
        self.decoder = Decoder(self.p_dims)
        self.logistic = nn.Sigmoid()

    def forward(self, x):

        ws = self.gate(x)
        w1, w2 = ws[:, 0].unsqueeze(1), ws[:, 1].unsqueeze(1)

        z1, mu1, logvar1 = self.ql(x)
        z2, mu2, logvar2 = self.qg(x)

        z = w1 * z1 + w2 * z2
        mu = w1 * mu1 + w2 * mu2
        var = w1 ** 2 * logvar1.exp() + w2 ** 2 * logvar2.exp()
        logvar = torch.log(var)

        logits = self.decoder(z)

        return logits, mu, logvar, z


class MAE_fix(nn.Module):

    def __init__(self, args):
        super(MAE_fix, self).__init__()

        self.config = args
        self.num_items = args.num_items

        # Automatic calculation of input and output dimensions based on the number of layers
        self.p_dims = sorted([args.latent_dim * (l + 1) for l in range(self.config.num_layers)])
        self.p_dims.append(self.num_items)
        # In and Out dimensions must equal to each other
        if args.q_dims:
            assert args.q_dims[0] == self.p_dims[-1], "In and Out dimensions must equal to each other"
            assert args.q_dims[-1] == self.p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = args.q_dims
        else:
            self.q_dims = self.p_dims[::-1]

        self.ql = Encoder(self.q_dims)
        self.qg = Encoder(self.q_dims)
        self.decoder = Decoder(self.p_dims)
        self.logistic = nn.Sigmoid()

    def forward(self, x):

        w1 = self.config.weight
        w2 = 1 - w1

        z1, mu1, logvar1 = self.ql(x)
        z2, mu2, logvar2 = self.qg(x)

        z = w1 * z1 + w2 * z2
        mu = w1 * mu1 + w2 * mu2
        var = w1 ** 2 * logvar1.exp() + w2 ** 2 * logvar2.exp()
        logvar = torch.log(var)

        logits = self.decoder(z)

        return logits, mu, logvar, z
