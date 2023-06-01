import torch
from torch.nn import (
    Module,
    Embedding,
    Sequential,
    Conv1d,
    ReLU,
    AdaptiveAvgPool1d,
    Linear,
)


class ThreeInputsNet(Module):
    def __init__(
        self, n_tokens, n_cat_features, concat_number_of_features, hid_size=64
    ):
        super(ThreeInputsNet, self).__init__()
        self.title_emb = Embedding(n_tokens, embedding_dim=hid_size)
        self.title_nn = Sequential(
            Conv1d(in_channels=hid_size, out_channels=3 * hid_size, kernel_size=3),
            ReLU(),
            AdaptiveAvgPool1d(1),
        )

        self.full_emb = Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        self.desc_nn = Sequential(
            Conv1d(in_channels=hid_size, out_channels=3 * hid_size, kernel_size=3),
            ReLU(),
            AdaptiveAvgPool1d(1),
        )

        self.inter_dense = Linear(
            in_features=concat_number_of_features, out_features=hid_size * 3
        )
        self.final_dense = Linear(in_features=hid_size * 3, out_features=1)

        self.category_out = Sequential(
            Linear(in_features=n_cat_features, out_features=hid_size * 3),
            ReLU(),
        )

    def forward(self, whole_input):
        input1, input2, input3 = whole_input
        title_beg = self.title_emb(input1).permute((0, 2, 1))
        title = self.title_nn(title_beg)

        full_beg = self.full_emb(input2).permute((0, 2, 1))
        full = self.desc_nn(full_beg)

        category = self.category_out(input3)

        concatenated = torch.cat(
            [
                title.view(title.size(0), -1),
                full.view(full.size(0), -1),
                category.view(category.size(0), -1),
            ],
            dim=1,
        )

        out = self.final_dense(self.inter_dense(concatenated))

        return out
