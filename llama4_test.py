import torch
from models.llama4.args import ModelArgs, MoEArgs
from models.llama4.model import Transformer


simple_llama2_config = ModelArgs(
    dim=256,
    n_layers=2,
    n_heads=32,
    n_kv_heads=8,
    vocab_size=3200,
    ffn_exp=4,
    ffn_dim_multiplier=4,
    moe_args=MoEArgs(num_experts=10, top_k=2),
)
net = Transformer.from_model_args(simple_llama2_config).cuda()


x = torch.randint(0, 3200, (20, 128)).cuda()
y = net(x)
print(y.shape)
y.sum().backward()
