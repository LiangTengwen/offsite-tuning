import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.pytorch_utils import Conv1D


class LoRALayer():
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

#子类
class LoRALinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
    # 
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    #
    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.T @
                           self.lora_B.T) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int, lora_alpha: int, lora_dropout: float, merge_weights: bool):
        # Create a LoRALinear layer from a linear layer
        lora_linear = cls(linear.in_features,
                          linear.out_features,
                          r=r,
                          lora_alpha=lora_alpha,
                          lora_dropout=lora_dropout,
                          merge_weights=merge_weights
                          )
        # Copy the weights
        lora_linear.weight.data = linear.weight.data
        if hasattr(linear, 'bias') and linear.bias is not None:
            lora_linear.bias.data = linear.bias.data
        return lora_linear

    @classmethod
    def from_conv1d(cls, conv1d: Conv1D, r: int, lora_alpha: int, lora_dropout: float, merge_weights: bool):
        """
        1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

        Basically works like a linear layer but the weights are transposed.
        """
        # Create a LoRALinear layer from a conv1d layer
        lora_linear = cls(conv1d.weight.size(0),
                          conv1d.weight.size(1),
                          r=r,
                          lora_alpha=lora_alpha,
                          lora_dropout=lora_dropout,
                          merge_weights=merge_weights
                          )
        # Copy the weights
        lora_linear.weight.data = conv1d.weight.data.T
        if hasattr(conv1d, 'bias') and conv1d.bias is not None:
            lora_linear.bias.data = conv1d.bias.data
        return lora_linear


class Adapter(nn.Module):
    def __init__(self, embed_dim: int, adapter_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, adapter_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(adapter_size, embed_dim)
        self.act_fn = nn.ReLU()

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x + residual
        return x

#在线性层的前向传播后添加一个适配器（adapter）模块。
#`layer`：待添加适配器模块的线性层。
#`embed_dim`：嵌入维度，适配器模块的输入维度。
#`adapter_size`：适配器模块的隐藏层大小（神经元数量）。
#`dropout`：适配器模块的丢弃率。
def add_adapter(layer, embed_dim: int, adapter_size: int, dropout: float):
    # add an adapter module after the forward pass of a linear layer
    # register a forward hook to the layer
    #函数内部定义了一个名为`forward_hook`的函数作为前向传播钩子。在这个钩子函数中，通过调用`adapter`模块对`output`进行处理并返回结果。

    #接下来，函数创建了一个`Adapter`对象，并将其赋值给`layer`对象的`adapter`属性。然后，通过调用`register_forward_hook`方法将前向钩子函数`forward_hook`注册到`layer`上，以实现在前向传播之后自动调用适配器模块。
    #此函数的作用是在线性层的输出后添加一个适配器模块，用于进一步处理输出数据。适配器模块可以在模型中引入额外的非线性变换，以提高模型的性能和表达能力。
    def forward_hook(module, input, output):
        return module.adapter(output)
    layer.adapter = Adapter(embed_dim, adapter_size, dropout)
    layer.register_forward_hook(forward_hook)

# 这段代码定义了一个名为`use_adapter`的函数，用于在给定的神经网络模型中使用适配器（adapter）模块。

# 该函数接受以下参数：

# `layers`：一个`nn.ModuleList`对象，包含要使用适配器模块的网络层。
# `adapter_size`：适配器模块的隐藏层大小（神经元数量）。
# `dropout`：适配器模块的丢弃率，默认为0.1。
# 首先，函数检查`layers[0]`的类型，以确定模型中包含的层类型。如果`layers`中的第一个层是`OPTDecoderLayer`类型，则表示模型是OPT（OpenAI Poly-encoders for Text-to-Text Transfer Transformer）解码器。在这种情况下，函数将适配器模块添加到每个层的`self_attn.out_proj`和`fc2`属性上。

# 如果`layers`中的第一个层是`GPT2Block`类型，则表示模型是GPT-2（Generative Pre-trained Transformer 2）模型。在这种情况下，函数将适配器模块添加到每个层的`attn.c_proj`和`mlp.c_proj`属性上。

# 如果模型中的层类型既不是`OPTDecoderLayer`也不是`GPT2Block`，则会引发`NotImplementedError`异常。
def use_adapter(layers: nn.ModuleList, adapter_size: int, dropout: float = 0.1):
    
    if isinstance(layers[0], OPTDecoderLayer):
        for layer in layers:
            add_adapter(layer.self_attn.out_proj,
                        layer.embed_dim, adapter_size, dropout)
            add_adapter(layer.fc2, layer.embed_dim, adapter_size, dropout)
    elif isinstance(layers[0], GPT2Block):
        for layer in layers:
            add_adapter(layer.attn.c_proj, layer.attn.embed_dim,
                        adapter_size, dropout)
            add_adapter(layer.mlp.c_proj, layer.attn.embed_dim,
                        adapter_size, dropout)
    else:
        raise NotImplementedError

    # freeze all parameters except the adapter modules
    # 接下来，函数冻结除适配器模块之外的所有参数，通过将参数的`requires_grad`属性设置为`False`来实现。然后，它迭代所有参数，并将`name`中包含字符串’adapter’的参数的`requires_grad`属性设置为`True`，以确保适配器模块的参数可以进行梯度更新。

    # 此函数的作用是将适配器模块添加到给定模型的指定层中，并在训练过程中冻结其他参数，只允许适配器模块的参数进行更新。这样可以在不修改原始模型结构的情况下，引入适配器模块以提高模型的性能和灵活性。
    for param in layers.parameters():
        param.requires_grad = False
    for name, param in layers.named_parameters():
        if 'adapter' in name:
            param.requires_grad = True

# 这段代码定义了一个名为`use_lora`的函数，用于在给定的神经网络模型中使用 LoRA（Localized Relevance Attention） 线性层。

# 该函数接受以下参数：

# `layers`：一个 `nn.ModuleList` 对象，包含要替换为 LoRALinear 层的网络层。
# `r`：LoRA 线性层中的相关区域大小。
# `lora_alpha`：LoRA 线性层的 alpha 参数，用于控制相关性的权重。
# `lora_dropout`：LoRA 线性层的丢弃率，默认为 0.1。
# `merge_weights`：一个布尔值，表示是否在 LoRALinear 层中合并权重，默认为 `False`。
# 函数首先检查 `layers[0]` 的类型，以确定模型中包含的层类型。如果 `layers` 中的第一个层是 `OPTDecoderLayer` 类型，则表示模型是 OPT（OpenAI Poly-encoders for Text-to-Text Transfer Transformer）解码器。在这种情况下，函数将通过 `LoRALinear.from_linear` 方法替换每个层的 `self_attn.q_proj`、`self_attn.k_proj`、`self_attn.v_proj`、`self_attn.out_proj`、`fc1` 和 `fc2` 这些线性层。

# 如果 `layers` 中的第一个层是 `GPT2Block` 类型，则表示模型是 GPT-2（Generative Pre-trained Transformer 2）模型。在这种情况下，函数将通过 `LoRALinear.from_conv1d` 方法替换每个层的 `attn.c_attn`、`attn.c_proj`、`mlp.c_fc` 和 `mlp.c_proj` 这些卷积层。

# 如果模型中的层类型既不是 `OPTDecoderLayer` 也不是 `GPT2Block`，则会引发 `NotImplementedError` 异常。

# 接下来，函数冻结除 LoRALinear 层之外的所有参数，通过将参数的 `requires_grad` 属性设置为 `False` 来实现。然后，它迭代所有参数，并将 `name` 中包含字符串 ‘lora’ 的参数的 `requires_grad` 属性设置为 `True`，以确保 LoRALinear 层的参数可以进行梯度更新。

# 此函数的作用是将 LoRALinear 层替换为给定模型的指定层中，并在训练过程中冻结其他参数，只允许 LoRALinear 层的参数进行更新。LoRA 线性层通过将相关性限制在局部区域内，可以提高模型的性能和效果。
def use_lora(layers: nn.ModuleList, r: int, lora_alpha: int, lora_dropout: float = 0.1, merge_weights: bool = False):
    # Replace all linear layers with LoRALinear layers
    if isinstance(layers[0], OPTDecoderLayer):
        for layer in layers:
            layer.self_attn.q_proj = LoRALinear.from_linear(
                layer.self_attn.q_proj, r, lora_alpha, lora_dropout, merge_weights)
            layer.self_attn.k_proj = LoRALinear.from_linear(
                layer.self_attn.k_proj, r, lora_alpha, lora_dropout, merge_weights)
            layer.self_attn.v_proj = LoRALinear.from_linear(
                layer.self_attn.v_proj, r, lora_alpha, lora_dropout, merge_weights)
            layer.self_attn.out_proj = LoRALinear.from_linear(
                layer.self_attn.out_proj, r, lora_alpha, lora_dropout, merge_weights)
            layer.fc1 = LoRALinear.from_linear(
                layer.fc1, r, lora_alpha, lora_dropout, merge_weights)
            layer.fc2 = LoRALinear.from_linear(
                layer.fc2, r, lora_alpha, lora_dropout, merge_weights)
    elif isinstance(layers[0], GPT2Block):
        for layer in layers:
            layer.attn.c_attn = LoRALinear.from_conv1d(
                layer.attn.c_attn, r, lora_alpha, lora_dropout, merge_weights)
            layer.attn.c_proj = LoRALinear.from_conv1d(
                layer.attn.c_proj, r, lora_alpha, lora_dropout, merge_weights)
            layer.mlp.c_fc = LoRALinear.from_conv1d(
                layer.mlp.c_fc, r, lora_alpha, lora_dropout, merge_weights)
            layer.mlp.c_proj = LoRALinear.from_conv1d(
                layer.mlp.c_proj, r, lora_alpha, lora_dropout, merge_weights)
    else:
        raise NotImplementedError

    # freeze all parameters except the LoRALinear layers
    for param in layers.parameters():
        param.requires_grad = False
    for name, param in layers.named_parameters():
        if 'lora' in name:
            param.requires_grad = True

# 这段代码定义了一个名为`use_bitfit`的函数，用于在给定的神经网络模型中使用 BitFit 训练方法。

# 该函数接受一个 `nn.Module` 对象作为参数，表示要应用 BitFit 的模型。

# 函数首先迭代模型的所有参数，并将它们的 `requires_grad` 属性设置为 `False`，从而将所有参数的梯度更新锁定。接着，它再次迭代所有参数，并检查每个参数的维度是否为 1。如果参数的维度为 1，意味着它是偏置参数（bias term），函数会将该参数的 `requires_grad` 属性设置为 `True`，以允许仅在训练过程中更新偏置参数。

# 这样，函数将使模型的所有参数（除了偏置参数）保持不变，只允许偏置参数进行梯度更新。BitFit 是一种用于模型压缩和量化的训练方法，通过冻结大部分模型参数并仅训练偏置参数，可以在减少模型参数量的同时保持较高的模型性能。
def use_bitfit(model: nn.Module):
    # freeze all parameters except the bias terms
    # train the bias terms only
    for param in model.parameters():
        param.requires_grad = False
    for param in model.parameters():
        if param.dim() == 1:
            param.requires_grad = True
