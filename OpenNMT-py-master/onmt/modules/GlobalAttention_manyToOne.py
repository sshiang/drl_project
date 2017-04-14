"""
Global attention takes a matrix and a query vector. It
then computes a parameterized convex combination of the matrix
based on the input query.


        H_1 H_2 H_3 ... H_n
          q   q   q       q
            |  |   |       |
              \ |   |      /
                      .....
                  \   |  /
                          a

Constructs a unit mapping.
    $$(H_1 + H_n, q) => (a)$$
    Where H is of `batch x n x dim` and q is of `batch x dim`.

    The full def is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.:

"""

import torch
import torch.nn as nn
import math

class GlobalAttention_ManyToOne(nn.Module):
    def __init__(self, dim):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim*3, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None
        self.mask_state = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        targetT = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, targetT).squeeze(2)  # batch x sourceL
        attn_state = torch.bmm(context_state, targetT).squeeze(2)
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        if self.mask_state is not None:
            attn_state.data.masked_fill_(self.mask_state, -float('inf'))

        attn = self.sm(attn)
        attn_state = self.sm(attn_state)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL
        attn3_state = attn_state.view(attn_state.size(0), 1, attn_state.size(1))  # batch x 1 x sourceL

        weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        weightedContext_state = torch.bmm(attn3_state, context_state).squeeze(1)  # batch x dim

        contextCombined = torch.cat((weightedContext, weightedContext_state, input), 1)

        contextOutput = self.tanh(self.linear_out(contextCombined))

        return contextOutput, attn, attn_state
