import torch
import torch.nn.functional as F



def compute_middle_attn_vector(delta, delta_bias, A, B, C, L, x_shape, cls_first="middle", full_vector=False):
    dt = F.softplus(delta + delta_bias.unsqueeze(0).unsqueeze(-1))
    dA = torch.exp(torch.einsum("bdl,dn->bldn", dt, A)) ### or dt@A!!
    dB = torch.einsum("bdl,bnl->bldn", dt, B.squeeze(1))
    AttnVecorOverCLS = torch.zeros(x_shape).to(dA.device) #BHL: L vectors per batch and channel
    cls_pos = 98
    for t in range(cls_pos):
        curr_C = C[:,:,:,cls_pos]
        currA = torch.ones(dA.shape[0],dA.shape[2],dA.shape[3]).to(dA.device)
        if t < (cls_pos-1):
            for i in range(cls_pos-1-t):
                currA = currA*dA[:,cls_pos-1-i,:,:]
        currB = dB[:,t,:,:]
        AttnVecorOverCLS[:,:,t] = torch.sum(curr_C*currA*currB, axis=-1)
    return AttnVecorOverCLS


# def compute_attn_matrix_fn(delta, delta_bias, A, B, C, L, x_shape, dtype=torch.float16):
#     dt = F.softplus(delta + delta_bias.unsqueeze(0).unsqueeze(-1))
#     dA = torch.exp(torch.einsum("bdl,dn->bldn", dt, A))
#     dB = torch.einsum("bdl,bnl->bldn", dt, B.squeeze(1))
#     AttnMatrixOverCLS = torch.zeros((x_shape[0], x_shape[1], x_shape[2], x_shape[2]),requires_grad=True).to(dtype).to(dA.device) #BHLL: L vectors per batch and channel
#     for r in range(L):
#         for c in range(r+1):
#             curr_C = C[:,:,:,r]
#             currA = torch.ones((dA.shape[0],dA.shape[2],dA.shape[3]),requires_grad=True, dtype = dtype).to(dA.device)
#             if c < r:
#                 for i in range(r-c):
#                     currA = currA*dA[:,r-i,:,:]
#             currB = dB[:,c,:,:]
#             AttnMatrixOverCLS[:,:,r,c] = torch.sum(curr_C*currA*currB, axis=-1)
#     return AttnMatrixOverCLS

def compute_attn_matrix_fn(delta, delta_bias, A, B, C, L, x_shape, dtype=torch.float16):
    dt = F.softplus(delta + delta_bias.unsqueeze(0).unsqueeze(-1))
    dA = torch.exp(torch.einsum("bdl,dn->bldn", dt, A))
    dB = torch.einsum("bdl,bnl->bldn", dt, B.squeeze(1))
    AttnMatrixOverCLS = torch.zeros((x_shape[0], x_shape[1], x_shape[2], x_shape[2]),requires_grad=True).to(dtype).to(dA.device) #BHLL: L vectors per batch and channel
    for r in range(L):
        for c in range(L):
            curr_C = C[:,:,:,r]
            currA = torch.ones((dA.shape[0],dA.shape[2],dA.shape[3]),requires_grad=True, dtype = dtype).to(dA.device)
            if c < r:
                for i in range(r-c):
                    currA = currA*dA[:,r-i,:,:]
            currB = dB[:,c,:,:]
            AttnMatrixOverCLS[:,:,r,c] = torch.sum(curr_C*currA*currB, axis=-1)
    return AttnMatrixOverCLS