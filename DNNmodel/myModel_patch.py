import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_patch(input, patch_len):
    # input:[batch,len,input_size]
    input = input.to(device)
    batch_size = input.shape[0]
    length = input.shape[1]
    # input:[batch,hidden,len]
    input = input.permute(0,2,1)
    if length % patch_len == 0:
        patch_num = int(length / patch_len)
    else:
        stride = patch_len - length % patch_len
        padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
        input = padding_patch_layer(input)
        patch_num = int(length / patch_len) + 1

    input = input.unfold(dimension=-1, size=patch_len,step=patch_len) 
    input = input.transpose(1, 2)
    input = input.reshape(batch_size,patch_num,-1)
    # input:[batch,patch_num,patch_len*input_size]
    return input,length

def random_masking(patch_input, mask_ratio):
    # xb: [batch,patch_num,hidden]
    batch,patch_num,hidden = patch_input.shape
    x = patch_input.clone()
    
    len_keep = int(patch_num * (1 - mask_ratio))
        
    noise = torch.rand(batch, patch_num, device=patch_input.device)  # noise in [0, 1], bs x L
        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)                                     # ids_restore: [bs x L]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]                                                 # ids_keep: [bs x len_keep]         
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, hidden))        # x_kept: [bs x len_keep x dim]
    # removed x
    x_removed = torch.zeros(batch, patch_num-len_keep, hidden, device=patch_input.device)                        # x_removed: [bs x (L-len_keep) x dim]
    x_ = torch.cat([x_kept, x_removed], dim=1)                                          # x_: [bs x L x dim]
    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,hidden))    # x_masked: [bs x num_patch x dim]
    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([batch, patch_num], device=x.device)                                          # mask: [bs x num_patch]
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)                                  # [bs x num_patch]

    return x_masked, x_kept, mask, ids_restore  
                                                                               



def patch_masking(input, patch_len,mask_ratio):
    # patch_input:[batch,patch_num,patch_len*input_size]
    patch_input,length = create_patch(input,patch_len)
    input_mask, _, mask, _ = random_masking(patch_input, mask_ratio)
    return patch_input,input_mask,mask,length


if __name__ == '__main__':

    input = torch.rand(1,10,2)
    print("input")
    print(input)
    print(' ')
    random_masking(input, 0.4)

