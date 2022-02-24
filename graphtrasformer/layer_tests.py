from tqdm import tqdm

from graphtrasformer.layers import *
if __name__=='__main__':




    layer = Transformer_Layer(
                             num_heads=4,
                             hidden_dim=64,
                             ffn_hidden_dim=128,
                             dropout=0.1,
                             attn_dropout=0.1,
                             temperature=1,
                             activation_fn='GELU')




    x = torch.randn(8, 100, 64)
    x[:,80:,:]=0
    mask = torch.zeros(8, 100,100)
    mask[:,:80,:80]=1


    #pred = layer(x,x_mask)

    out,attn = layer.attention(x,mask)


