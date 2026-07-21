import numpy as np
import torch
def get_batch(dataset:np.ndarray,batch_size:int,context_len:int,device:str):
    starts=np.random.randint(0,len(dataset)-context_len,size=batch_size)
    offsets=np.arange(context_len)
    input_indices=starts[:,None]+offsets
    target_indices=input_indices+1

    input_np=dataset[input_indices]
    target_np=dataset[target_indices]

    inputs=torch.tensor(input_np,dtype=torch.long,device=device)
    targets=torch.tensor(target_np,dtype=torch.long,device=device)
    
    return inputs,targets
