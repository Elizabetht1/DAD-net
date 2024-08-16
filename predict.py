import torch
from tools.utils import get_dataset,Params
from runner import load_model
from torch import nn 
import numpy as np

class pretrainedTSC(nn.Module):
    def __init__(self,mod_name,dataset_name):

        super().__init__()

        self.dataset_name = dataset_name
        self.model_name = mod_name
        self.device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu")

        self.model = self.get_model()
        state_dict =torch.load(f"./tools/{mod_name}/model_weights_{dataset_name}_classification.pth")()
        self.model.load_state_dict(state_dict)
    
    
    def get_model(self):
        _,_,ds = get_dataset(self.dataset_name,0,0)
        dataset_params = ds.params
        model_params = Params(f"./tools/{self.model_name}/params.json").params
        return load_model(self.model_name,"classification",self.device,self.dataset_name,model_params,dataset_params,debug=False)

    def __call__(self,x):
        '''
        input: x – an individual sample
        '''
        self.model.eval()
        raw_preds = self.model(torch.unsqueeze(torch.from_numpy(x),0))
        one_hot_preds = np.zeros(raw_preds.shape)
        one_hot_preds[:,torch.argmax(raw_preds)] =1
        return one_hot_preds
