from captum.attr import Saliency
import torch
import numpy as np
from aux_functions import transfer_to_pytorch, visualizar_explicacion

def saliencymaps_explainer(X, instances_to_explain, model):
   
    model_pt = transfer_to_pytorch(model)
    model_pt.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_pt = model_pt.to(device)
    X_test_tensor = torch.tensor(X.values, dtype=torch.float32)
    X_test_tensor = X_test_tensor.to(device)
    feature_names = X.columns.tolist()
    
    saliency = Saliency(model_pt)
    saliency_exp = []
    
    for i in instances_to_explain:
        instance = X_test_tensor[i].unsqueeze(0)
        
        # Calcular para ambas clases
        attr_clase0 = saliency.attribute(instance, target=0, abs=False).cpu().detach().numpy().squeeze()
        attr_clase1 = saliency.attribute(instance, target=1, abs=False).cpu().detach().numpy().squeeze()
        
        # Formato (2, n_features) - igual que LIME e IG
        attributions = np.array([attr_clase0, attr_clase1], dtype=np.float32)
        
        saliency_exp.append(attributions)
        print(attributions)
        visualizar_explicacion(attributions, feature_names)
        
    return saliency_exp