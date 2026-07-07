from aux_functions import transfer_to_pytorch, visualizar_explicacion
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
import numpy as np
from aux_functions import transfer_to_pytorch, crear_arreglo_opuesto_organizado_por_prediccion, normalize_explanation, visualizar_explicacion

def ig_explainer(X, instances_to_explain, model):
    
    feature_names = X.columns.tolist()
    model_pt = transfer_to_pytorch(model)
    #Aplicar Integrated Gradients
    ig = IntegratedGradients(model_pt)
    X_test_tensor = torch.tensor(X.values, dtype=torch.float32)

    # Seleccionar un ejemplo para calcular las contribuciones
    # Por ejemplo, selecciona la primera muestra de X_test_tensor
    ig_exp = []
    for i in instances_to_explain:
        instance = X_test_tensor[i].unsqueeze(0)  # Asegúrate de que la entrada tenga forma (1, input_size)

        # Calcular las contribuciones de las características
        attributions_clase0 = ig.attribute(instance, target=0).numpy().squeeze()
        attributions_clase1 = ig.attribute(instance, target=1).numpy().squeeze()


        # Convertir las atribuciones a un array de NumPy
        attributions_np = np.array([attributions_clase0, attributions_clase1])
        #explanations = crear_arreglo_opuesto_organizado_por_prediccion(model_pt, attributions_np, instance)
        print(attributions_np)
        ig_exp.append(np.array(attributions_np, dtype=np.float32)) 
        visualizar_explicacion(attributions_np, feature_names)
    
    return ig_exp
