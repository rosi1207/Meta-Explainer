import numpy as np
import torch
def modify_instance(instance, relevances, num_features_to_modify, feature_stds=None, seed = 42):
    """
    Perturba las características más importantes de una instancia basándose en los valores SHAP,
    añadiendo ruido aleatorio, con manejo especial para características enteras.

    Args:
        instance (array-like): Instancia original que se va a modificar.
        relevances (array-like): Valores de relevancia para esa instancia.
        num_features_to_modify (int): Número de características a modificar.
        feature_stds (array-like): Desviación estándar de las características (precalculada).
        seed(int): Semilla usada para la generación aleatoria de los números
    Returns:
        array-like: Instancia perturbada.
    """

    np.random.seed(seed)

    # Convertir a NumPy para evitar problemas con pandas
    instance_perturbed = np.array(instance.copy())

    shap_values_abs = np.abs(relevances)
    max_abs = np.max(shap_values_abs)
    if max_abs > 0:
        rel = shap_values_abs / max_abs 
    else:
        rel = shap_values_abs
    sorted_indices = np.argsort(rel)

    # Índices de las características más importantes según la lista de relevancias
    important_features = sorted_indices[-num_features_to_modify:]

    #print(f"Important features:{important_features}")

    for i in important_features:
        std = feature_stds.iloc[i] if feature_stds is not None else 1

        #print(f"Index {i}, std value: {std}")
        # Generar el valor aleatorio para la perturbación
        perturbation = int(np.round(np.random.normal(0, std*0.1)))

        instance_perturbed[i] += perturbation

    return instance_perturbed



def fidelity(relevances, y_selected, x_selected,model, num_features_to_modify = 3, num_repetitions = 1000, **kwargs ):
    """
    Calcula la fidelidad del modelo al realizar perturbaciones en una instancia basada en los valores SHAP.

    Parameters:
    - model: El modelo de predicción (por ejemplo, un clasificador entrenado)
    - instance: La instancia original sobre la que realizar las perturbaciones (array de NumPy)
    - relevances: Los valores de relevancia para la instancia original
    - class_index: Índice de la neurona que tiene la clasificación más probable.
    - num_features_to_modify: El número de características más importantes a modificar
    - num_repetitions: El número de perturbaciones a realizar
    - feature_stds (array-like): Desviación estándar de las características (precalculada).

    Returns:
    - fidelidad: Valor de Fidelidad después de las perturbaciones
    """
    feature_stds = kwargs['feature_stds']
    fidelity = []
    fidelity_scores = []
    
    for e in range(len(x_selected)):
        # Predicción original del modelo
        # Para PyTorch
        with torch.no_grad():
          input_tensor = torch.tensor(x_selected[e].reshape(1, -1), dtype=torch.float32)
          output = model(input_tensor)
          original_pred = torch.argmax(output).item()
        
        rel_instance = relevances[e][y_selected[e]]
        
        for i in range(num_repetitions):
            # Crear la versión perturbada de la instancia
            instance_perturbed = modify_instance(x_selected[e], rel_instance,
                                                 num_features_to_modify, feature_stds, seed=i)
            # Predicción del modelo tras la perturbación
            with torch.no_grad():
              input_tensor = torch.tensor(instance_perturbed.reshape(1, -1), dtype=torch.float32)
              output = model(input_tensor)
              perturbed_pred = torch.argmax(output).item()
              

            # Calcular fidelidad para esta repetición
            fidelity_scores.append(int(original_pred == perturbed_pred))

        # Fidelidad promedio tras varias perturbaciones por instancia
        fidelity.append(np.mean(fidelity_scores))
    #print(fidelity)

    return fidelity
