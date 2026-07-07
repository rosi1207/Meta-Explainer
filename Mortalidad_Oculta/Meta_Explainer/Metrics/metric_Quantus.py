from quantus import FaithfulnessEstimate, PixelFlipping,  Complexity, Sparseness
import numpy as np

#fidelidad
def faithfulnessEstimate(a_batch, y_batch, x_batch, model, **kwargs):
    
    a_batch = np.array(a_batch)
    y_batch = np.array(y_batch)
    x_batch = np.array(x_batch)
    
    # Filtrar atribuciones para la clase correcta
    a_batch_filtered = []
    for i in range(len(y_batch)):
        clase = y_batch[i]
        a_batch_filtered.append(a_batch[i][clase])
    a_batch = np.array(a_batch_filtered)
    
    faithfulness_estimate = FaithfulnessEstimate(
        features_in_step=1,           # número de características a eliminar por paso
        abs=False,                     # usar valores absolutos de atribuciones
        normalise=True,               # normalizar atribuciones antes de usar
        perturb_baseline="mean",      # valor de reemplazo: "mean", "black", "white", "random", "uniform"
        return_aggregate=False,       # devolver lista por instancia (False) o promedio (True)
        disable_warnings=True
    )
    
    scores = faithfulness_estimate(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        softmax=False 
    )
    
    return scores

def pixelFlipping(a_batch, y_batch, x_batch, model , **kwargs):

    a_batch = np.array(a_batch)
    y_batch = np.array(y_batch)
    x_batch = np.array(x_batch)
    
    a_batch_filtered = []
    for i in range(len(y_batch)):
        clase = y_batch[i]
        a_batch_filtered.append(a_batch[i][clase])
    
    a_batch = np.array(a_batch_filtered) 

    pixel_flipping = PixelFlipping(
    features_in_step=1,          # número de características a eliminar por paso (
    abs=False,                    # usar valores absolutos de atribuciones
    normalise=True,              # normalizar atribuciones antes de usar
    perturb_baseline="mean",     # valor de reemplazo: "mean", "black", "white", "random", "uniform"
    return_aggregate=False,      # devolver lista por instancia (False) o promedio (True)
    return_auc_per_sample=True,  # devolver AUC en lugar de la curva completa
    disable_warnings=True
    )

    scores = pixel_flipping(
    model=model,
    x_batch=x_batch,
    y_batch=y_batch,
    a_batch=a_batch
    )
    return scores

# complejidad
def complexity(a_batch, y_batch, x_batch, model, **kwargs):
    a_batch = np.array(a_batch)
    y_batch = np.array(y_batch)
    x_batch = np.array(x_batch)
    
    # Filtrar atribuciones para la clase correcta (3D -> 2D)
    if len(a_batch.shape) == 3:
        a_batch_filtered = []
        for i in range(len(y_batch)):
            clase = y_batch[i]
            a_batch_filtered.append(a_batch[i][clase])
        a_batch = np.array(a_batch_filtered)
    
    complexity_metric = Complexity(
        abs=True,
        normalise=True,
        disable_warnings=True
    )
    
    scores = complexity_metric(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        softmax=False
    )
    return scores

def sparseness(a_batch, y_batch, x_batch, model, **kwargs):
    
    a_batch = np.array(a_batch)
    y_batch = np.array(y_batch)
    x_batch = np.array(x_batch)
    
    # Filtrar atribuciones para la clase correcta (3D -> 2D)
    if len(a_batch.shape) == 3:
        a_batch_filtered = []
        for i in range(len(y_batch)):
            clase = y_batch[i]
            a_batch_filtered.append(a_batch[i][clase])
        a_batch = np.array(a_batch_filtered)
    
    sparseness_metric = Sparseness(
        abs=True,           # usar valores absolutos
        normalise=True,     # normalizar atribuciones
        disable_warnings=True
    )
    
    scores = sparseness_metric(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        softmax=False
    )
    return scores