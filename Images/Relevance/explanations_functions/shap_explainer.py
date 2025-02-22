import shap
import numpy as np
from Metrics.LSE_Analysis import lse_calculate

def shap_explanations_and_lse_all_classes(model, x_selected, x_train, y_selected, threshold = 0.85):
    background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
    e = shap.DeepExplainer(model, background)

    shap_explanations = []
    lse_shap = []

    i=0
    for image in x_selected:
        x_input = np.expand_dims(image, axis=0)
        explanations_per_class = []
        lse_per_image = []

        # Generar SHAP para cada clase
        shap_values = e.shap_values(x_input)
        reshaped_shap_values = np.transpose(shap_values[0], (3, 0, 1, 2))  # Ajustar las dimensiones
        print(np.array(reshaped_shap_values).shape)
        # print(np.array(shap_values).shape)
        for class_idx in range(10):  # Suponiendo que tienes 10 clases
            explanations_per_class.append(reshaped_shap_values[class_idx])
            
        # Calcular LSE para la clase actual
        lse_per_image.append(lse_calculate(explanations_per_class, y_selected[i], filter=threshold))

        shap_explanations.append(explanations_per_class)
        lse_shap.append(lse_per_image)

        i+=1

    return shap_explanations, lse_shap