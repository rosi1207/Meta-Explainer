import innvestigate
import numpy as np
import tensorflow as tf
from Metrics.LSE_Analysis import lse_calculate 

def lrp_explanations_and_lse_all_classes(model, x_selected, y_selected, threshold = 0.85):
    model_wo_softmax_ = innvestigate.model_wo_softmax(model)
    analyzer = innvestigate.create_analyzer("lrp.sequential_preset_a_flat", model_wo_softmax_, neuron_selection_mode="index")

    lrp_explanations = []
    lse_lrp = []

    i=0
    for image in x_selected:
        x_input = np.expand_dims(image, axis=0)
        explanations_per_class = []
        lse_per_image = []

        # Generar LRP para cada clase
        for class_idx in range(10):  # Suponiendo que tienes 10 clases
            a = analyzer.analyze(x_input, neuron_selection=class_idx)
            explanation_image = np.squeeze(a)

            explanations_per_class.append(explanation_image)

        # Calcular LSE para la clase actual
        lse_per_image.append(lse_calculate(explanations_per_class, y_selected[i], filter=threshold))

        lrp_explanations.append(explanations_per_class)
        lse_lrp.append(lse_per_image)

        i+=1

    return lrp_explanations, lse_lrp