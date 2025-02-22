from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from matplotlib import cm
import numpy as np
from Metrics.LSE_Analysis import lse_calculate


def gradcam_explanations_and_lse_all_classes(model, x_selected, y_selected, threshold = 0.85):
    replace2linear = ReplaceToLinear()
    gradcam = GradcamPlusPlus(model, model_modifier=replace2linear, clone=True)

    gradcam_explanations = []
    lse_gradcam = []
    i = 0
    for image in x_selected:
        x_input = np.expand_dims(image, axis=0)
        lse_per_image = []

        explanation_per_class = []
        # Generar GradCAM++ para cada clase
        for class_idx in range(10):  # Suponiendo que tienes 10 clases
            score = CategoricalScore([class_idx])
            cam = gradcam(score, x_input, penultimate_layer=-4)

            # Normalizar y almacenar
            heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
            explanation_per_class.append(heatmap)

        # Calcular LSE para la clase actual
        lse_per_image.append(lse_calculate(explanation_per_class, y_selected[i], filter=threshold))

        gradcam_explanations.append(explanation_per_class)
        lse_gradcam.append(lse_per_image)
        i+=1

    return gradcam_explanations, lse_gradcam