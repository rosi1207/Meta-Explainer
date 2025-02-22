
import numpy as np
from skimage.color import rgb2gray
from lime import lime_image
from skimage.segmentation import mark_boundaries
from lime.wrappers.scikit_image import SegmentationAlgorithm
from Metrics.LSE_Analysis import lse_calculate

# Función de predicción
def lime_explanations_and_lse_all_classes(model, x_selected_rgb, y_selected, threshold = 0.85):
    
  def predict_fn(images):
    gray_images = np.stack([rgb2gray(img) for img in images], axis=0)
    gray_images = np.expand_dims(gray_images, axis=-1)
    return model.predict(gray_images)

  segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
  explainer = lime_image.LimeImageExplainer()
  lime_explanations = []
  lse_lime = []
  i=0
  for image_rgb in x_selected_rgb:
      print(np.array(image_rgb).shape)
      image_rgb = np.squeeze(image_rgb, axis=2)
      print(np.array(image_rgb).shape)
      explanations_per_class = []
      lse_per_image = []
      # Generar LIME para cada clase
      explanation = explainer.explain_instance(image_rgb, classifier_fn=predict_fn, top_labels=10, hide_color=0, num_samples=10000, segmentation_fn=segmenter)
      for class_idx in range(10): 
          _, mask = explanation.get_image_and_mask(class_idx, positive_only=False, num_features=1000, hide_rest=False, min_weight=0)
          explanations_per_class.append(mask)
      # Calcular LSE para la clase actual
      lse_per_image.append(lse_calculate(explanations_per_class, y_selected[i], filter=threshold))
      lime_explanations.append(explanations_per_class)
      lse_lime.append(lse_per_image)
      i+=1
  return lime_explanations, lse_lime
