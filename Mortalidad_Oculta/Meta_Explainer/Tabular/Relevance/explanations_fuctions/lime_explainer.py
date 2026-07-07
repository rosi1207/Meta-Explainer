import lime
import lime.lime_tabular
from aux_functions import normalize_explanation, procesar_salida_lime, visualizar_explicacion

def lime_explainer(X, instances_to_explain, model):
    
# Crear el explicador LIME
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values,
        mode='classification',
        feature_names=X.columns,
        discretize_continuous=True
        )
    lime_exp = []
    feature_names = X.columns.tolist()
    for i in instances_to_explain:
        # Generar la explicación LIME
        explanation = explainer.explain_instance(
            data_row=X.iloc[i].values,
            predict_fn=model.predict_proba,
            top_labels=2
        )
        explanation_ = procesar_salida_lime(explanation.as_map())

        lime_exp.append(explanation_)
        print(explanation_)
        # Visualizar la explicación
        visualizar_explicacion(explanation_, feature_names)
        
    lime_exp = normalize_explanation(lime_exp)
        
    return lime_exp