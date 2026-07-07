import shap
from aux_functions import normalize_explanation, separar_arreglo_entrada, visualizar_explicacion

def shap_explainer(X, instances_to_explain, model):
    
    explainer = shap.KernelExplainer(model.predict_proba, X)
    feature_names = X.columns.tolist()

    shap_exp = []
    for i in instances_to_explain:
        shap_values = explainer.shap_values(X.iloc[i])
        resultado = separar_arreglo_entrada(shap_values)
        shap_exp.append(resultado)
        print(shap_exp)
        visualizar_explicacion(resultado, feature_names,)
   
    shap_exp = normalize_explanation(shap_exp)
    
    return shap_exp 