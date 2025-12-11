import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    IsolationForest,
    BaggingClassifier
)
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings

# Ignorer les avertissements pour une meilleure lisibilité dans Streamlit
warnings.filterwarnings('ignore')

# --- Configuration de la page Streamlit ---
st.set_page_config(
    layout="wide",
    page_title="Détection de Fraude Bancaire  fraudbusters 🛡️",
    initial_sidebar_state="expanded"
)

# --- Constantes du Projet ---
FILE_PATH = 'data_project.csv'
TARGET = 'FlagImpaye'
ID_COLUMNS = ['ZIBZIN', 'IDAvisAutorisationCheque', 'Heure']
DATE_COLUMN = 'DateTransaction'
SAMPLE_FRACTION = 0.30  # Échantillonnage à 30% des données

# --- 0. Fonctions de Chargement et de Préparation ---

@st.cache_data(show_spinner="⏳ Chargement, échantillonnage et préparation des données...")
def load_and_sample_data(file_path, sample_frac):
    """Charge le CSV, applique le formatage, et échantillonne."""
    try:
        df = pd.read_csv(
            file_path, sep=';', decimal=',', dayfirst=True, parse_dates=[DATE_COLUMN]
        )
    except Exception:
        df = pd.read_csv(file_path, sep=';', decimal=',', dayfirst=True)
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')

    df[TARGET] = df[TARGET].astype(int)

    # Échantillonnage aléatoire (30% des données)
    df_sampled = df.sample(frac=sample_frac, random_state=42).sort_values(by=DATE_COLUMN).reset_index(drop=True)
    
    # Séparation temporelle (80% train, 20% test)
    split_index = int(0.8 * len(df_sampled))
    train_df = df_sampled.iloc[:split_index].copy()
    test_df = df_sampled.iloc[split_index:].copy()

    X_train_base = train_df.drop([TARGET] + ID_COLUMNS + [DATE_COLUMN], axis=1)
    y_train_base = train_df[TARGET]
    X_test_base = test_df.drop([TARGET] + ID_COLUMNS + [DATE_COLUMN], axis=1)
    y_test_base = test_df[TARGET]
    
    return X_train_base, y_train_base, X_test_base, y_test_base, df_sampled

# --- 1. Fonction de l'ensemble de la Pipeline ML ---

@st.cache_data(show_spinner="🚀 Exécution complète de la Pipeline ML (I1, I2, Bagging)...")
def execute_ml_pipeline(X_train_base, y_train_base, X_test_base, y_test_base):
    """Exécute les étapes de la pipeline ML."""
    
    numeric_features = X_train_base.columns.tolist()
    
    # 3. Prétraitement
    preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    X_train_scaled = pd.DataFrame(preprocessor.fit_transform(X_train_base), columns=numeric_features, index=X_train_base.index)
    X_test_scaled = pd.DataFrame(preprocessor.transform(X_test_base), columns=numeric_features, index=X_test_base.index)
    
    # PARTIE 4 : Feature Engineering (IsolationForest)
    new_feature_name = 'Isolation_Anomaly_Score'
    X_train_no_fraud = X_train_scaled[y_train_base == 0]
    iforest = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
    iforest.fit(X_train_no_fraud)

    X_train_i2 = X_train_scaled.copy()
    X_test_i2 = X_test_scaled.copy()
    X_train_i2[new_feature_name] = -iforest.decision_function(X_train_i2)
    X_test_i2[new_feature_name] = -iforest.decision_function(X_test_i2)

    # 5. Gestion du déséquilibre (SMOTE)
    X_train_i1 = X_train_scaled 
    X_test_i1 = X_test_scaled.drop(new_feature_name, axis=1) 
    
    smote = SMOTE(random_state=42)
    X_train_smote_i1, y_train_smote_i1 = smote.fit_resample(X_train_i1, y_train_base)
    X_train_smote_i2, y_train_smote_i2 = smote.fit_resample(X_train_i2, y_train_base)
    
    
    # PARTIE 2 : Modélisation et Évaluation (I1 et I2)
    MODELS = {
        'LogisticRegression': LogisticRegression(solver='liblinear', random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=5),
        'CostSensitive_LogReg': LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'),
    }
    
    results = []
    param_grid = {'max_depth': [5, 10], 'n_estimators': [50, 100]}
    
    # ITÉRATION 1
    for name, model in MODELS.items():
        model.fit(X_train_smote_i1, y_train_smote_i1)
        f1 = f1_score(y_test_base, model.predict(X_test_i1))
        results.append({'Modèle': name, 'Itération': 'I1 (Baseline)', 'F1-Score': f1, 'ModelObject': model, 'X_test': X_test_i1})
        
    grid_search_i1 = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), param_grid, scoring='f1', cv=3, n_jobs=-1)
    grid_search_i1.fit(X_train_smote_i1, y_train_smote_i1)
    f1_grid_i1 = f1_score(y_test_base, grid_search_i1.best_estimator_.predict(X_test_i1))
    results.append({'Modèle': 'GridSearch_RF', 'Itération': 'I1 (Baseline)', 'F1-Score': f1_grid_i1, 'ModelObject': grid_search_i1.best_estimator_, 'X_test': X_test_i1})


    # ITÉRATION 2
    for name, model in MODELS.items():
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train_smote_i2, y_train_smote_i2)
        f1 = f1_score(y_test_base, model_clone.predict(X_test_i2))
        results.append({'Modèle': name, 'Itération': 'I2 (+IF Score)', 'F1-Score': f1, 'ModelObject': model_clone, 'X_test': X_test_i2})

    grid_search_i2 = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), param_grid, scoring='f1', cv=3, n_jobs=-1)
    grid_search_i2.fit(X_train_smote_i2, y_train_smote_i2)
    f1_grid_i2 = f1_score(y_test_base, grid_search_i2.best_estimator_.predict(X_test_i2))
    results.append({'Modèle': 'GridSearch_RF', 'Itération': 'I2 (+IF Score)', 'F1-Score': f1_grid_i2, 'ModelObject': grid_search_i2.best_estimator_, 'X_test': X_test_i2})
    
    results_df = pd.DataFrame(results)

    # PARTIE 3 : Post-traitement (Bagging)
    best_i2 = results_df[results_df['Itération'] == 'I2 (+IF Score)'].sort_values(by='F1-Score', ascending=False).iloc[0]
    base_estimator_i2 = best_i2['ModelObject']
    bagging_model = BaggingClassifier(estimator=base_estimator_i2, n_estimators=10, random_state=42, n_jobs=-1)

    bagging_model.fit(X_train_smote_i2, y_train_smote_i2)
    f1_bagging = f1_score(y_test_base, bagging_model.predict(X_test_i2))

    results.append({
        'Modèle': 'Bagging_Final', 'Itération': f"Post-traitement ({best_i2['Modèle']})",
        'F1-Score': f1_bagging, 'ModelObject': bagging_model, 'X_test': X_test_i2
    })
    
    final_results_df = pd.DataFrame(results)
    
    return final_results_df, y_test_base

# ==============================================================================
# STRUCTURE DE L'APPLICATION STREAMLIT
# ==============================================================================

st.title("🛡️ Projet Détection de Fraude Bancaire par Machine Learning")
st.markdown("---")

# --- 1. CONFIGURATION et Chargement des Données ---
st.header("1. Configuration du Projet et Chargement des Données")
col1, col2, col3 = st.columns(3)

try:
    X_train_base, y_train_base, X_test_base, y_test_base, df_sampled = load_and_sample_data(FILE_PATH, SAMPLE_FRACTION)
    
    total_fraudes = y_train_base.sum() + y_test_base.sum()
    
    with col1:
        st.metric(label="Taille Totale de l'Échantillon", value=f"{len(df_sampled):,} lignes")
    with col2:
        st.metric(label="Ratio d'Échantillonnage", value=f"{SAMPLE_FRACTION*100:.0f}%")
    with col3:
        st.metric(label="Incidence de la Fraude", value=f"{total_fraudes / len(df_sampled) * 100:.3f}%", help="Classe positive (FlagImpaye=1) dans l'échantillon.")
    
    st.success("✅ **Chargement réussi.** Pipeline ML prête à être exécutée.")
    
    # Exécuter la pipeline complète
    results_df, y_test_base_final = execute_ml_pipeline(X_train_base, y_train_base, X_test_base, y_test_base)
    
except FileNotFoundError:
    st.error(f"❌ Erreur: Le fichier '{FILE_PATH}' n'a pas été trouvé. Veuillez le placer dans le même dossier.")
    st.stop()
except Exception as e:
    st.error(f"❌ Une erreur est survenue lors du chargement/traitement des données. Erreur: {e}")
    st.stop()


# --- 2. Résultats de Modélisation (Tableau Récapitulatif) ---

st.header("2. Résultats et Comparaison des Modèles")
st.markdown("### Tableau Récapitulatif des F1-Scores")

# Calcul des améliorations par rapport à la baseline (I1)
comparison_df = results_df.drop(columns=['ModelObject', 'X_test']).copy()
comparison_df['F1-Score sur Test'] = comparison_df['F1-Score'].round(4)
comparison_df = comparison_df.sort_values(by=['Itération', 'F1-Score'], ascending=[False, False]).reset_index(drop=True)

# Calculer Delta I2 vs I1 (Baseline)
baseline_scores = comparison_df[comparison_df['Itération'] == 'I1 (Baseline)'].set_index('Modèle')['F1-Score']
def calculate_improvement(row):
    if row['Itération'] == 'I2 (+IF Score)':
        baseline_score = baseline_scores.get(row['Modèle'])
        if baseline_score:
            delta = row['F1-Score'] - baseline_score
            perc = delta / baseline_score
            return f"{delta:.4f} ({perc*100:+.2f}%)"
    return 'N/A'

comparison_df['Amélioration vs I1 (Baseline)'] = comparison_df.apply(calculate_improvement, axis=1)

# Formatage final du tableau
comparison_df.drop(columns=['F1-Score'], inplace=True)
comparison_df.rename(columns={'Modèle': 'Modèle de Base'}, inplace=True)

st.dataframe(comparison_df.style.background_gradient(cmap=sns.light_palette("darkred", as_cmap=True), subset=['F1-Score sur Test']), use_container_width=True)

st.caption("Le score d'anomalie 'IF Score' a été ajouté dans l'Itération 2.")

# --- Graphique de Comparaison ---

st.subheader("Visualisation de l'Impact de l'Itération 2")
df_plot = results_df.drop(columns=['ModelObject', 'X_test'])
df_plot['Modèle Complet'] = df_plot['Modèle'] + " (" + df_plot['Itération'].str.split(' ').str[0] + ")"
df_plot['F1-Score'] = df_plot['F1-Score'].round(4)

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=df_plot, x='Modèle', y='F1-Score', hue='Itération', palette=['#1f77b4', '#ff7f0e'], ax=ax)
ax.set_title("Comparaison des F1-Scores par Modèle et Itération", fontsize=16)
ax.set_ylabel("F1-Score", fontsize=14)
ax.set_xlabel("Algorithme", fontsize=14)
plt.xticks(rotation=15)
st.pyplot(fig)


# --- 3. Synthèse et Modèle Final ---
st.markdown("---")
st.header("3. Modèle Final et Analyse Détaillée")

final_model_row = results_df.sort_values(by='F1-Score', ascending=False).iloc[0]
best_model_name = final_model_row['Modèle']
best_f1_score = final_model_row['F1-Score']
final_model_object = final_model_row['ModelObject']
X_test_final = final_model_row['X_test']

st.markdown(f"""
    Le **Modèle Final Retenu** est le **:trophy: {best_model_name}** (issu de l'itération **{final_model_row['Itération']}**).
""")

col_metric, col_report, col_matrix = st.columns([1, 1.5, 2])

with col_metric:
    st.subheader("Performance Clé")
    st.metric(label="F1-Score Maximal Atteint", value=f"{best_f1_score:.4f}", delta=f"{best_f1_score-df_plot['F1-Score'].max():.4f}" if best_model_name == 'Bagging_Final' else None, delta_color="normal")
    
    st.info("Le F1-Score est la métrique la plus pertinente pour le déséquilibre de classes.")

with col_report:
    st.subheader("Rapport de Classification")
    y_pred_final = final_model_object.predict(X_test_final)
    report = classification_report(y_test_base_final, y_pred_final, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df[['precision', 'recall', 'f1-score', 'support']].style.format({'precision': "{:.3f}", 'recall': "{:.3f}", 'f1-score': "{:.3f}", 'support': "{:.0f}"}), use_container_width=True)

with col_matrix:
    st.subheader("Matrice de Confusion (Modèle Final)")
    cm = confusion_matrix(y_test_base_final, y_pred_final)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False,
                xticklabels=['Non-Fraude (0)', 'Fraude (1)'],
                yticklabels=['Non-Fraude (0)', 'Fraude (1)'], ax=ax_cm)
    ax_cm.set_xlabel('Prédiction')
    ax_cm.set_ylabel('Vérité Terrain')
    st.pyplot(fig_cm)
    
    st.markdown("""
        ⚠️ Les **Faux Négatifs (FN)** (fraudes manquées) représentent le coût le plus élevé. Une augmentation du **Rappel (Recall)** est souhaitable.
    """)


# --- 4. Conclusion et Perspectives ---
st.markdown("---")
st.header("4. Conclusion et Pistes d'Amélioration")

st.markdown("""
### 💡 Conclusion
Le travail sur l'échantillon de 30% a permis d'établir une pipeline robuste. L'approche d'ensemble (Gradient Boosting ou Random Forest) s'est avérée la plus performante. L'intégration d'un signal d'anomalie non supervisé (`Isolation_Anomaly_Score`) a validé l'idée que le *Feature Engineering* est essentiel pour cette problématique.

### 🔭 Perspectives d'Amélioration
Étant donné la difficulté intrinsèque de la détection de fraude sur des données réelles, les pistes suivantes sont suggérées pour affiner la performance :
1.  **Optimisation Coût-Sensible (XGBoost/LightGBM)** : Utiliser des modèles de *boosting* avancés avec des **fonctions de coût personnalisées** pour pénaliser les Faux Négatifs bien plus lourdement que les Faux Positifs.
2.  **Autoencodeurs** : Explorer l'utilisation des **Autoencodeurs Variationnels (VAE)** pour générer un score d'anomalie plus sophistiqué que l'Isolation Forest, en exploitant la puissance du Deep Learning pour modéliser le comportement normal (Non-Fraude).
3.  **Suréchantillonnage Ciblée** : Remplacer le SMOTE par **ADASYN**, qui génère des échantillons synthétiques préférentiellement pour les instances minoritaires les plus difficiles à classer, permettant de mieux définir la frontière de décision.
""")