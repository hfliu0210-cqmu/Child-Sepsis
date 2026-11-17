import streamlit as st
import pandas as pd
import json
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
import base64


# =====================================================
# Page configuration
# =====================================================

st.set_page_config(
    page_title="CMU-iPedSep",
    page_icon="🩺",
    layout="wide"
)



# =====================================================
# Load model
# =====================================================

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("xgboost_sepsis_model.json")

    with open("feature_names.json", "r", encoding="utf-8") as f:
        features = json.load(f)

    return model, features



model, features = load_model()



# =====================================================
# CSS
# =====================================================

st.markdown(
"""
<style>

.section-title {

    font-family: "italic";

    font-size: 30px;

    font-weight: bold;

    color:#164A7B;

}


/* CMU-iPedSep introduction area */
.model-intro {
    background: #EEF3F8;
    border-left: 5px solid #B22222;
    padding: 12px 16px;
    margin: 18px 0 16px 0;
    border-radius: 4px;
    color: #A61B1B;
    font-size: 21px;
    font-weight: 600;
}

/* Introduction tabs */
/* Tab list layout */
div[data-testid="stTabs"] [role="tablist"],
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    display: flex !important;
    gap: 10px !important;
    border-bottom: 1px solid #D8E2EC !important;
    background: transparent !important;
    align-items: stretch !important;
}

/* Tab title cards: support both Streamlit and BaseWeb DOM variants */
div[data-testid="stTabs"] [role="tab"],
div[data-testid="stTabs"] button[data-baseweb="tab"],
div[data-testid="stTabs"] [data-baseweb="tab"] {
    background: #E7EEF6 !important;
    background-color: #E7EEF6 !important;
    color: #31536F !important;
    border: 1px solid #D7E2EC !important;
    border-bottom: 1px solid #D7E2EC !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 11px 18px !important;
    margin: 0 !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    box-shadow: 0 1px 3px rgba(35, 65, 95, 0.08) !important;
}

/* Force bold text inside tab buttons */
div[data-testid="stTabs"] [role="tab"] *,
div[data-testid="stTabs"] button[data-baseweb="tab"] *,
div[data-testid="stTabs"] [data-baseweb="tab"] * {
    color: inherit !important;
    font-weight: 700 !important;
}

/* Selected tab */
div[data-testid="stTabs"] [role="tab"][aria-selected="true"],
div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"],
div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
    background: #DCE8F4 !important;
    background-color: #DCE8F4 !important;
    color: #164A7B !important;
    border-color: #B9CCDF !important;
    box-shadow: inset 0 -3px 0 #164A7B !important;
}

/* Keep the tab content panel clean and coordinated */
div[data-testid="stTabs"] div[role="tabpanel"] {
    background: #F7F9FC;
    border: 1px solid #E1E8EF;
    border-top: 3px solid #164A7B;
    border-radius: 0 0 6px 6px;
    padding: 18px 20px;
}

.result-number {

    font-family:"Times New Roman";

    font-size:50px;

    font-weight:bold;

    color:#B22222;

    text-align:center;

}


body {

    font-family:"Times New Roman";

}



/* Overall scientific website background */
html, body, [data-testid="stAppViewContainer"] {
    background: #F3F6FA !important;
    color: #243447;
}

/* Main application canvas */
[data-testid="stAppViewContainer"] > .main {
    background: #F3F6FA !important;
}
[data-testid="stMain"] {
    background: #F3F6FA !important;
}
.block-container {
    background: transparent;
    padding-top: 1.6rem;
    padding-bottom: 2.2rem;
    max-width: 1500px;
}

/* Header and toolbar blend into the page */
[data-testid="stHeader"] {
    background: rgba(243, 246, 250, 0.96) !important;
}

/* Input widgets retain a clean white clinical interface */
[data-testid="stNumberInput"],
[data-testid="stSelectbox"] {
    background: #FFFFFF;
    border-radius: 8px;
}

/* Subtle separation for the two main prediction panels */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #E1E8F0;
}

/* Tabs and information panels use the same cool scientific palette */
div[data-testid="stTabs"] div[role="tabpanel"] {
    background: #FFFFFF !important;
    border-color: #DCE5EF !important;
}

/* Dividers are intentionally understated */
hr {
    border-color: #DCE4ED !important;
}


/* Ensure all three Streamlit tab titles are visibly bold */
div[data-testid="stTabs"] button[role="tab"],
div[data-testid="stTabs"] button[role="tab"] *,
div[data-testid="stTabs"] button[role="tab"] p,
div[data-testid="stTabs"] button[role="tab"] span {
    font-weight: 700 !important;
}


        /* Developer profile layout */
        .developer-name {
            font-size: 20px;
            font-weight: 700;
            color: #223A57;
            margin-bottom: 6px;
        }
        .developer-bio {
            font-size: 16px;
            line-height: 1.7;
            color: #31445F;
            margin-bottom: 4px;
        }

</style>
""",
unsafe_allow_html=True
)



# =====================================================
# Full width title banner
# =====================================================


if os.path.exists("title.png"):


    with open(
        "title.png",
        "rb"
    ) as f:

        encoded = base64.b64encode(
            f.read()
        ).decode()



    st.markdown(

    f"""

    <style>

    .title-banner img {{

        width:100%;

        height:auto;

        display:block;

    }}

    </style>


    <div class="title-banner">

    <img src="data:image/png;base64,{encoded}">

    </div>

    """,

    unsafe_allow_html=True

    )


else:


    st.warning(
        "title.png not found"
    )



# =====================================================
# Model purpose and information modules
# =====================================================

st.markdown(
    """
    <div class="model-intro">
        This model aims to predict sepsis risk in children admitted for acute infection.
    </div>
    """,
    unsafe_allow_html=True
)

tab_sepsis, tab_cohort, tab_developers = st.tabs([
    "Definition of sepsis",
    "Introduction to the model development cohort",
    "Introduction of the main developers"
])

with tab_sepsis:
    st.markdown(
        """
        **Sepsis** is a life-threatening condition characterized by
        organ dysfunction resulting from a dysregulated host response to infection (according to
        Surviving Sepsis Campaign International Guidelines for the Management of Sepsis and Septic Shock in Children 2026. Pediatr Crit Care Med.).

        For CMU-iPedSep, sepsis status is determined according to the
        operational definition and outcome-labeling criteria used in the
        original model development cohort.
        """
    )

with tab_cohort:
    st.markdown(
        """
        **Development cohort**

        The model was developed using a retrospective cohort of **25,426
        children admitted with acute infection** to the Children's Hospital of Chongqing Medical University, National Clinical
        Research Center for Children and Adolescents' Health and Diseases
        in China between **2012 and 2025**.

        - **Sepsis:** 2,788 (11.0%) children progressed to sepsis(8.0 [3.0-24.0] months; 1763 [63.2%] male)
        - **Infection only:** 22,638 (89.0%) children remained non-septic (11.0 [5.0-31.0] months; 14,036 [62.0%] male)

        Ten candidate machine-learning algorithms were compared, and
        **XGBoost** was selected for the final model. CMU-iPedSep uses
        routinely available admission indicators to estimate the risk
        of sepsis in children admitted for acute infection.
        """
    )


with tab_developers:
    developers = [
        {
            "name": "Hai-Feng Liu, MD, PhD",
            "photo": "haifeng_liu.jpg",
            "bio": "Postdoctoral Fellow at the National Clinical Research Center for Children and Adolescents' Health and Diseases and Children's Hospital of Chongqing Medical University. His research focuses on severe pediatric infections and artificial intelligence-based clinical models."
        },
        {
            "name": "Yushun Wan, PhD",
            "photo": "yushun_wan.png",
            "bio": "Professor and doctoral supervisor at the College of Basic Medicine, Chongqing Medical University. His research focuses on respiratory viral infection and transmission, and the development of antiviral vaccines and drugs."
        },
        {
            "name": "Enmei Liu, MD, PhD",
            "photo": "enmei_liu.png",
            "bio": "Vice President of Chongqing Medical University and Chief Scientist of the National Key Research and Development Program of China. Her research focuses on respiratory virus molecular epidemiology, early-warning models, and pathogenesis in children."
        },
        {
            "name": "Na Zang, MD, PhD",
            "photo": "na_zang.png",
            "bio": "Chief Physician and Professor at the National Clinical Research Center for Children and Adolescents' Health and Diseases and Children's Hospital of Chongqing Medical University. Her research focuses on severe respiratory viral infections in children, chronic cough, and allergic diseases."
        }
    ]

    for idx, dev in enumerate(developers):
        col_photo, col_text = st.columns([1, 5], gap="medium")
        with col_photo:
            if os.path.exists(dev["photo"]):
                st.image(dev["photo"], width=130)
            else:
                st.info("Photo not found")
        with col_text:
            st.markdown(f'<div class="developer-name">{dev["name"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="developer-bio">{dev["bio"]}</div>', unsafe_allow_html=True)
        if idx < len(developers) - 1:
            st.markdown("<div style=\"height:14px\"></div>", unsafe_allow_html=True)

st.divider()



# =====================================================
# Main layout
# =====================================================


left, right = st.columns(
    [1,1]
)



# =====================================================
# Left panel
# =====================================================


with left:


    st.markdown(

    """

    <div class="section-title">

    Patient Information Input

    </div>

    """,

    unsafe_allow_html=True

    )


    patient = {}



    patient["Age"] = st.number_input(

        "Age (months)",

        min_value=0,

        max_value=216,

        value=None

    )



    patient["Underlying condition"] = st.selectbox(

        "Underlying condition",

        options=[None,0,1],


        format_func=lambda x:

        "Please select"

        if x is None

        else

        (

            "No underlying condition"

            if x==0

            else

            "Yes (Prematurity / Chronic diseases / Severe malnutrition)"

        )

    )



    patient["Lymphocyte"] = st.number_input(

        "Lymphocyte (×10⁹/L)",

        min_value=0.0,

        value=None,

        step=0.1,

        format="%.1f"

    )



    patient["Monocyte"] = st.number_input(

        "Monocyte (×10⁹/L)",

        min_value=0.0,

        value=None,

        step=0.1,

        format="%.1f"

    )



    patient["Hemoglobin"] = st.number_input(

        "Hemoglobin (g/L)",

        min_value=0.0,

        value=None,

        step=0.1,

        format="%.1f"

    )



    patient["MCHC"] = st.number_input(

        "MCHC (g/L)",

        min_value=0.0,

        value=None,

        step=0.1,

        format="%.1f"

    )



    patient["RDWCV"] = st.number_input(

        "RDW-CV (%)",

        min_value=0.0,

        value=None,

        step=0.1,

        format="%.1f"

    )



    patient["NLR"] = st.number_input(

        "Neutrophil-to-Lymphocyte Ratio (NLR)",

        min_value=0.0,

        value=None,

        step=0.01,

        format="%.2f"

    )



    predict = st.button(

        "Predict Sepsis Probability"

    )



# =====================================================
# Right panel
# =====================================================


with right:


    st.markdown(

    """
    <div class="section-title">

    Prediction Output

    </div>

    """,

    unsafe_allow_html=True

    )



    if predict:


        missing=[]


        for k,v in patient.items():

            if v is None:

                missing.append(k)



        if len(missing)>0:


            st.error(

                "Prediction failed. Missing variables: "

                +

                ", ".join(missing)

            )



        else:


            X = pd.DataFrame(
                [patient]
            )


            X = X[
                features
            ]



            probability = model.predict_proba(
                X
            )[0][1]



            st.markdown(

            f"""

            <div class="result-number">

            {probability*100:.2f}%

            </div>

            """,

            unsafe_allow_html=True

            )


            st.write(

                "Predicted probability of sepsis"

            )



            st.divider()



            st.subheader(

                "Individual Feature Contribution Analysis"

            )



            explainer = shap.TreeExplainer(
                model
            )


            shap_values = explainer.shap_values(
                X
            )



            fig = plt.figure(
                figsize=(8,4)
            )


            shap.plots.waterfall(

                shap.Explanation(

                    values=shap_values[0],

                    base_values=explainer.expected_value,

                    data=X.iloc[0].to_numpy(),

                    feature_names=features

                ),

                show=False

            )


            st.pyplot(fig)


# =====================================================
# Footer
# =====================================================


st.divider()


st.caption(

"""
CMU-iPedSep is an AI-assisted clinical decision support tool.
The predicted probability should be interpreted together with clinical judgement.
"""

)
