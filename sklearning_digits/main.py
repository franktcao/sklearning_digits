import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix as get_confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from plotly.subplots import make_subplots


st.set_page_config(layout="wide")
st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    # padding-left: 5rem;
                    # padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    return load_digits(as_frame=True).frame

@st.cache_data
def split_data(features, target, test_set_ratio: float = 0.2):
    return train_test_split(features, target, test_size=test_set_ratio,)

@st.cache_data
def train_model(features, labels, **kwargs):
    model = XGBClassifier(n_jobs=-1, **kwargs)
    # st.write(model.get_params())
    model.fit(features, labels)
    return model



if __name__ == "__main__":
    df = load_data()
    with st.sidebar:
        st.write("## Configuration")
        with st.expander("### Dataset Params"):
            total_samples = st.slider("Total Samples", value=df.shape[0], min_value=2, max_value=df.shape[0])
            train_set_ratio = st.slider("Training Set Ratio", value=0.8, min_value=0.0, max_value=1.0)
        test_set_ratio = 1 - train_set_ratio
        
        with st.expander("### Model Params"):
            n_estimators = st.selectbox("N. Estimators", [1, 100, 200, 500, 1000], 1)
            max_depth = st.selectbox("Max Depth", [None, 10, 50, 100, 500, 1000])
            learning_rate = st.selectbox("Learning Rate", [None, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000])
            gamma = st.selectbox("Min. Split Loss", [None, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000])
        
        model_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "gamma": gamma,
        }

    df = df.sample(total_samples)
    target = df["target"]
    features = df[(col for col in df.columns if col != "target")]
    features_train, features_test, labels_train, labels_test = (
        split_data(features, target, test_set_ratio)
    )
    
    model = train_model(features_train, labels_train, **model_params)

    predictions = model.predict(features_test)
    
    matches = (predictions == labels_test).reset_index(drop=True)
    match_indices = matches[matches].index.tolist()
    mismatch_indices = matches[~matches].index.tolist()

    st.write("# Summary")
    columns = st.columns(4)
    with columns[0]:
        st.metric("Total Samples: ", total_samples)
    with columns[1]:
        st.metric("Test Set Size: ", len(matches))
    with columns[2]:
        st.metric("Correct: ", len(match_indices))
    with columns[3]:
        st.metric("Incorrect: ", len(mismatch_indices))

    st.write("## Correct Examples")
    good_index = match_indices[0]
    if st.button("New correct example"):
        good_index = np.random.choice(match_indices)
    index = good_index
    digit_raw = features_test.iloc[index].values.astype(int)
    digit = digit_raw.reshape(8, 8)
    columns = st.columns(3)
    with columns[0]:
        st.write("## Data\n\n")
        st.text("")
        st.text("")
        st.text("")
        st.table(digit)
        st.write(", ".join(str(x) for x in digit_raw.tolist()))
    with columns[1]:
        st.write("## Image")
        fig = px.imshow(digit, text_auto=True, aspect=1)
        fig.update_layout(xaxis=dict(tickmode='linear',))
        st.plotly_chart(fig, use_container_width=True)
    with columns[2]:
        st.write("## Label/Prediction")
        st.text("")
        st.text("")
        st.text("")
        label = labels_test.iloc[index]
        st.metric("Label", label)
        prediction = predictions[index]
        st.metric("Prediction", prediction)
    
    st.write("## Incorrect Examples")
    bad_index = mismatch_indices[0]
    if st.button("New incorrect example"):
        bad_index = np.random.choice(mismatch_indices)
    index = bad_index
    digit_raw = features_test.iloc[index].values.astype(int)
    digit = digit_raw.reshape(8, 8)
    columns = st.columns(3)
    with columns[0]:
        st.write("## Data\n\n")
        st.text("")
        st.text("")
        st.text("")
        st.table(digit)
        st.write(", ".join(str(x) for x in digit_raw.tolist()))
    with columns[1]:
        st.write("## Image")
        fig = px.imshow(digit, text_auto=True, aspect=1)
        fig.update_layout(xaxis=dict(tickmode='linear',))
        st.plotly_chart(fig, use_container_width=True)
    with columns[2]:
        st.write("## Label/Prediction")
        st.text("")
        st.text("")
        st.text("")
        label = labels_test.iloc[index]
        st.metric("Label", label)
        prediction = predictions[index]
        st.metric("Prediction", prediction)


    st.write("# Evaluation")
    confusion_matrix = get_confusion_matrix(labels_test, predictions)

    fig = px.imshow(confusion_matrix, text_auto=True, aspect=1)
    fig.update_layout(
        xaxis_title="Prediction",
        yaxis_title="Truth Label",
        xaxis=dict(tickmode='linear'), 
        yaxis=dict(tickmode='linear'),
    )
    st.plotly_chart(fig, use_container_width=True)

    titles = np.arange(0, 10).astype(str).tolist()
    multi_plot = make_subplots(
        rows=10, 
        cols=10, 
        y_title="Truth Label",
        x_title="Predictions", 
        row_titles=titles,
        column_titles=titles,
    )

    df = pd.concat(
        [
            features_test.reset_index(drop=True), 
            labels_test.reset_index(drop=True), 
            pd.Series(predictions, name="prediction")
        ], 
        axis="columns",
    )
    
    st.button("New samples")
    for row in range(1, 11):
        label_df = df[df["target"] == row - 1]
        if len(label_df) == 0:
            continue
        for col in range(1, 11):
            this_df = label_df[label_df["prediction"] == col - 1]
            n_samples = len(this_df)
            if n_samples == 0:
                continue
                this_plot = px.imshow(np.zeros((8, 8)), text_auto=True, aspect=1) 
            else:
                cols = [col for col in this_df if col.startswith("pix")]
                cols.sort(reverse=True)
                i_sample = np.random.choice(n_samples)
                this_digit = this_df[cols].iloc[i_sample].values.reshape(8, 8)
                this_digit = np.flip(this_digit, axis=1)
                this_digit *= n_samples / 16
                this_plot = px.imshow(
                    this_digit, text_auto=True, aspect=1, 
                                    #   color_continuous_scale="Jet",
                                      color_continuous_scale="Rainbow",
                                      ) 
            
            multi_plot.add_trace(
                this_plot.data[0], row=row, col=col,
            )
    multi_plot.update_layout(
        height=800, width=800, yaxis_title="Y AXIS TITLE",
        # y_title="Truth Label",
        # color_continuous_scale="Rainbow",
        
    )
    multi_plot.update_xaxes(visible=False, showticklabels=False)
    multi_plot.update_yaxes(visible=False, showticklabels=False)
    multi_plot
    # st.plotly_chart(multi_plot, use_container_width=False)
    import plotly.graph_objects as go

    # go.Image(this_digit)