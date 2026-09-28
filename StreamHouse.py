import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

def train_test_split(Data_X, Data_Y):
    N = len(Data_X)

    X_Train = Data_X.loc[:int(N * 0.8)]
    X_Test = Data_X.loc[int(N * 0.8):]
    Y_Train = Data_Y.loc[:int(N * 0.8)]
    Y_Test = Data_Y.loc[int(N * 0.8):]

    return [X_Train, X_Test, Y_Train, Y_Test]

def sep_cat(Data_Y):
    col = Data_Y.columns
    String_Col = Data_Y[col].select_dtypes(include=['object', 'string']).columns.tolist()
    for i in String_Col:
        Data_Y[i] = Data_Y.groupby(i).ngroup()
    return Data_Y

def MSE(new_pred, Y):
    diff = Y - new_pred
    sq = diff ** 2
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(Y, new_pred)
    ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='red')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)


st.title("Multivariate")


uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"],
    accept_multiple_files=False
)

if uploaded_file:
    DF = pd.read_csv(uploaded_file)
    st.header("Dataframe")
    st.dataframe(DF)
    col = np.array(DF.columns)
    Base = st.selectbox("Select Base Option", col, index = None)

    if Base is not None:
        col = col[col != Base]
        st.title("Remove the Options")

        arr = []
        columns = st.columns(5)
        for i, o in enumerate(col):
            with columns[i % 5]:
                arr.append(st.checkbox(o))

        col = col[np.invert(arr)]

        Data_X = DF[Base]
        Cat_Col = DF[col].select_dtypes(include=['object', 'string']).columns.tolist()
        Data_Y = sep_cat(DF[col])

        st.header("Updated DataFrame")
        st.dataframe(Data_Y)

        N = len(Data_X)
        X = np.hstack((np.ones((N, 1)), Data_Y.values))
        A = (np.linalg.inv((X.T) @ X) @ (X.T)) @ np.array(Data_X)
        y = X @ A
        Intercept = A[0]
        Slope = A[1:]
        mse = np.mean((np.array(Data_X) - y) ** 2)

        st.write("Predicted Value")
        st.success(y)
        st.write("Intercept")
        st.success(Intercept)
        st.write("Slope")
        st.success(Slope)
        st.write("MSE")
        st.success(mse)

        st.header("Graphs")
        mse_value = MSE(y, Data_X)

        Graph_Col = st.columns(2)
        for i, o in enumerate(col):
            with Graph_Col[i % 2]:
                step = max(1, N // 3000)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(Data_Y[o][::step], Data_X[::step], s=3, alpha=0.3)
                line = Intercept + Slope[i] * Data_Y[o]
                ax.plot(Data_Y[o], line, color='red')
                ax.set_title(o)
                ax.set_ylabel(Base)
                st.pyplot(fig)
                plt.close(fig)

        st.header("Predict " + Base)
        inp = []
        for o in col:
            if o in Cat_Col:
                mp = DF.groupby(o).ngroup().groupby(DF[o]).first().to_dict()
                inp.append(mp[st.selectbox(o, list(mp))])
            else:
                inp.append(st.number_input(o, value=float(DF[o].median())))

        st.success("Predicted " + Base + " : " + str(np.hstack(([1], np.array(inp, dtype=float))) @ A))