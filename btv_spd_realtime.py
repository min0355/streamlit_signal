import time  
import pandas as pd
import numpy as np
import streamlit as st 
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from keras.models import load_model



st.set_page_config(
    page_title="BTV_AV4 8K Spindle Survival Test (72 hr)",
    layout="wide",
)



# Data Prepare
FilePath = "/home/kym/ML/output/btv_spd/streamlit/TEST_AV4_8K_72 hr survival_augmentation.csv"
scaler_call = joblib.load("/home/kym/ML/output/btv_spd/streamlit/btv_spd_rscaler.pkl")
model_call = load_model("/home/kym/ML/output/btv_spd/streamlit/btv_av4_spd_pretrained_model.h5")

@st.experimental_memo
def get_data() -> pd.DataFrame:
    return pd.read_csv(FilePath)

df = get_data()
df_length = df.shape[0]




# dash-board title

st.title("AV4 8K Spindle Max Survival Test (72 hr)")



placeholder_1 = st.empty()
placeholder_2 = st.empty()
# near real-time simulation
Pie_Value = [0, 0]

for minutes in range(df_length):
    
    ndf = df.iloc[minutes:minutes+30]
    result_mapping = {
        "OK":0,
        "NG":1
    }
    ndf.loc[:, "label"] = ndf.label.map(result_mapping)
    
    
    
    with placeholder_1.container():
        Condition_Record, Condition_Est, Dummy_Area = st.columns((1,1,1))

        with Condition_Record:
            st.markdown("## Recorded Condition (Real value)")
            label_Value = ndf["label"][minutes+29]

            if label_Value == 0:
                st.success("OK")
            else:
                st.error("NG")

        with Condition_Est:
            
            st.markdown("## Estimated Condition By AI")
            ndf.drop(['Index'], axis=1)
            
            new_x_df = pd.concat([ndf['sensor'][29:], ndf['speed'][29:], ndf['label'][29:]], axis=1)
            new_x_df_scale = scaler_call.transform(new_x_df)
            new_x_df_scale = np.delete(new_x_df_scale, 2, axis=1)
            new_x_df_scale = new_x_df_scale.reshape(1,1,2)
            label_Est = model_call.predict(new_x_df_scale)
            
            
            if label_Est < 0.5:
                st.success("OK")
                Pie_Value[0] = Pie_Value[0] + 1
            else:
                st.error("NG")
                Pie_Value[1] = Pie_Value[1] + 1

        
    with placeholder_2.container():
        
        Spd_Graph, Fail_Ratio, Spd_Data = st.columns((2,1,1))
        with Spd_Graph:
            st.markdown("## Spindle Condition")
            fig = make_subplots(rows=2, cols=1,
                                subplot_titles=("Sensor Signal", "Motor Speed"))
            fig.update_layout(margin=dict(l = 20,
                                         r=20,
                                         b=50,
                                         t=20,
                                         pad = 2))
            fig.add_trace(go.Scatter(x=ndf.Index, y=ndf.sensor,
                                        mode = "lines",
                                        name = 'Vibration signal'),
                                        row=1, col=1)
            fig.add_trace(go.Scatter(x=ndf.Index, y=ndf.speed,
                                        mode = "lines",
                                        name = 'Spindle Speed (r/min)'),
                                        row=2, col=1)

            
            fig.add_annotation(x=ndf.Index[minutes+29], y=ndf.sensor[minutes+29],
                                text = 'CV',
                                showarrow = True,
                                arrowhead= 1,
                                row=1, col=1)
            fig.add_annotation(x=ndf.Index[minutes+29], y=ndf.speed[minutes+29],
                                text = 'CV',
                                showarrow = True,
                                arrowhead= 1,
                                row=2, col=1)

            st.plotly_chart(fig, use_container_width= True)
            
        with Fail_Ratio:
            st.markdown("## Estimated NG Ratio")
            labels = ['OK', 'NG']
            
            fig = go.Figure(data =[go.Pie(labels = labels, values= Pie_Value, hole=0.3)])
            st.plotly_chart(fig, use_container_width= True)
            
                
        with Spd_Data:
            st.markdown("## Recorded Data View")
            view_NDF = ndf.drop(['timestamp'], axis=1)
            result_mapping = {
                0: "OK",
                1: "NG"
                }
            view_NDF.loc[:,"label"]=view_NDF.label.map(result_mapping)
            st.dataframe(view_NDF[20:])


        
        time.sleep(1)
