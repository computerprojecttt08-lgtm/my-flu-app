import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
import datetime

# 1. ตั้งค่าหน้าตาแอป (Mobile First)
st.set_page_config(page_title="Flu Forecast", page_icon="🌡️", layout="centered")

# 2. ปรับแต่งความสวยงามด้วย CSS (เน้นวงกลมและสีสดใสตามภาพตัวอย่าง)
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #FFFFFF 0%, #FFF5F5 100%); }
    .circle-container { display: flex; justify-content: center; padding: 20px; }
    .circle {
        width: 220px; height: 220px;
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);
        border-radius: 50%;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        color: white; box-shadow: 0 10px 25px rgba(255, 107, 107, 0.4);
        border: 6px solid white;
    }
    .circle-val { font-size: 45px; font-weight: bold; line-height: 1; }
    .circle-unit { font-size: 16px; opacity: 0.9; margin-top: 5px; }
    .title-text { color: #D32F2F; text-align: center; font-weight: bold; font-size: 24px; }
    </style>
    """, unsafe_allow_html=True)

# 3. โหลดข้อมูลและประมวลผล (Logic จาก flu_autoSRM.py)
@st.cache_data
def get_data_and_predict():
    df = pd.read_excel('data_flu.xlsx')
    df = df.rename(columns={'date_dt': 'Date', 'Patient rate per 100,000': 'Flu_Rate'})
    df['Date'] = pd.to_datetime(df['Date'])
    
    # ใช้ค่า Order ที่เสถียรที่สุดสำหรับข้อมูลของคุณ
    model = SARIMAX(df['Flu_Rate'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
    results = model.fit(disp=False)
    
    forecast = results.get_forecast(steps=1)
    pred_rate = max(0, forecast.predicted_mean.iloc[0])
    
    # คำนวณเป็นจำนวนคน (อ้างอิงประชากรจากโค้ดเดิมของคุณ)
    POPULATION = 66097304 
    pred_cases = pred_rate * (POPULATION / 100000)
    
    return pred_cases, pred_rate, df

try:
    pred_cases, pred_rate, df = get_data_and_predict()

    # 4. ส่วนแสดงผลบนมือถือ
    st.markdown("<p class='title-text'>พยากรณ์ผู้ป่วยไข้หวัดใหญ่สัปดาห์หน้า</p>", unsafe_allow_html=True)

    # แสดงวงกลมตัวเลข
    st.markdown(f"""
        <div class="circle-container">
            <div class="circle">
                <div class="circle-unit">คาดการณ์ผู้ป่วยใหม่</div>
                <div class="circle-val">{int(pred_cases):,}</div>
                <div class="circle-unit">ราย</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"<p style='text-align: center; color: #555;'>คิดเป็นอัตรา {pred_rate:.2f} ต่อแสนประชากร</p>", unsafe_allow_html=True)

    # 5. กราฟแนวโน้มด้านล่าง
    st.subheader("📊 แนวโน้มย้อนหลังและการพยากรณ์")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Flu_Rate'], name='ข้อมูลจริง', line=dict(color='#FF6B6B', width=3)))
    
    next_date = df['Date'].max() + datetime.timedelta(days=7)
    fig.add_trace(go.Scatter(x=[next_date], y=[pred_rate], name='จุดพยากรณ์', 
                             marker=dict(color='#D32F2F', size=12, symbol='star')))

    fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=300, 
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error("กรุณาตรวจสอบไฟล์ data_flu.xlsx ในโฟลเดอร์แอปของคุณ")
