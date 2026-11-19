import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import random

# 設定網頁標題與圖示
st.set_page_config(page_title="情緒分析與預測儀表板", page_icon="📊", layout="centered")

st.title("📊 產品評論情緒分析與預測系統")
st.markdown("這是一個結合 **規則/關鍵字模擬評分** 與 **XGBoost 機器學習預測** 的互動式系統。")

# 1. 初始化資料集與訓練模型 (在實務中，建議讀取已存檔的模型與 Vectorizer)
@st.cache_resource
def get_trained_model():
    data = {
        'reviews': [
            "這款產品真的太棒了！品質超出預期，物超所值，下次一定會再買。",
            "出貨速度蠻快的，包裝也很完整，大致上還滿意。",
            "用起來感覺普通，沒有想像中好，但還可以接受。",
            "有些小瑕疵，說明書寫得不是很清楚，使用起來有點不順手。",
            "非常糟糕的購物體驗！拿到沒幾天就壞了，客服態度也極差，千萬別買！",
            "非常好，極力推薦！",
            "品質一般般，不會再買了。",
            "送貨超慢，商品還有受損，氣死了！",
            "很棒的商品，CP值高。",
            "太爛了，根本不能用。"
        ],
        'ratings': [5, 4, 3, 2, 1, 5, 2, 1, 5, 1]
    }
    df = pd.DataFrame(data)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['reviews']).toarray()
    y = (df['ratings'] > 3).astype(int)
    
    model = xgb.XGBClassifier(eval_metric='logloss')
    model.fit(X, y)
    return vectorizer, model

vectorizer, xgb_model = get_trained_model()

# 模擬關鍵字星等比對
def simulate_sentiment_by_keywords(text):
    text_lower = text.lower()
    if any(w in text_lower for w in ["太棒", "超棒", "極好", "物超所值", "非常滿意", "五星"]):
        return "5"
    elif any(w in text_lower for w in ["好", "滿意", "不錯", "推薦", "讚"]):
        return "4"
    elif any(w in text_lower for w in ["太爛", "極差", "糟糕", "爛", "垃圾", "騙人", "別買"]):
        return "1"
    elif any(w in text_lower for w in ["差", "瑕疵", "不順", "失望", "不滿"]):
        return "2"
    return "3"

# 介面設計
st.subheader("✍️ 請輸入您的商品評論")
user_input = st.text_area("在下方輸入一段文字：", value="這家店的服務態度很好，但商品稍微貴了一點。", height=100)

if st.button("🚀 進行分析", use_container_width=True):
    if user_input.strip() == "":
        st.warning("⚠️ 請先輸入評論內容再進行分析！")
    else:
        st.divider()
        st.subheader("🎯 預測結果展示")
        
        # 建立兩欄顯示結果
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("📊 星等評估結果")
            score = simulate_sentiment_by_keywords(user_input)
            score_num = int(score)
            st.markdown(f"### **{score} / 5 顆星**")
            st.markdown("⭐" * score_num)
            
        with col2:
            st.success("🤖 XGBoost 模型預測")
            # 特徵處理與模型預測
            vectorized_text = vectorizer.transform([user_input]).toarray()
            prediction = xgb_model.predict(vectorized_text)[0]
            
            if prediction == 1:
                st.markdown("### **🟢 正面評論**")
                st.caption("(系統判定 Rating > 3)")
            else:
                st.markdown("### **🔴 負面評論**")
                st.caption("(系統判定 Rating <= 3)")
