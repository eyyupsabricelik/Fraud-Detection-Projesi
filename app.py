import pandas as pd
import joblib
from flask import Flask, request, jsonify
from datetime import datetime

# --- 1. AYARLAR VE YÜKLEMELER ---
app = Flask(__name__)

print("⏳ Modeller yükleniyor...")
try:
    model = joblib.load('final_fraud_model.pkl')
    encoders = joblib.load('encoders_dict.pkl')
    print("✅ Model ve Encoder'lar başarıyla yüklendi!")
except FileNotFoundError:
    print("❌ HATA: .pkl dosyaları bulunamadı! Lütfen aynı klasörde olduklarından emin olun.")
    exit()

# --- 2. FEATURE ENGINEERING (NOTEBOOK'TAKİ MANTIK) ---
def apply_production_features(data):
    """
    Canlı sistemden gelen tek satırlık veri için özellik türetme.
    DİKKAT: Burada demo amaçlı anlık hesaplanıyor.
    """
    df_processed = data.copy()
    
    # Zaman Özellikleri
    df_processed['Timestamp'] = pd.to_datetime(df_processed['Timestamp'])
    df_processed['Hour'] = df_processed['Timestamp'].dt.hour
    df_processed['DayOfWeek'] = df_processed['Timestamp'].dt.dayofweek
    
    # Davranışsal Özellikler (Simülasyon)
    # Not: Tekil istekte geçmiş verisi olmadığı için bu değerler 
    # API'ye dışarıdan gönderilmeli veya veritabanından sorgulanmalıdır.
    # Eğer gönderilmezse, kodun çalışması için varsayılan değerler atıyoruz.
    if 'Customer_Freq' not in df_processed.columns:
        df_processed['Customer_Freq'] = 1 # İlk işlem varsayımı
    
    if 'Customer_Avg_Amount' not in df_processed.columns:
        # Geçmiş yoksa, ortalama = şu anki tutar olur (Amount_Diff = 0 olur)
        df_processed['Customer_Avg_Amount'] = df_processed['Amount (TRY)']
        
    df_processed['Amount_Diff'] = df_processed['Amount (TRY)'] - df_processed['Customer_Avg_Amount']
    
    # Gereksiz Sütunları Temizle
    cols_to_drop = ['Transaction ID', 'Customer ID', 'Timestamp']
    df_processed = df_processed.drop(columns=[c for c in cols_to_drop if c in df_processed.columns])
    
    return df_processed

# --- 3. API ENDPOINT (KARŞILAMA NOKTASI) ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Gelen JSON verisini al
        json_data = request.get_json()
        df = pd.DataFrame([json_data])
        
        # 2. Özellik Mühendisliği
        df_clean = apply_production_features(df)
        
        # 3. Encoding (Metin -> Sayı)
        for col, le in encoders.items():
            if col in df_clean.columns:
                val = df_clean[col].iloc[0]
                if val in le.classes_:
                    df_clean[col] = le.transform([val])
                else:
                    df_clean[col] = 0 
        
        # 3.5. Sütun Sırasını Eşitle (Reordering)
        # Modelin eğitim sırasında gördüğü sütun sırasını birebir uyguluyoruz.
        if hasattr(model, 'feature_names_in_'):
            df_clean = df_clean[model.feature_names_in_]

        # 4. Tahmin
        prediction = model.predict(df_clean)[0]
        probability = model.predict_proba(df_clean)[0][1] 
        
        # 5. Cevap Oluştur
        result = {
            "is_fraud": int(prediction),
            "fraud_probability": float(probability),
            "risk_level": "YÜKSEK" if probability > 0.7 else ("ORTA" if probability > 0.4 else "DÜŞÜK"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return jsonify(result)

    except Exception as e:
        # Hata mesajını daha detaylı görelim
        return jsonify({"error": str(e), "message": "Veri işlenirken hata oluştu."})

# --- 4. SUNUCUYU BAŞLAT ---
if __name__ == '__main__':
    print("🚀 Trendyol Fraud API 5001 portunda çalışıyor...")
    app.run(debug=True, port=5001)