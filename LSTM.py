# 1. ติดตั้งไลบรารีที่จำเป็น
!pip install tensorflow gradio pandas numpy scikit-learn matplotlib seaborn -q

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import io
from PIL import Image

# ตัวแปรระบบ
trained_model = None
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
feature_cols = ['heart_rate', 'body_temperature', 'hydration_level',
                'sleep_quality', 'recovery_score', 'stress_level',
                'training_load', 'training_intensity']
time_steps = 10

# 2. ฟังก์ชันเทรนโมเดล LSTM และรันประมวลผลทั้งไฟล์ CSV
def process_csv_and_predict(file_obj):
    global trained_model, scaler_X, scaler_y, feature_cols
    if file_obj is None:
        return "❌ กรุณาอัปโหลดไฟล์ CSV ก่อนครับ", None, None

    try:
        df = pd.read_csv(file_obj.name)
        df_sub = df[feature_cols + ['fatigue_index']].dropna().reset_index(drop=True)

        if len(df_sub) < 20:
            return "❌ ข้อมูลในไฟล์มีน้อยเกินไป กรุณาตรวจสอบไฟล์ CSV", None, None

        data_X = df_sub[feature_cols].values
        data_y = df_sub['fatigue_index'].values.reshape(-1, 1)

        scaled_X = scaler_X.fit_transform(data_X)
        scaled_y = scaler_y.fit_transform(data_y)

        X_seq, y_seq = [], []
        for i in range(len(scaled_X) - time_steps):
            X_seq.append(scaled_X[i:(i + time_steps)])
            y_seq.append(scaled_y[i + time_steps])

        X_seq, y_seq = np.array(X_seq), np.array(y_seq)

        # สร้างและเทรนโมเดล LSTM
        model = Sequential([
            LSTM(32, return_sequences=False, input_shape=(time_steps, len(feature_cols))),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_seq, y_seq, epochs=8, batch_size=32, verbose=0)
        trained_model = model

        # ทำนายผลสำหรับชุดข้อมูลทดสอบ (เช่น 15 แถวแรก)
        preds_scaled = model.predict(X_seq[:15])
        preds = scaler_y.inverse_transform(preds_scaled).flatten()

        # สร้างตารางผลลัพธ์
        results_data = []
        bar_colors = []
        for idx, val in enumerate(preds):
            if val < 50:
                zone = "🟢 Green Zone (Safe - ปกติ)"
                bar_colors.append("#2e7d32") # เขียว
            elif val < 65:
                zone = "🟡 Yellow Zone (Fatigue Onset - เริ่มล้า)"
                bar_colors.append("#fbc02d") # เหลือง
            elif val < 80:
                zone = "🟠 Orange Zone (Fatigue - ล้า)"
                bar_colors.append("#ef6c00") # ส้ม
            else:
                zone = "🔴 Red Zone (High Risk - ความเสี่ยงสูง)"
                bar_colors.append("#FF0000") # แดง

            results_data.append([idx + 1, round(float(val), 6), zone])

        result_df = pd.DataFrame(results_data, columns=["Test Sequence ID (Test Data)", "คะแนนความล้า / ระดับความล้า (%)", "สถานะการประเมิน (Heat Map)"])

        # สร้างกราฟ Bar Chart ตามสไตล์ในรูปตัวอย่าง
        fig, ax = plt.subplots(figsize=(10, 4.5))
        bars = ax.bar(range(1, len(preds) + 1), preds, color=bar_colors, edgecolor='black', linewidth=0.6)

        # เส้น Thresholds
        ax.axhline(y=50, color='#fbc02d', linestyle='--', linewidth=1, label='Fatigue Onset (50%)')
        ax.axhline(y=65, color='#ef6c00', linestyle='--', linewidth=1, label='High Risk (65%)')
        ax.axhline(y=80, color='#d32f2f', linestyle='--', linewidth=1, label='Critical Threshold (80%)')

        ax.set_ylim(0, 100)
        ax.set_xlim(0, len(preds) + 1)
        ax.set_xlabel("Test Sequence ID", fontsize=11, fontweight='bold')
        ax.set_ylabel("Risk Score (%)", fontsize=11, fontweight='bold')
        ax.set_title("Dynamic Heat Map Risk Assessment (LSTM Model Output)", fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', frameon=True)
        plt.tight_layout()

        # แปลงกราฟเป็นรูปภาพสำหรับ Gradio
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        chart_img = Image.open(buf)
        plt.close(fig)

        status_msg = f"✅ บันทึกโมเดลและประมวลผลจากไฟล์ CSV เรียบร้อยแล้ว!"
        return status_msg, result_df, chart_img

    except Exception as e:
        return f"❌ เกิดข้อผิดพลาด: {str(e)}", None, None

# 3. สร้างหน้า UI ด้วย Gradio Blocks
with gr.Blocks() as demo:
    gr.Markdown("# 🏃‍♂️ LSTM Model Heatmap")
    gr.Markdown("อัปโหลดไฟล์ CSV เพื่อให้โมเดล LSTM คำนวณความเสี่ยงเหนื่อยล้า และแสดงผลเป็นตารางและกราฟ Heatmap อัตโนมัติ")

    with gr.Row():
        file_input = gr.File(label="อัปโหลดไฟล์ csv")

    run_btn = gr.Button("🚀 ประมวลผลและสร้าง Heatmap", variant="primary")
    status_output = gr.Textbox(label="สถานะระบบ")

    gr.Markdown("---")

    # ตารางผลลัพธ์
    table_output = gr.Dataframe(label="ตารางแสดงผลลัพธ์ความเสี่ยง (Test Data & Heat Map Status)")

    # กราฟ Heatmap
    chart_output = gr.Image(label="Dynamic Heat Map Risk Assessment (LSTM Model Output)")

    # ผูกปุ่มการทำงาน
    run_btn.click(
        fn=process_csv_and_predict,
        inputs=[file_input],
        outputs=[status_output, table_output, chart_output]
    )

# รันแอปพลิเคชัน
demo.launch(theme=gr.themes.Soft(), share=True, debug=True)