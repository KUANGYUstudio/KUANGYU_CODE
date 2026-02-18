import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import numpy as np
import sys
import os
import subprocess

# --- 1. 介面設定 (加入 Wide Mode 和 icon) ---
st.set_page_config(
    page_title="光聿KUANGYU - AI 動作實驗室",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS 極致美化 (像 App 一樣的質感) ---
st.markdown("""
    <style>
    /* 全局字體優化 */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* 標題樣式 */
    h1 {
        color: #0E1117;
        font-weight: 700;
        letter-spacing: -1px;
    }
    
    /* 側邊欄按鈕美化 */
    div.stButton > button:first-child {
        border-radius: 8px;
        font-weight: bold;
        border: 1px solid #E0E0E0;
    }
    
    /* 強調主要按鈕 (紅色系，呼應光聿) */
    div.stButton > button.st-emotion-cache-19rxjzo {
        border-color: #FF4B4B;
        color: #FF4B4B;
    }

    /* Checkbox 文字加大，方便手指點擊 */
    .stCheckbox label {
        font-size: 16px !important;
        font-weight: 600;
        cursor: pointer;
    }
    
    /* 調整列間距，讓介面更緊湊 */
    div[data-testid="column"] {
        padding: 2px;
    }
    
    /* 隱藏預設選單 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 核心初始化 (快取加速) ---
@st.cache_resource
def load_mediapipe():
    return mp.solutions.pose, mp.solutions.drawing_utils

mp_pose, mp_drawing = load_mediapipe()

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180.0 else angle

def draw_dashboard(image, label, angle, x, y, color):
    overlay = image.copy()
    # 半透明黑底，讓數據更清楚
    cv2.rectangle(overlay, (x, y - 30), (x + 180, y + 15), (20, 20, 20), -1) 
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # 文字美化
    cv2.putText(image, f"{label}", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(image, f"{int(angle)}", (x + 110, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(image, (x + 160, y - 5), 4, (255, 255, 255), 1) # 畫一個圓圈代表度數

# --- Session State 初始化 ---
if 'analyzed_data' not in st.session_state: st.session_state['analyzed_data'] = [] 
if 'video_meta' not in st.session_state: st.session_state['video_meta'] = {}
if 'source_video_path' not in st.session_state: st.session_state['source_video_path'] = None
if 'current_file_name' not in st.session_state: st.session_state['current_file_name'] = ""
if 'is_processed' not in st.session_state: st.session_state['is_processed'] = False
if 'uploader_key' not in st.session_state: st.session_state['uploader_key'] = 0

# --- 初始化部位選擇狀態 (為了做一鍵全選) ---
body_parts = ['l_hip', 'l_knee', 'l_ankle', 'r_hip', 'r_knee', 'r_ankle']
for part in body_parts:
    if part not in st.session_state: st.session_state[part] = False # 預設全關
    if f"t_{part}" not in st.session_state: st.session_state[f"t_{part}"] = False

# --- UX 功能函數：一鍵切換 ---
def toggle_all(state):
    for part in body_parts:
        st.session_state[part] = state
        st.session_state[f"t_{part}"] = state # 連殘影也一起開關

# --- Sidebar 控制中心 ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2972/2972185.png", width=50) # 加入一個單車小圖示 (可換成工作室Logo)
    st.title("控制中心")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "📂 步驟 1: 上傳影片", 
        type=["mp4", "mov"], 
        key=f"uploader_{st.session_state['uploader_key']}"
    )
    
    st.markdown("---")
    
    # 重新整理按鈕
    if st.button("🔄 重置 / 上傳新檔", use_container_width=True):
        st.session_state['is_processed'] = False
        st.session_state['current_file_name'] = ""
        st.session_state['analyzed_data'] = []
        st.session_state['uploader_key'] += 1
        st.rerun()
        
    # [修改 1] 這裡改成全大寫 KUANGYU
    st.caption("KUANGYU Studio v9.9 UX Pro")

# --- Main 主畫面 ---
# [修改 2] 這裡拿掉 emoji，改成指定標題
st.title("光聿KUANGYU｜AI動作實驗室")

if uploaded_file:
    # 載入影片
    if uploaded_file.name != st.session_state['current_file_name']:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()
        st.session_state['current_file_name'] = uploaded_file.name
        st.session_state['source_video_path'] = tfile.name
        st.session_state['is_processed'] = False
        st.session_state['analyzed_data'] = []

    # 狀態 1: 尚未分析 (顯示大按鈕)
    if not st.session_state['is_processed']:
        st.info(f"✅ 已載入影片: {uploaded_file.name}")
        
        # 使用 columns 讓按鈕置中或好看一點
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("🚀 啟動 AI 智能掃描", type="primary", use_container_width=True):
                with st.spinner("🔍 AI 正在建立 3D 骨架模型..."):
                    cap = cv2.VideoCapture(st.session_state['source_video_path'])
                    if not cap.isOpened():
                        st.error("❌ 影片格式錯誤")
                    else:
                        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        MAX_WIDTH = 960 
                        if orig_width > MAX_WIDTH:
                            scale_factor = MAX_WIDTH / orig_width
                            new_width = MAX_WIDTH
                            new_height = int(orig_height * scale_factor)
                        else:
                            new_width = orig_width
                            new_height = orig_height
                            
                        st.session_state['video_meta'] = {'width': new_width, 'height': new_height, 'fps': fps, 'total_frames': total_frames}
                        temp_landmarks = []
                        bar = st.progress(0)
                        
                        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
                            frame_count = 0
                            while cap.isOpened():
                                ret, frame = cap.read()
                                if not ret: break
                                if orig_width > MAX_WIDTH:
                                    frame = cv2.resize(frame, (new_width, new_height))
                                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                results = pose.process(image)
                                temp_landmarks.append(results.pose_landmarks if results.pose_landmarks else None)
                                frame_count += 1
                                if total_frames > 0: bar.progress(min(frame_count/total_frames, 1.0))
                        cap.release()
                        st.session_state['analyzed_data'] = temp_landmarks
                        st.session_state['is_processed'] = True
                        st.rerun()

    # 狀態 2: 分析完成，顯示儀表板
    else:
        meta = st.session_state['video_meta']
        landmarks_data = st.session_state['analyzed_data']
        
        # 左右佈局：左邊是影片，右邊是控制面板 (更符合寬螢幕習慣)
        col_video, col_ctrl = st.columns([2, 1])
        
        with col_ctrl:
            st.markdown("### 🛠 數據與殘影設定")
            
            # --- UX 神器：一鍵全選按鈕 ---
            b1, b2 = st.columns(2)
            with b1:
                if st.button("✅ 全選", use_container_width=True):
                    toggle_all(True)
                    st.rerun()
            with b2:
                if st.button("⬜ 重選", use_container_width=True):
                    toggle_all(False)
                    st.rerun()

            st.markdown("---")
            
            # 使用 Tabs 分頁，讓介面更乾淨
            tab1, tab2 = st.tabs(["👈 左側數據", "👉 右側數據"])
            
            with tab1:
                lc1, lc2 = st.columns([3, 1])
                st.session_state['l_hip'] = lc1.checkbox("左髖 (Hip)", value=st.session_state['l_hip'])
                st.session_state['t_l_hip'] = lc2.checkbox("殘影", value=st.session_state['t_l_hip'], key="k_t_l_hip", disabled=not st.session_state['l_hip'])

                lc3, lc4 = st.columns([3, 1])
                st.session_state['l_knee'] = lc3.checkbox("左膝 (Knee)", value=st.session_state['l_knee'])
                st.session_state['t_l_knee'] = lc4.checkbox("殘影", value=st.session_state['t_l_knee'], key="k_t_l_knee", disabled=not st.session_state['l_knee'])

                lc5, lc6 = st.columns([3, 1])
                st.session_state['l_ankle'] = lc5.checkbox("左踝 (Ankle)", value=st.session_state['l_ankle'])
                st.session_state['t_l_ankle'] = lc6.checkbox("殘影", value=st.session_state['t_l_ankle'], key="k_t_l_ankle", disabled=not st.session_state['l_ankle'])

            with tab2:
                rc1, rc2 = st.columns([3, 1])
                st.session_state['r_hip'] = rc1.checkbox("右髖 (Hip)", value=st.session_state['r_hip'])
                st.session_state['t_r_hip'] = rc2.checkbox("殘影", value=st.session_state['t_r_hip'], key="k_t_r_hip", disabled=not st.session_state['r_hip'])

                rc3, rc4 = st.columns([3, 1])
                st.session_state['r_knee'] = rc3.checkbox("右膝 (Knee)", value=st.session_state['r_knee'])
                st.session_state['t_r_knee'] = rc4.checkbox("殘影", value=st.session_state['t_r_knee'], key="k_t_r_knee", disabled=not st.session_state['r_knee'])

                rc5, rc6 = st.columns([3, 1])
                st.session_state['r_ankle'] = rc5.checkbox("右踝 (Ankle)", value=st.session_state['r_ankle'])
                st.session_state['t_r_ankle'] = rc6.checkbox("殘影", value=st.session_state['t_r_ankle'], key="k_t_r_ankle", disabled=not st.session_state['r_ankle'])

            # 資料庫對應
            metrics_db = {
                "l_hip":   ("L-Hip", mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, (50, 255, 50), mp_pose.PoseLandmark.LEFT_HIP),
                "l_knee":  ("L-Knee", mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, (50, 255, 50), mp_pose.PoseLandmark.LEFT_KNEE),
                "l_ankle": ("L-Ankle", mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX, (50, 255, 50), mp_pose.PoseLandmark.LEFT_ANKLE),
                "r_hip":   ("R-Hip", mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, (255, 255, 50), mp_pose.PoseLandmark.RIGHT_HIP),
                "r_knee":  ("R-Knee", mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE, (255, 255, 50), mp_pose.PoseLandmark.RIGHT_KNEE),
                "r_ankle": ("R-Ankle", mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX, (255, 255, 50), mp_pose.PoseLandmark.RIGHT_ANKLE),
            }

            active_metrics = []
            keys = ['l_hip', 'l_knee', 'l_ankle', 'r_hip', 'r_knee', 'r_ankle']
            for k in keys:
                if st.session_state[k]:
                    active_metrics.append(metrics_db[k] + (st.session_state[f"t_{k}"],))

        # --- 影片生成 ---
        tfile_output_avi = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
        output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
        out = cv2.VideoWriter(tfile_output_avi, fourcc, meta['fps'], (meta['width'], meta['height']))
        cap = cv2.VideoCapture(st.session_state['source_video_path'])
        
        w, h = meta['width'], meta['height']
        total_frames = meta['total_frames']
        
        # 儀表板位置調整
        dashboard_positions = {
            "L-Hip": (20, 100), "L-Knee": (20, 160), "L-Ankle": (20, 220),
            "R-Hip": (w - 200, 100), "R-Knee": (w - 200, 160), "R-Ankle": (w - 200, 220)
        }
        
        frame_idx = 0
        TRAIL_LENGTH = 60 
        STEP = 2          
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        # 顯示即時進度
        status_text = st.empty()
        progress_bar = st.progress(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if orig_w > 960: frame = cv2.resize(frame, (w, h))

            current_landmarks = landmarks_data[frame_idx] if frame_idx < len(landmarks_data) else None
            
            if current_landmarks:
                mp_drawing.draw_landmarks(
                    frame, current_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1), # 骨架線改細一點，更有質感
                    mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=1)
                )
                lm = current_landmarks.landmark
                
                # 標題
                cv2.putText(frame, "LEFT SIDE", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2, cv2.LINE_AA)
                cv2.putText(frame, "RIGHT SIDE", (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 50), 2, cv2.LINE_AA)
                
                for label, idx_a, idx_b, idx_c, color, track_idx, show_trail_flag in active_metrics:
                    try:
                        p1 = [lm[idx_a.value].x, lm[idx_a.value].y]
                        p2 = [lm[idx_b.value].x, lm[idx_b.value].y]
                        p3 = [lm[idx_c.value].x, lm[idx_c.value].y]
                        angle = calculate_angle(p1, p2, p3)
                        
                        if label in dashboard_positions:
                            pos_x, pos_y = dashboard_positions[label]
                            draw_dashboard(frame, label, angle, pos_x, pos_y, color)
                    except: pass
                    
                    if show_trail_flag:
                        for t in range(STEP, TRAIL_LENGTH, STEP):
                            prev_idx = frame_idx - t
                            if prev_idx >= 0:
                                lm_prev = landmarks_data[prev_idx]
                                if lm_prev:
                                    pt = (int(lm_prev.landmark[track_idx.value].x * w), int(lm_prev.landmark[track_idx.value].y * h))
                                    cv2.circle(frame, pt, 3, color, -1) 
            out.write(frame)
            frame_idx += 1
            if total_frames > 0 and frame_idx % 5 == 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))
                status_text.text(f"🎨 AI 繪圖運算中: {int(frame_idx/total_frames*100)}%")
        
        cap.release()
        out.release()
        
        progress_bar.empty()
        status_text.text("⚙️ 最終壓縮轉檔中...")
        
        with st.spinner("🎬 製作最終影片 (H.264)..."):
            subprocess.call([
                'ffmpeg', '-y', '-i', tfile_output_avi, 
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '25',
                output_video_path
            ])
        status_text.empty()
        
        # --- 最終結果呈現區 ---
        with col_video:
            st.success("✨ 分析完成！")
            st.video(output_video_path)
            with open(output_video_path, 'rb') as f:
                video_bytes = f.read()
            # 下載按鈕改大一點
            st.download_button("📥 下載分析影片至手機", video_bytes, "kuangyu_analysis.mp4", "video/mp4", type="primary", use_container_width=True)

else:
    # 初始歡迎畫面
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 50px;'>
        <h3>👈 請從左側選單上傳訓練影片</h3>
        <p>支援格式: MP4, MOV (建議長度: 15-30秒)</p>
    </div>
    """, unsafe_allow_html=True)