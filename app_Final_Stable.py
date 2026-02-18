import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import numpy as np
import sys
import os

# --- 1. 介面設定 ---
st.set_page_config(
    page_title="光聿工作室 - AI 動作實驗室",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 強制導航保護 ---
venv_packages = os.path.join(os.getcwd(), "venv/lib/python3.10/site-packages")
if venv_packages not in sys.path:
    sys.path.insert(0, venv_packages)

# --- CSS 美化 ---
st.markdown("""
    <style>
    .main {background-color: #FFFFFF;}
    .stCheckbox label {
        font-size: 18px !important;
        font-weight: bold;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 核心初始化 ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180.0 else angle

def draw_dashboard(image, label, angle, x, y, color):
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y - 25), (x + 160, y + 10), (0, 0, 0), -1) 
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    cv2.putText(image, f"{label}:", (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    cv2.putText(image, f"{int(angle)}", (x + 110, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "o", (x + 145, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

# --- Session State ---
if 'analyzed_data' not in st.session_state: st.session_state['analyzed_data'] = [] 
if 'video_meta' not in st.session_state: st.session_state['video_meta'] = {}
if 'source_video_path' not in st.session_state: st.session_state['source_video_path'] = None
if 'current_file_name' not in st.session_state: st.session_state['current_file_name'] = ""
if 'is_processed' not in st.session_state: st.session_state['is_processed'] = False

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ 控制中心")
    st.markdown("---")
    uploaded_file = st.file_uploader("1. 上傳影片檔", type=["mp4", "mov"])
    st.markdown("---")
    if st.button("🔄 清除暫存 / 上傳新檔"):
        st.session_state['is_processed'] = False
        st.session_state['current_file_name'] = ""
        st.rerun()
    st.markdown("---")
    st.caption("光聿工作室 v9.2 Compatible")

# --- Main ---
st.title("🚴 光聿工作室 | AI 動作實驗室")

if uploaded_file:
    if uploaded_file.name != st.session_state['current_file_name']:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()
        st.session_state['current_file_name'] = uploaded_file.name
        st.session_state['source_video_path'] = tfile.name
        st.session_state['analyzed_data'] = []
        st.session_state['video_meta'] = {}
        st.session_state['is_processed'] = False

    # 狀態 1: 尚未分析
    if not st.session_state['is_processed']:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info(f"已載入: {uploaded_file.name}")
            if st.button("🚀 啟動 AI 全身掃描", type="primary"):
                with st.spinner("AI 正在計算骨架與運動軌跡..."):
                    cap = cv2.VideoCapture(st.session_state['source_video_path'])
                    if not cap.isOpened():
                        st.error("❌ 影片格式錯誤")
                    else:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        st.session_state['video_meta'] = {'width': width, 'height': height, 'fps': fps, 'total_frames': total_frames}
                        temp_landmarks = []
                        bar = st.progress(0)
                        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
                            frame_count = 0
                            while cap.isOpened():
                                ret, frame = cap.read()
                                if not ret: break
                                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                results = pose.process(image)
                                temp_landmarks.append(results.pose_landmarks if results.pose_landmarks else None)
                                frame_count += 1
                                if total_frames > 0: bar.progress(min(frame_count/total_frames, 1.0))
                        cap.release()
                        st.session_state['analyzed_data'] = temp_landmarks
                        st.session_state['is_processed'] = True
                        st.rerun()

    # 狀態 2: 矩陣控制面板 + 殘影繪圖
    else:
        meta = st.session_state['video_meta']
        landmarks_data = st.session_state['analyzed_data']
        col_ctrl, col_video = st.columns([1, 2])
        
        with col_ctrl:
            st.markdown("### 🎯 顯示設定")
            
            show_trail = st.checkbox("💫 顯示殘影軌跡 (Long Exposure)", value=True)
            st.markdown("---")
            
            c_left, c_right = st.columns(2)
            with c_left:
                st.info("**👈 左側 (Left)**")
                l_hip = st.checkbox("左髖 (Hip)", value=False, key="l_hip")
                l_knee = st.checkbox("左膝 (Knee)", value=True, key="l_knee")
                l_ankle = st.checkbox("左踝 (Ankle)", value=False, key="l_ankle")
            with c_right:
                st.warning("**👉 右側 (Right)**")
                r_hip = st.checkbox("右髖 (Hip)", value=False, key="r_hip")
                r_knee = st.checkbox("右膝 (Knee)", value=True, key="r_knee")
                r_ankle = st.checkbox("右踝 (Ankle)", value=False, key="r_ankle")

            metrics_db = {
                "l_hip":   ("L-Hip", mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, (50, 255, 50), mp_pose.PoseLandmark.LEFT_HIP),
                "l_knee":  ("L-Knee", mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, (50, 255, 50), mp_pose.PoseLandmark.LEFT_KNEE),
                "l_ankle": ("L-Ankle", mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX, (50, 255, 50), mp_pose.PoseLandmark.LEFT_ANKLE),
                "r_hip":   ("R-Hip", mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, (255, 255, 50), mp_pose.PoseLandmark.RIGHT_HIP),
                "r_knee":  ("R-Knee", mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE, (255, 255, 50), mp_pose.PoseLandmark.RIGHT_KNEE),
                "r_ankle": ("R-Ankle", mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX, (255, 255, 50), mp_pose.PoseLandmark.RIGHT_ANKLE),
            }

            active_metrics = []
            if l_hip: active_metrics.append(metrics_db["l_hip"])
            if l_knee: active_metrics.append(metrics_db["l_knee"])
            if l_ankle: active_metrics.append(metrics_db["l_ankle"])
            if r_hip: active_metrics.append(metrics_db["r_hip"])
            if r_knee: active_metrics.append(metrics_db["r_knee"])
            if r_ankle: active_metrics.append(metrics_db["r_ankle"])
            
            st.markdown("---")
            
        # --- 影片生成 ---
        output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        # 關鍵修改：將 mp4v 改為 avc1 (H.264)，解決黑畫面問題
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        
        out = cv2.VideoWriter(output_video_path, fourcc, meta['fps'], (meta['width'], meta['height']))
        
        cap = cv2.VideoCapture(st.session_state['source_video_path'])
        w, h = meta['width'], meta['height']
        
        dashboard_positions = {
            "L-Hip": (20, 100), "L-Knee": (20, 150), "L-Ankle": (20, 200),
            "R-Hip": (w - 200, 100), "R-Knee": (w - 200, 150), "R-Ankle": (w - 200, 200)
        }
        
        frame_idx = 0
        TRAIL_LENGTH = 70 # 保持超長軌跡
        STEP = 2          # 保持高密度
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            current_landmarks = landmarks_data[frame_idx] if frame_idx < len(landmarks_data) else None
            
            if current_landmarks:
                mp_drawing.draw_landmarks(
                    frame, current_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                )
                lm = current_landmarks.landmark
                
                cv2.putText(frame, "LEFT", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2, cv2.LINE_AA)
                cv2.putText(frame, "RIGHT", (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 50), 2, cv2.LINE_AA)
                
                for label, idx_a, idx_b, idx_c, color, track_idx in active_metrics:
                    try:
                        p1 = [lm[idx_a.value].x, lm[idx_a.value].y]
                        p2 = [lm[idx_b.value].x, lm[idx_b.value].y]
                        p3 = [lm[idx_c.value].x, lm[idx_c.value].y]
                        angle = calculate_angle(p1, p2, p3)
                        
                        if label in dashboard_positions:
                            pos_x, pos_y = dashboard_positions[label]
                            draw_dashboard(frame, label, angle, pos_x, pos_y, color)
                    except: pass
                    
                    if show_trail:
                        for t in range(STEP, TRAIL_LENGTH, STEP):
                            prev_idx = frame_idx - t
                            if prev_idx >= 0:
                                lm_prev = landmarks_data[prev_idx]
                                if lm_prev:
                                    pt = (int(lm_prev.landmark[track_idx.value].x * w), int(lm_prev.landmark[track_idx.value].y * h))
                                    cv2.circle(frame, pt, 3, color, -1) 

            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        
        with col_video:
            # 顯示影片
            st.video(output_video_path)
            
        with col_ctrl:
            with open(output_video_path, 'rb') as f:
                video_bytes = f.read()
            st.download_button("🎥 下載分析影片", video_bytes, "kuangyu_pro_analysis.mp4", "video/mp4", type="primary")

else:
    st.info("👈 請從左側選單上傳影片。")