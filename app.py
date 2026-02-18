import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import numpy as np
import sys
import os
import subprocess

# --- 0. 核心常數設定 (視覺微調版 v17.3) ---
# OpenCV 色彩格式為 BGR.
DOT_COLOR = (180, 100, 240)        # 關節點點 (紮實粉紫)
LEFT_LINE_COLOR = (220, 110, 50)   # 左側線條 (深邃藍)
RIGHT_LINE_COLOR = (80, 200, 255)  # 右側線條 (飽和金黃)
SKELETON_COLOR = (255, 255, 255)   # 骨架連線 (純白)

# [尺寸鎖定]
LINE_THICKNESS = 2    # 線條粗細
DOT_RADIUS = 2        # 點點半徑

# --- 1. 介面設定 ---
st.set_page_config(
    page_title="光聿KUANGYU - AI 動作實驗室",
    page_icon=None, 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CSS 極致美化 ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    .stApp {
        background-color: #F5F7F9;
        font-family: 'Inter', sans-serif;
    }

    .header-container {
        background-color: #1E1E1E;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .header-title {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 1px;
        margin: 0;
    }
    .header-subtitle {
        font-size: 0.9rem;
        color: #A0A0A0;
        margin-top: 5px;
        font-weight: 400;
    }

    .stForm {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #E0E0E0;
    }
    
    div[data-testid="stFileUploader"] {
        background-color: #FFFFFF;
        border: 1px dashed #E0E0E0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease;
        padding: 0.6rem 1.2rem;
        font-size: 16px;
    }
    
    button[kind="primaryFormSubmit"] {
        background: linear-gradient(135deg, #FF4B4B 0%, #D42F2F 100%);
        color: white;
        box-shadow: 0 4px 6px rgba(255, 75, 75, 0.2);
        width: 100%;
        border: none;
    }
    button[kind="primaryFormSubmit"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(255, 75, 75, 0.3);
    }
    
    div.stButton > button.st-emotion-cache-19rxjzo {
        background: linear-gradient(135deg, #FF4B4B 0%, #D42F2F 100%);
        color: white;
        width: 100%;
    }

    .stCheckbox label {
        font-size: 15px !important;
        font-weight: 600;
        color: #333;
    }

    .stRadio label {
        font-weight: 600;
        color: #333;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stVideo {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        width: 100%;
    }
    
    .panel-header {
        font-size: 16px;
        font-weight: bold;
        color: #333;
        margin-bottom: 15px;
        border-left: 4px solid #FF4B4B;
        padding-left: 10px;
        display: block;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. 核心功能函式 ---

# [補回] 自動裁切透明邊框的小工具
def crop_transparent_borders(image):
    if image.shape[2] != 4: return image # 如果沒有透明通道就不裁
    
    # 找出所有不透明的像素 (Alpha > 0)
    alpha_channel = image[:, :, 3]
    coords = cv2.findNonZero(alpha_channel)
    if coords is None: return image # 全透明就不裁
    
    # 取得最小外接矩形
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]

# [補回 v7] 浮水印小幫手：自動裁切 + 90%定位
def add_watermark(frame, logo_path="KUANGYU_logo_v.png", scale=0.10, bg_padding=0):
    """
    1. 自動裁切 Logo 透明邊，讓圓圈緊貼。
    2. 定位到畫面右下角 (90%, 90%) 位置。
    """
    if not os.path.exists(logo_path): return frame
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if logo is None: return frame

    # --- 步驟 1: 自動裁切多餘白邊 ---
    logo = crop_transparent_borders(logo)

    frame_h, frame_w = frame.shape[:2]
    logo_h, logo_w = logo.shape[:2]

    # --- 步驟 2: 計算尺寸 (稍微放大到 10% 讓它清楚一點) ---
    new_width = int(frame_w * scale)
    new_height = int(logo_h * (new_width / logo_w))
    
    try:
        logo = cv2.resize(logo, (new_width, new_height), interpolation=cv2.INTER_AREA)
    except: return frame

    # --- 步驟 3: 右下角精準定位 (90% x 90%) ---
    target_center_x = int(frame_w * 0.90)
    target_center_y = int(frame_h * 0.90)

    x_offset = target_center_x - (new_width // 2)
    y_offset = target_center_y - (new_height // 2)

    # 邊界保護
    if y_offset < 0: y_offset = 0
    if x_offset < 0: x_offset = 0
    if y_offset + new_height > frame_h: new_height = frame_h - y_offset
    if x_offset + new_width > frame_w: new_width = frame_w - x_offset
    if new_width <= 0 or new_height <= 0: return frame
    
    logo = logo[:new_height, :new_width]

    # --- 步驟 4: 繪製極致緊貼的白圈 ---
    center_x = x_offset + new_width // 2
    center_y = y_offset + new_height // 2
    
    # 計算對角線半徑，bg_padding 設為 0
    diagonal = np.sqrt(new_width**2 + new_height**2)
    radius = int(diagonal / 2) + bg_padding

    cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), -1)

    # --- 步驟 5: 疊加 Logo ---
    if logo.shape[2] == 4:
        alpha = logo[:, :, 3] / 255.0
        roi = frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width]
        for c in range(3):
            roi[:, :, c] = (alpha * logo[:, :, c] + (1.0 - alpha) * roi[:, :, c])
        frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = roi
    else:
        frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = logo

    return frame

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
    
    # 迷你儀表板背景
    box_w = 110
    box_h = 35
    cv2.rectangle(overlay, (x, y - 25), (x + box_w, y + 10), (20, 20, 20), -1) 
    
    alpha = 0.8
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # 標籤
    cv2.putText(image, f"{label}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    
    # 數字
    angle_text = f"{int(angle)}"
    cv2.putText(image, angle_text, (x + 60, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    
    # 度數圈圈 (白色)
    cv2.circle(image, (x + 100, y - 5), 2, (255, 255, 255), 1)

# --- Session State ---
if 'analyzed_data' not in st.session_state: st.session_state['analyzed_data'] = [] 
if 'video_meta' not in st.session_state: st.session_state['video_meta'] = {}
if 'source_video_path' not in st.session_state: st.session_state['source_video_path'] = None
if 'current_file_name' not in st.session_state: st.session_state['current_file_name'] = ""
if 'is_processed' not in st.session_state: st.session_state['is_processed'] = False

# --- 初始化部位 ---
body_parts = ['l_hip', 'l_knee', 'l_ankle', 'r_hip', 'r_knee', 'r_ankle']
for part in body_parts:
    if part not in st.session_state: st.session_state[part] = False 
    if f"t_{part}" not in st.session_state: st.session_state[f"t_{part}"] = False

# --- Main Layout ---

st.markdown("""
    <div class="header-container">
        <p class="header-title">光聿KUANGYU｜AI動作實驗室</p>
        <p class="header-subtitle">Professional Motion Analysis System</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("上傳訓練影片 (MP4/MOV)", type=["mp4", "mov"])

if uploaded_file:
    if uploaded_file.name != st.session_state['current_file_name']:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()
        st.session_state['current_file_name'] = uploaded_file.name
        st.session_state['source_video_path'] = tfile.name
        st.session_state['is_processed'] = False
        st.session_state['analyzed_data'] = []

    # 狀態 A: 尚未分析
    if not st.session_state['is_processed']:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("啟動 AI 智能掃描", type="primary"):
                with st.spinner("正在建構 3D 骨架模型..."):
                    cap = cv2.VideoCapture(st.session_state['source_video_path'])
                    if not cap.isOpened():
                        st.error("影片格式錯誤")
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

    # 狀態 B: 分析完成
    else:
        st.markdown("<hr style='margin: 20px 0; border: 0; border-top: 1px solid #eee;'>", unsafe_allow_html=True)
        
        meta = st.session_state['video_meta']
        landmarks_data = st.session_state['analyzed_data']
        
        col_video, col_ctrl = st.columns([2, 1])
        
        # --- Form: 控制面板 ---
        with col_ctrl:
            with st.form("control_panel"):
                st.markdown("<span class='panel-header'>顯示設定</span>", unsafe_allow_html=True)
                
                trail_mode = st.radio(
                    "軌跡風格",
                    ["無限疊加 (連續線條)", "漸淡軌跡 (彗星尾巴)"],
                    index=0
                )
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # 左側控制項
                st.markdown("<span class='panel-header' style='border-color: rgb{LEFT_LINE_COLOR};'>左側數據 (Left - Blue)</span>", unsafe_allow_html=True)
                l_c1, l_c2 = st.columns(2)
                with l_c1:
                    l_hip = st.checkbox("左髖", value=st.session_state.get('l_hip', False))
                    l_knee = st.checkbox("左膝", value=st.session_state.get('l_knee', True))
                    l_ankle = st.checkbox("左踝", value=st.session_state.get('l_ankle', False))
                with l_c2:
                    t_l_hip = st.checkbox("軌跡", value=st.session_state.get('t_l_hip', False), key="t_l_hip_f")
                    t_l_knee = st.checkbox("軌跡", value=st.session_state.get('t_l_knee', True), key="t_l_knee_f")
                    t_l_ankle = st.checkbox("軌跡", value=st.session_state.get('t_l_ankle', False), key="t_l_ankle_f")
                
                st.markdown("<br>", unsafe_allow_html=True)

                # 右側控制項
                st.markdown("<span class='panel-header' style='border-color: rgb{RIGHT_LINE_COLOR};'>右側數據 (Right - Yellow)</span>", unsafe_allow_html=True)
                r_c1, r_c2 = st.columns(2)
                with r_c1:
                    r_hip = st.checkbox("右髖", value=st.session_state.get('r_hip', False))
                    r_knee = st.checkbox("右膝", value=st.session_state.get('r_knee', True))
                    r_ankle = st.checkbox("右踝", value=st.session_state.get('r_ankle', False))
                with r_c2:
                    t_r_hip = st.checkbox("軌跡", value=st.session_state.get('t_r_hip', False), key="t_r_hip_f")
                    t_r_knee = st.checkbox("軌跡", value=st.session_state.get('t_r_knee', True), key="t_r_knee_f")
                    t_r_ankle = st.checkbox("軌跡", value=st.session_state.get('t_r_ankle', False), key="t_r_ankle_f")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                submitted = st.form_submit_button("確認並生成影片", type="primary")

                if submitted:
                    st.session_state.update({
                        'l_hip': l_hip, 'l_knee': l_knee, 'l_ankle': l_ankle,
                        't_l_hip': t_l_hip, 't_l_knee': t_l_knee, 't_l_ankle': t_l_ankle,
                        'r_hip': r_hip, 'r_knee': r_knee, 'r_ankle': r_ankle,
                        't_r_hip': t_r_hip, 't_r_knee': t_r_knee, 't_r_ankle': t_r_ankle
                    })

            # 資料庫：使用調色後的品牌常數
            metrics_db = {
                "l_hip":   ("L-Hip", mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, LEFT_LINE_COLOR, mp_pose.PoseLandmark.LEFT_HIP),
                "l_knee":  ("L-Knee", mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, LEFT_LINE_COLOR, mp_pose.PoseLandmark.LEFT_KNEE),
                "l_ankle": ("L-Ankle", mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX, LEFT_LINE_COLOR, mp_pose.PoseLandmark.LEFT_ANKLE),
                "r_hip":   ("R-Hip", mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, RIGHT_LINE_COLOR, mp_pose.PoseLandmark.RIGHT_HIP),
                "r_knee":  ("R-Knee", mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE, RIGHT_LINE_COLOR, mp_pose.PoseLandmark.RIGHT_KNEE),
                "r_ankle": ("R-Ankle", mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX, RIGHT_LINE_COLOR, mp_pose.PoseLandmark.RIGHT_ANKLE),
            }

            active_metrics = []
            keys = ['l_hip', 'l_knee', 'l_ankle', 'r_hip', 'r_knee', 'r_ankle']
            for k in keys:
                if st.session_state.get(k, False):
                    active_metrics.append(metrics_db[k] + (st.session_state.get(f"t_{k}", False),))

        with col_video:
            tfile_output_avi = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
            output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            
            fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
            out = cv2.VideoWriter(tfile_output_avi, fourcc, meta['fps'], (meta['width'], meta['height']))
            cap = cv2.VideoCapture(st.session_state['source_video_path'])
            
            w, h = meta['width'], meta['height']
            total_frames = meta['total_frames']
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            
            dashboard_positions = {
                "L-Hip": (20, 100), "L-Knee": (20, 160), "L-Ankle": (20, 220),
                "R-Hip": (w - 200, 100), "R-Knee": (w - 200, 160), "R-Ankle": (w - 200, 220)
            }
            
            path_storage = {} 
            frame_idx = 0
            status_text = st.empty()
            progress_bar = st.progress(0)

            if "無限疊加" in trail_mode:
                IS_FADE_MODE = False
                MAX_TRAIL_LENGTH = None 
            else:
                IS_FADE_MODE = True
                MAX_TRAIL_LENGTH = 100 # 設定軌跡長度

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if orig_w > 960: frame = cv2.resize(frame, (w, h))

                current_landmarks = landmarks_data[frame_idx] if frame_idx < len(landmarks_data) else None
                
                if current_landmarks:
                    # [視覺修正] 點點: 紮實粉/2px, 線條: 純白/2px
                    mp_drawing.draw_landmarks(
                        frame, current_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=DOT_COLOR, thickness=DOT_RADIUS, circle_radius=DOT_RADIUS),
                        mp_drawing.DrawingSpec(color=SKELETON_COLOR, thickness=LINE_THICKNESS, circle_radius=2)
                    )
                    lm = current_landmarks.landmark
                    
                    cv2.putText(frame, "LEFT SIDE", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, LEFT_LINE_COLOR, 2, cv2.LINE_AA)
                    cv2.putText(frame, "RIGHT SIDE", (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RIGHT_LINE_COLOR, 2, cv2.LINE_AA)
                    
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
                            px = int(lm[track_idx.value].x * w)
                            py = int(lm[track_idx.value].y * h)
                            
                            if label not in path_storage: path_storage[label] = []
                            path_storage[label].append((px, py))
                            
                            points_list = path_storage[label]
                            
                            if IS_FADE_MODE:
                                # 彗星模式：真正透明漸層 (addWeighted)
                                points_list = points_list[-MAX_TRAIL_LENGTH:]
                                path_storage[label] = points_list
                                
                                for i in range(1, len(points_list)):
                                    intensity = i / len(points_list)
                                    alpha = intensity * 0.8 # 透明度係數
                                    
                                    # 建立單層 Overlay 避免重複疊加造成過慢，這裡做單次線段疊加
                                    overlay = frame.copy()
                                    cv2.line(overlay, points_list[i-1], points_list[i], color, LINE_THICKNESS, cv2.LINE_AA)
                                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                            
                            else:
                                # 無限模式 (實線)
                                if len(points_list) > 1:
                                    pts = np.array(points_list, np.int32)
                                    pts = pts.reshape((-1, 1, 2))
                                    cv2.polylines(frame, [pts], False, color, LINE_THICKNESS, cv2.LINE_AA)

                # [補回] 呼叫自動裁切 + 精準定位浮水印 (重要！)
                frame = add_watermark(frame, logo_path="KUANGYU_logo_v.png", scale=0.10, bg_padding=0)

                out.write(frame)
                frame_idx += 1
                if total_frames > 0 and frame_idx % 5 == 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))
                    status_text.text(f"AI 繪圖運算中: {int(frame_idx/total_frames*100)}%")
            
            cap.release()
            out.release()
            
            progress_bar.empty()
            status_text.text("最終壓縮轉檔中...")
            
            with st.spinner("製作最終影片 (H.264)..."):
                subprocess.call([
                    'ffmpeg', '-y', '-i', tfile_output_avi, 
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '25',
                    output_video_path
                ])
            status_text.empty()
            
            st.success("分析完成")
            st.video(output_video_path)
            with open(output_video_path, 'rb') as f:
                video_bytes = f.read()
            st.download_button("下載影片", video_bytes, "kuangyu_analysis.mp4", "video/mp4", type="primary", use_container_width=True)