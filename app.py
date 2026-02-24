import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import numpy as np
import sys
import os
import subprocess
import gc
from mediapipe.framework.formats import landmark_pb2

# --- 0. 核心常數設定 ---
# OpenCV 色彩格式為 BGR (藍, 綠, 紅)
DOT_COLOR = (180, 100, 240)        # 關節點點 (紮實粉紫)
LEFT_LINE_COLOR = (255, 255, 0)    # 左側線條 (螢光青藍 Cyan in BGR)
RIGHT_LINE_COLOR = (80, 200, 255)  # 右側線條 (飽和金黃 Yellow in BGR)
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
    
    .mobile-tip {
        font-size: 14px;
        color: #666;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 8px;
        margin-top: 10px;
        text-align: center;
        border: 1px solid #ddd;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. 核心功能函式 ---

def crop_transparent_borders(image):
    if image.shape[2] != 4: return image
    alpha_channel = image[:, :, 3]
    coords = cv2.findNonZero(alpha_channel)
    if coords is None: return image
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]

def create_white_border_sticker(logo_img, border_thickness=5):
    h, w = logo_img.shape[:2]
    pad = border_thickness + 2
    padded_logo = np.zeros((h + pad * 2, w + pad * 2, 4), dtype=np.uint8)
    padded_logo[pad:pad+h, pad:pad+w] = logo_img

    alpha = padded_logo[:, :, 3]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_thickness*2+1, border_thickness*2+1))
    dilated_alpha = cv2.dilate(alpha, kernel)
    
    border_layer = np.zeros_like(padded_logo)
    border_layer[:] = (255, 255, 255, 255)
    border_layer[:, :, 3] = dilated_alpha
    
    final_sticker = border_layer.copy()
    logo_alpha_mask = padded_logo[:, :, 3] / 255.0
    for c in range(0, 3):
        final_sticker[:, :, c] = (1.0 - logo_alpha_mask) * border_layer[:, :, c] + logo_alpha_mask * padded_logo[:, :, c]
    final_sticker[:, :, 3] = cv2.bitwise_or(border_layer[:, :, 3], padded_logo[:, :, 3])
    
    return final_sticker

def add_watermark(frame, logo_path="KUANGYU_logo_v.png"):
    if not os.path.exists(logo_path): return frame
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if logo is None: return frame

    logo = crop_transparent_borders(logo)
    frame_h, frame_w = frame.shape[:2]
    logo_h, logo_w = logo.shape[:2]
    if frame_w > frame_h: scale = 0.05 
    else: scale = 0.10
    
    new_width = int(frame_w * scale)
    new_height = int(logo_h * (new_width / logo_w))
    try: logo_resized = cv2.resize(logo, (new_width, new_height), interpolation=cv2.INTER_AREA)
    except: return frame

    border_px = max(3, min(8, int(new_width * 0.04))) 
    sticker_logo = create_white_border_sticker(logo_resized, border_thickness=border_px)
    sticker_h, sticker_w = sticker_logo.shape[:2]

    margin_right = int(sticker_w * 0.2)
    margin_bottom = int(sticker_w * 0.2)
    x_offset = frame_w - sticker_w - margin_right
    y_offset = frame_h - sticker_h - margin_bottom
    if y_offset < 0: y_offset = 0
    if x_offset < 0: x_offset = 0
    
    alpha = sticker_logo[:, :, 3] / 255.0
    h_part = min(sticker_h, frame_h - y_offset)
    w_part = min(sticker_w, frame_w - x_offset)
    roi = frame[y_offset:y_offset+h_part, x_offset:x_offset+w_part]
    logo_part = sticker_logo[:h_part, :w_part]
    alpha_part = alpha[:h_part, :w_part]
    for c in range(3):
        roi[:, :, c] = (alpha_part * logo_part[:, :, c] + (1.0 - alpha_part) * roi[:, :, c])
    frame[y_offset:y_offset+h_part, x_offset:x_offset+w_part] = roi
    return frame

@st.cache_resource
def load_mediapipe(): return mp.solutions.pose, mp.solutions.drawing_utils
mp_pose, mp_drawing = load_mediapipe()

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180.0 else angle

def draw_dashboard(image, label, angle, x, y, color):
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y - 25), (x + 110, y + 10), (20, 20, 20), -1) 
    cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
    cv2.putText(image, f"{label}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(image, f"{int(angle)}", (x + 60, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.circle(image, (x + 100, y - 5), 2, (255, 255, 255), 1)

# --- Session State ---
if 'analyzed_data' not in st.session_state: st.session_state['analyzed_data'] = [] 
if 'video_meta' not in st.session_state: st.session_state['video_meta'] = {}
if 'source_video_path' not in st.session_state: st.session_state['source_video_path'] = None
if 'current_file_name' not in st.session_state: st.session_state['current_file_name'] = ""
if 'is_processed' not in st.session_state: st.session_state['is_processed'] = False
for part in ['l_hip', 'l_knee', 'l_ankle', 'r_hip', 'r_knee', 'r_ankle']:
    if part not in st.session_state: st.session_state[part] = False 
    if f"t_{part}" not in st.session_state: st.session_state[f"t_{part}"] = False

# --- Main Logic ---
st.markdown("<div class='header-container'><p class='header-title'>光聿KUANGYU｜AI動作實驗室</p><p class='header-subtitle'>Professional Motion Analysis System</p></div>", unsafe_allow_html=True)

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
        gc.collect() 

    if not st.session_state['is_processed']:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("啟動 AI 智能掃描", type="primary"):
                with st.spinner("正在建構 3D 骨架模型 (標準穩定模式)..."):
                    cap = cv2.VideoCapture(st.session_state['source_video_path'])
                    if not cap.isOpened(): st.error("影片格式錯誤")
                    else:
                        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        MAX_WIDTH = 960 
                        if orig_width > MAX_WIDTH:
                            scale_factor = MAX_WIDTH / orig_width
                            new_width = MAX_WIDTH
                            new_height = int(orig_height * scale_factor)
                        else: new_width, new_height = orig_width, orig_height
                            
                        st.session_state['video_meta'] = {'width': new_width, 'height': new_height, 'fps': fps, 'total_frames': total_frames}
                        temp_landmarks_data = []
                        bar = st.progress(0)
                        
                        # [v23 穩定修復] 
                        # 改回 model_complexity=1 以符合雲端伺服器權限
                        # 維持 min_tracking_confidence=0.7 以確保精準度
                        with mp_pose.Pose(
                            min_detection_confidence=0.6, 
                            min_tracking_confidence=0.7, 
                            model_complexity=1, 
                            smooth_landmarks=True
                        ) as pose:
                            frame_count = 0
                            while cap.isOpened():
                                ret, frame = cap.read()
                                if not ret: break
                                if orig_width > MAX_WIDTH: frame = cv2.resize(frame, (new_width, new_height))
                                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                results = pose.process(image)
                                
                                if results.pose_landmarks:
                                    lm_list = [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
                                    temp_landmarks_data.append(lm_list)
                                else:
                                    temp_landmarks_data.append(None)
                                    
                                frame_count += 1
                                if total_frames > 0: bar.progress(min(frame_count/total_frames, 1.0))
                                
                                if frame_count % 30 == 0:
                                    gc.collect()
                                    
                        cap.release()
                        st.session_state['analyzed_data'] = temp_landmarks_data
                        st.session_state['is_processed'] = True
                        gc.collect()
                        st.rerun()

    else:
        st.markdown("<hr style='margin: 20px 0; border: 0; border-top: 1px solid #eee;'>", unsafe_allow_html=True)
        meta = st.session_state['video_meta']
        landmarks_data = st.session_state['analyzed_data']
        col_video, col_ctrl = st.columns([2, 1])
        
        with col_ctrl:
            with st.form("control_panel"):
                st.markdown("<span class='panel-header'>顯示設定</span>", unsafe_allow_html=True)
                trail_mode = st.radio("軌跡風格", ["無限疊加 (連續線條)", "漸淡軌跡 (彗星尾巴)"], index=0)
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown("<span class='panel-header' style='border-color: #00FFFF;'>左側數據 (Left - Cyan)</span>", unsafe_allow_html=True)
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
                st.markdown("<span class='panel-header' style='border-color: #FFC850;'>右側數據 (Right - Yellow)</span>", unsafe_allow_html=True)
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

            metrics_db = {
                "l_hip":   ("L-Hip", mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, LEFT_LINE_COLOR, mp_pose.PoseLandmark.LEFT_HIP),
                "l_knee":  ("L-Knee", mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, LEFT_LINE_COLOR, mp_pose.PoseLandmark.LEFT_KNEE),
                "l_ankle": ("L-Ankle", mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX, LEFT_LINE_COLOR, mp_pose.PoseLandmark.LEFT_ANKLE),
                "r_hip":   ("R-Hip", mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, RIGHT_LINE_COLOR, mp_pose.PoseLandmark.RIGHT_HIP),
                "r_knee":  ("R-Knee", mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE, RIGHT_LINE_COLOR, mp_pose.PoseLandmark.RIGHT_KNEE),
                "r_ankle": ("R-Ankle", mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX, RIGHT_LINE_COLOR, mp_pose.PoseLandmark.RIGHT_ANKLE),
            }
            active_metrics = [metrics_db[k] + (st.session_state.get(f"t_{k}", False),) for k in metrics_db if st.session_state.get(k, False)]

        with col_video:
            tfile_output_avi = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
            output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            
            out = cv2.VideoWriter(tfile_output_avi, cv2.VideoWriter_fourcc(*'MJPG'), meta['fps'], (meta['width'], meta['height']))
            cap = cv2.VideoCapture(st.session_state['source_video_path'])
            
            dashboard_positions = {
                "L-Hip": (20, 100), "L-Knee": (20, 145), "L-Ankle": (20, 190),
                "R-Hip": (meta['width'] - 135, 100), "R-Knee": (meta['width'] - 135, 145), "R-Ankle": (meta['width'] - 135, 190)
            }
            path_storage = {} 
            frame_idx = 0
            status_text = st.empty()
            progress_bar = st.progress(0)
            IS_FADE_MODE = "漸淡" in trail_mode
            MAX_TRAIL_LENGTH = 100 if IS_FADE_MODE else None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                curr_h, curr_w = frame.shape[:2]
                target_w, target_h = meta['width'], meta['height']
                if curr_w != target_w or curr_h != target_h:
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

                raw_lm_data = landmarks_data[frame_idx] if frame_idx < len(landmarks_data) else None
                current_landmarks = None
                
                if raw_lm_data:
                    current_landmarks = landmark_pb2.NormalizedLandmarkList()
                    for lm_vals in raw_lm_data:
                        current_landmarks.landmark.add(x=lm_vals[0], y=lm_vals[1], z=lm_vals[2], visibility=lm_vals[3])

                if current_landmarks:
                    mp_drawing.draw_landmarks(frame, current_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=DOT_COLOR, thickness=DOT_RADIUS, circle_radius=DOT_RADIUS),
                        mp_drawing.DrawingSpec(color=SKELETON_COLOR, thickness=LINE_THICKNESS, circle_radius=2))
                    lm = current_landmarks.landmark
                    
                    cv2.putText(frame, "LEFT SIDE", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, LEFT_LINE_COLOR, 2, cv2.LINE_AA)
                    cv2.putText(frame, "RIGHT SIDE", (meta['width'] - 135, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RIGHT_LINE_COLOR, 2, cv2.LINE_AA)
                    
                    for label, idx_a, idx_b, idx_c, color, track_idx, show_trail_flag in active_metrics:
                        try:
                            angle = calculate_angle([lm[idx_a.value].x, lm[idx_a.value].y], [lm[idx_b.value].x, lm[idx_b.value].y], [lm[idx_c.value].x, lm[idx_c.value].y])
                            if label in dashboard_positions: draw_dashboard(frame, label, angle, *dashboard_positions[label], color)
                        except: pass
                        
                        if show_trail_flag:
                            px, py = int(lm[track_idx.value].x * meta['width']), int(lm[track_idx.value].y * meta['height'])
                            if label not in path_storage: path_storage[label] = []
                            path_storage[label].append((px, py))
                            points_list = path_storage[label]
                            
                            if IS_FADE_MODE:
                                points_list = points_list[-MAX_TRAIL_LENGTH:]
                                path_storage[label] = points_list
                                for i in range(1, len(points_list)):
                                    alpha = (i / len(points_list)) * 0.8 
                                    overlay = frame.copy()
                                    cv2.line(overlay, points_list[i-1], points_list[i], color, LINE_THICKNESS, cv2.LINE_AA)
                                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                            else:
                                if len(points_list) > 1:
                                    cv2.polylines(frame, [np.array(points_list, np.int32).reshape((-1, 1, 2))], False, color, LINE_THICKNESS, cv2.LINE_AA)

                frame = add_watermark(frame, logo_path="KUANGYU_logo_v.png")

                out.write(frame)
                frame_idx += 1
                if meta['total_frames'] > 0 and frame_idx % 5 == 0:
                    progress_bar.progress(min(frame_idx / meta['total_frames'], 1.0))
                    status_text.text(f"AI 繪圖運算中: {int(frame_idx/meta['total_frames']*100)}%")
                
                if frame_idx % 30 == 0:
                    gc.collect()
            
            cap.release()
            out.release()
            progress_bar.empty()
            status_text.text("最終壓縮轉檔中...")
            
            try:
                subprocess.run(['ffmpeg', '-y', '-i', tfile_output_avi, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'ultrafast', '-crf', '25', output_video_path], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                st.error(f"影片轉檔失敗 (FFmpeg Error): {e.stderr}")
            
            status_text.empty()
            st.success("分析完成")
            st.video(output_video_path)
            st.markdown("<div class='mobile-tip'>手機版若無法下載，請點擊上方影片播放器右下角「⋮」或長按按鈕選擇「分享 / 下載」</div>", unsafe_allow_html=True)
            with open(output_video_path, 'rb') as f: st.download_button("下載影片", f.read(), "kuangyu_analysis.mp4", "video/mp4", type="primary", use_container_width=True)
