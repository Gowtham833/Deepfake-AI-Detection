import cv2
import os
import logging
from src.utils.io import ensure_dir

logger = logging.getLogger(__name__)

def extract_frames_from_video(video_path, output_dir, fps=None, max_frames=None, quality=95):
    ensure_dir(output_dir)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video {video_path}")
        return {'status': 'error', 'reason': 'cannot open video'}
        
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_interval = 1
    if fps and orig_fps:
        frame_interval = max(1, int(orig_fps / fps))
        
    extracted_count = 0
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            out_path = os.path.join(output_dir, f"{video_name}_frame_{extracted_count:04d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            extracted_count += 1
            
            if max_frames and extracted_count >= max_frames:
                break
                
        frame_idx += 1
        
    cap.release()
    return {
        'status': 'success', 
        'extracted_count': extracted_count, 
        'source_video': video_name,
        'original_fps': orig_fps,
        'total_frames': total_frames
    }

def extract_frames_from_directory(video_dir, output_dir, config):
    results = []
    fps = config.get('extraction_fps', 5)
    max_frames = config.get('max_frames_per_video', 50)
    
    for root, _, files in os.walk(video_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, video_dir)
                vid_out_dir = os.path.join(output_dir, rel_path)
                
                res = extract_frames_from_video(video_path, vid_out_dir, fps=fps, max_frames=max_frames)
                results.append(res)
                
    return results
