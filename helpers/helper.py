import random
import time
import cv2

def generateRandomColors(existing_colors):
    while True:
        color = tuple(random.randint(0, 255) for _ in range(3))
        # Avoid red-ish colors and ensure uniqueness
        if color not in existing_colors and not (color[2] > 150 and color[1] < 100 and color[0] < 100):
            return color
        
def tracking(results, frame, bbox_coords, bbox_colors, track_history, selected_id, red_timers):
    boxes_xyxy = results[0].boxes.xyxy.cpu()
    current_ids = []

    try:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        for track_id, box_xyxy in zip(track_ids, boxes_xyxy):
            x1, y1, x2, y2 = map(int, box_xyxy)
            track = track_history[track_id]
            track.append((float(x1), float(y1)))

            if len(track) > 50:
                track.pop(0)

            current_ids.append(track_id)
            bbox_coords[track_id] = (x1, y1, x2, y2)
            
            if track_id not in bbox_colors:
                bbox_colors[track_id] = generateRandomColors(bbox_colors.values())
            
            color = bbox_colors[track_id]
            
            if selected_id == track_id:
                color = (0, 0, 255)  # Red color
                elapsed_time = int(time.time() - red_timers[track_id])
                cv2.putText(frame, f'Timer: {elapsed_time}s', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    except Exception as e:
        print(f"Error: {e}")
        pass

    return frame