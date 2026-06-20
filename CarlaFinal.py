import carla
import cv2
import numpy as np
import time
import random

def process_camera_image(image, vehicle, lks_state):
    raw_image = np.array(image.raw_data)
    reshaped_image = raw_image.reshape((image.height, image.width, 4))
    rgb_image = reshaped_image[:, :, :3].copy()
    
    # --- ה-IPM שלנו ללא המשולשים השחורים ---
    src_points = np.float32([
        [0,   520],  
        [345, 320],  
        [455, 320],  
        [800, 520]   
    ])
    
    dst_points = np.float32([
        [0,   600],  
        [0,   0],    
        [800, 0],    
        [800, 600]   
    ])
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    bird_eye_view = cv2.warpPerspective(rgb_image, matrix, (800, 600))
    
    gray = cv2.cvtColor(bird_eye_view, cv2.COLOR_BGR2GRAY)
    _, binary_lanes = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # חישוב הסטייה
    moments = cv2.moments(binary_lanes)
    deviation = 0
    if moments["m00"] > 0: 
        lane_center_x = int(moments["m10"] / moments["m00"])
        car_center_x = 400
        deviation = lane_center_x - car_center_x
        
        cv2.line(bird_eye_view, (car_center_x, 600), (car_center_x, 500), (0, 255, 0), 2)
        cv2.line(bird_eye_view, (lane_center_x, 600), (lane_center_x, 500), (255, 0, 0), 2)
    
    # --- מערכת LKS משולבת בקרת מהירות (Braking System) ---
    
    if not lks_state['is_active'] and abs(deviation) > 160:
        lks_state['is_active'] = True
        lks_state['frames'] = 10  
        lks_state['nudge_steer'] = max(min(deviation * 0.001, 0.15), -0.15)
        
    if lks_state['is_active']:
        intervention_control = carla.VehicleControl()
        
        # אנחנו לוקחים פיקוד על המהירות - קודם כל עוזבים את הגז!
        intervention_control.throttle = 0.0 
        
        # שלב א': בלימה ודחיפה חזרה לנתיב
        if lks_state['frames'] > 3:
            cv2.putText(bird_eye_view, "LKS: BRAKING & NUDGING!", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(rgb_image, (0,0), (800,600), (0, 0, 255), 3)
            
            intervention_control.steer = lks_state['nudge_steer']
            intervention_control.brake = 0.4 # הפעלת בלמים בעוצמה של 40% כדי להאט בבטחה
        
        # שלב ב': יישור ההגה ושחרור הבלמים (הכנה לחזרה לאוטופיילוט)
        else:
            cv2.putText(bird_eye_view, "LKS: STRAIGHTENING...", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.rectangle(rgb_image, (0,0), (800,600), (0, 165, 255), 3)
            
            intervention_control.steer = -lks_state['nudge_steer'] * 0.5 
            intervention_control.brake = 0.0 # עוזבים את הברקס כדי שהרכב יתייצב
            
        vehicle.apply_control(intervention_control)
        lks_state['frames'] -= 1
        
        if lks_state['frames'] <= 0:
            lks_state['is_active'] = False 
            
    else:
        cv2.putText(bird_eye_view, "LKS: Monitoring Autopilot", (50, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ציור ותצוגה
    pts = np.int32(src_points).reshape((-1, 1, 2))
    cv2.polylines(rgb_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    
    cv2.imshow("1. Driver View (LKS Active)", rgb_image)
    cv2.imshow("2. BEV & LKS Intervention", bird_eye_view)
    cv2.waitKey(1)


def main():
    print("מתחבר ל-CARLA...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    world = client.load_world('Town04')
    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("רכב נוצר בהצלחה!")
    
    vehicle.set_autopilot(True) 
    tm = client.get_trafficmanager(8000)
    
    tm.auto_lane_change(vehicle, True)
    # הרכב ייסע די מהר (רק 5% מתחת למהירות המותרת) כדי שיהיה קשה בסיבובים
    tm.global_percentage_speed_difference(5.0) 

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '100')

    camera_transform = carla.Transform(carla.Location(x=-0.5, z=2.8), carla.Rotation(pitch=-8.0))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    lks_state = {'is_active': False, 'frames': 0, 'nudge_steer': 0.0}
    camera.listen(lambda image: process_camera_image(image, vehicle, lks_state))

    print("מערכת LKS חכמה (כולל בקרת בלימה) רצה! לחץ Ctrl+C לעצירה.")
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nמנקה...")
    finally:
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()
        print("להתראות!")

if __name__ == '__main__':
    main()