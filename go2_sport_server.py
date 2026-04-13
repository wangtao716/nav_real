import time
import sys
import socket
import math
import threading
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

# 导入 ROS 2 和图像处理相关的库
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

# --- 1. 将 ROS 2 节点和服务器逻辑封装到一个类中 ---
class RobotControlServer(Node):
    def __init__(self, network_interface: str):
        # 初始化 ROS 2 节点
        super().__init__('robot_control_server_node')
        
        # --- 共享状态标志 ---
        self.is_path_clear = True  # True 代表路径清晰，False 代表有障碍物
        self.last_check_time = time.time()
        self.lock = threading.Lock() # 用于线程安全地访问标志

        # --- ROS 2 相关初始化 ---
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/depth/image_rect_raw',  # 订阅深度图像话题
            self.depth_image_callback,
            10)
        self.get_logger().info('Obstacle detector integrated and started.')

        # --- Unitree SDK 相关初始化 ---
        ChannelFactoryInitialize(0, network_interface)
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        self.get_logger().info('Unitree SportClient initialized.')

    def depth_image_callback(self, msg):
        """
        每当收到深度图像时调用此回调函数。
        为了避免过于频繁的计算，我们限制为大约每秒检查一次。
        """
        current_time = time.time()
        if current_time - self.last_check_time < 1.0:
            return # 距离上次检查不足1秒，直接返回

        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')
            return

        # --- 障碍物检测逻辑 ---
        OBSTACLE_DISTANCE_THRESHOLD_METERS = 0.2 # 安全距离阈值
        threshold_mm = OBSTACLE_DISTANCE_THRESHOLD_METERS * 1000

        h, w = depth_image.shape
        roi = depth_image[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
        roi_valid_depths = roi[roi > 0]
        
        path_is_currently_clear = True
        if roi_valid_depths.size > 0:
            min_depth_in_roi = np.min(roi_valid_depths)
            if min_depth_in_roi < threshold_mm:
                path_is_currently_clear = False
                self.get_logger().warn(f'OBSTACLE DETECTED! Min distance: {min_depth_in_roi / 1000.0:.2f}m', throttle_duration_sec=2)
        
        # --- 线程安全地更新标志 ---
        with self.lock:
            self.is_path_clear = path_is_currently_clear
        
        self.last_check_time = current_time

    def move_continuously(self, vx: float, vy: float, vyaw: float, duration: float):
        """在指定的时间内，高频发送移动指令以维持运动。"""
        print(f"Executing move command for {duration:.2f} seconds...")
        start_time = time.time()
        while time.time() - start_time < duration:
            with self.lock:
                if not self.is_path_clear:
                    print("Execution aborted: Path is blocked.")
                    return False
                    break # 检测到障碍物，立即中断移动循环
            
            self.sport_client.Move(vx, vy, vyaw)
            time.sleep(0.05)
        
        self.sport_client.StopMove()
        print("Move finished.")
        return True
    
    def move_continuously1(self, vx: float, vy: float, vyaw: float, duration: float):
        """在指定的时间内，高频发送移动指令以维持运动。"""
        print(f"Executing move command for {duration:.2f} seconds...")
        start_time = time.time()
        while time.time() - start_time < duration:            
            self.sport_client.Move(vx, vy, vyaw)
            time.sleep(0.05)
        
        self.sport_client.StopMove()
        print("Move finished.")

    def parse_and_execute_command(self, command: str):
        """解析命令字符串并调用相应的 sport_client 方法"""
        cmd = command.lower().strip()
        print(f"Received command: '{cmd}'")
        LINEAR_SPEED = 0.25
        ANGULAR_SPEED = 0.5

        if "forward" in cmd:
            print("Executing: Move forward")
            # --- 2. 直接检查内部标志，而不是调用外部脚本 ---
            with self.lock:
                if not self.is_path_clear:
                    error_msg = "Execution failed: Path is blocked by an obstacle."
                    print(error_msg)
                    return error_msg
            
            try:
                parts = cmd.split()
                distance = float(parts[2])
                MOVE_DURATION = distance / LINEAR_SPEED
                flag = self.move_continuously(0.25, 0, 0, MOVE_DURATION)
                if not flag:
                    return "Execution failed: Path is blocked by an obstacle."
                return "Command 'move forward' executed successfully."
            except (IndexError, ValueError):
                return "Invalid 'forward' command. Usage: forward <distance_in_meters>"


        elif "scan the area" in cmd:
            print("Executing: Scan the area")
            self.sport_client.StopMove()
            return "Command 'scan the area' executed successfully."
        
        elif "stop" in cmd:
            print("Executing: Stop")
            self.sport_client.StopMove()
            return "Command 'stop' executed successfully."
            
        elif "left" in cmd and "turn" in cmd:
            print("Executing: Turn left")
            try:
                parts = cmd.split()
                angle_degrees = float(parts[2])
                MOVE_DURATION = abs(math.radians(angle_degrees)) / ANGULAR_SPEED
                self.move_continuously1(0, 0, 0.5, MOVE_DURATION)
                return f"Command 'turn left' executed successfully."
            except (IndexError, ValueError):
                return "Invalid 'turn left' command. Usage: turn left <degrees>"
            
        elif "right" in cmd and "turn" in cmd:
            print("Executing: Turn right")
            try:
                parts = cmd.split()
                angle_degrees = float(parts[2])
                MOVE_DURATION = abs(math.radians(angle_degrees)) / ANGULAR_SPEED
                self.move_continuously1(0, 0, -0.5, MOVE_DURATION)
                return f"Command 'turn right' executed successfully."
            except (IndexError, ValueError):
                return "Invalid 'turn right' command. Usage: turn right <degrees>"
            
        else:
            msg = f"Unknown command: '{cmd}'. Doing nothing."
            print(msg)
            self.sport_client.StopMove()
            return msg

    def run_socket_server(self):
        """设置并运行 Socket 服务器监听循环。"""
        HOST = '0.0.0.0'
        PORT = 12345
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen()
            self.get_logger().info(f"Robot server listening on {HOST}:{PORT}")
            
            while rclpy.ok():
                try:
                    conn, addr = s.accept()
                    with conn:
                        print(f"Connected by {addr}")
                        while rclpy.ok():
                            data = conn.recv(1024)
                            if not data:
                                break
                            command_str = data.decode('utf-8')
                            execution_result = self.parse_and_execute_command(command_str)
                            conn.sendall(execution_result.encode('utf-8'))
                        print(f"Connection with {addr} closed.")
                except Exception as e:
                    self.get_logger().error(f"Socket server error: {e}")
                    break

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} networkInterface")
        sys.exit(-1)

    # --- 3. 主函数中初始化所有组件并使用多线程 ---
    rclpy.init(args=None)
    
    # 创建我们的集成化服务器/节点实例
    server_node = RobotControlServer(network_interface=sys.argv[1])
    
    # 创建并启动一个新线程来运行 ROS 2 的 spin()，这样它就不会阻塞主线程
    ros_thread = threading.Thread(target=rclpy.spin, args=(server_node,), daemon=True)
    ros_thread.start()
    
    try:
        # 主线程负责运行 Socket 服务器
        server_node.run_socket_server()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # 清理
        server_node.destroy_node()
        rclpy.shutdown()
        ros_thread.join() # 等待 ROS 线程结束
        print("Shutdown complete.")

if __name__ == "__main__":
    main()