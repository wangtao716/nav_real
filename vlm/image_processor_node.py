import rclpy
from rclpy.node import Node
from rclpy.task import Future
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor_node')
        
        # 创建一个 Future 对象，用作完成信号
        self.future = Future()

        # 1. 创建订阅者
        # 订阅名为 /image_raw 的话题，消息类型为 sensor_msgs.msg.Image
        # 当有新消息到达时，调用 self.image_callback 方法
        self.subscription = self.create_subscription(
            CompressedImage,  # 使用 CompressedImage 类型以处理压缩图像
            #'/camera/camera/color/image_raw',  # 这是您要订阅的话题名称
            '/camera/color/image_raw/compressed',  # 这是您要订阅的话题名称
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning

        # 2. 创建发布者 (可选，用于发布处理后的图像)
        # 将处理后的图像发布到 /image_processed 话题
        # self.publisher_ = self.create_publisher(Image, '/image_processed', 10)

        # 3. 初始化 CvBridge
        # CvBridge 用于在 ROS Image 消息和 OpenCV 图像之间进行转换
        self.bridge = CvBridge()

        self.get_logger().info('Image processor node started, waiting for one image...')

    def image_callback(self, msg):
        """
        这是每次接收到图像消息时都会被调用的回调函数。
        """
        # 如果 Future 已经完成（即已经处理过一张图片），则直接返回
        if self.future.done():
            return

        self.get_logger().info('Receiving raw image...')

        try:
            # 4. 将 ROS Image 消息转换为 OpenCV 图像
            # "bgr8" 表示我们期望的编码格式是 8位、BGR通道顺序的彩色图像
            current_frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            self.future.set_result(False) # 发出失败信号
            return

        # 新增：将捕获的原始彩色图像保存到文件
        save_path = "/tmp/capture.jpg"
        try:
            cv2.imwrite(save_path, current_frame)
            self.get_logger().info(f'Successfully saved image to {save_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save image: {e}')
            self.future.set_result(False) # 发出失败信号
            return

        # 5. 在这里进行您的图像处理 (可选)
        # gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # try:
        #     # 6. 将处理后的 OpenCV 图像转换回 ROS Image 消息并发布
        #     processed_msg = self.bridge.cv2_to_imgmsg(gray_frame, "mono8")
        #     self.publisher_.publish(processed_msg)
        #     self.get_logger().info('Publishing processed image...')
        # except Exception as e:
        #     self.get_logger().error(f'Failed to publish image: {e}')
            # 即使发布失败，我们仍然认为捕获和保存成功了，所以不设置失败信号

        # 7. 设置 Future 结果为 True，表示任务成功完成
        self.get_logger().info('Task complete. Shutting down node.')
        self.future.set_result(True)


def main(args=None):
    rclpy.init(args=args)
    image_processor = ImageProcessor()
    
    # 修改：阻塞节点，直到 future 完成（即一张图片被处理完）
    rclpy.spin_until_future_complete(image_processor, image_processor.future)
    
    # 销毁节点并关闭 rclpy
    image_processor.destroy_node()
    rclpy.shutdown()
    print("Node has been shut down.")

if __name__ == '__main__':
    main()