
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class TestPublisher(Node):
    def __init__(self):
        super().__init__('test_publisher')
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_dehazed',
            self.listener_callback,
            10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.bridge = CvBridge()
        self.received = False
        self.start_time = time.time()

    def timer_callback(self):
        # Create a dummy image (random noise or solid color)
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.putText(img, 'Test', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.publisher_.publish(msg)
        self.get_logger().info('Published test image')
        
        # Timeout after 10 seconds
        if time.time() - self.start_time > 100:
            self.get_logger().error('Timeout waiting for response')
            raise SystemExit

    def listener_callback(self, msg):
        self.get_logger().info('Received dehazed image!')
        self.received = True
        raise SystemExit

def main(args=None):
    rclpy.init(args=args)
    test_publisher = TestPublisher()
    try:
        rclpy.spin(test_publisher)
    except SystemExit:
        pass
    test_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
