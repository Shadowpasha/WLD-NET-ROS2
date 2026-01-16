import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torch
import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add the package directory to sys.path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

try:
    from . import dehazing_model
    from . import Feature_Processing
except ImportError:
    import dehazing_model
    import Feature_Processing

class WebcamDehazingNode(Node):
    def __init__(self):
        super().__init__('webcam_dehazing_node')

        # Parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('output_topic', '/camera/image_dehazed')
        self.declare_parameter('device_id', 0)
        
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.device_id = self.get_parameter('device_id').get_parameter_value().integer_value

        if not self.model_path:
            possible_model_path = current_dir.parent / 'models' / 'RD_dehazing_model_final.pth'
            if possible_model_path.exists():
                self.model_path = str(possible_model_path)
            else:
               self.get_logger().warn("No model_path provided and could not find default model. Please provide model_path parameter.")

        self.get_logger().info(f"Loading model from: {self.model_path}")

        # Model Init
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Running on device: {self.device}")

        self.model = dehazing_model.Dehazing_Model()
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
        
        self.model.to(self.device)
        self.model.eval()

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, self.output_topic, 10)
        
        # Webcam Init
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open webcam device {self.device_id}")
        
        # Timer for processing (aim for ~30 FPS or as fast as possible)
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.get_logger().info(f"Webcam dehazing node started, publishing to {self.output_topic}")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame")
            return

        # Resize to 480p for performance
        height, width = frame.shape[:2]
        new_height = 480
        new_width = int(width * (new_height / height))
        frame = cv2.resize(frame, (new_width, new_height))

        # Preprocess
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_img = torch.from_numpy(rgb_image.transpose((2, 0, 1))).float().div(255.0)
        tensor_img = tensor_img.unsqueeze(0).to(self.device)

        # Padding
        h, w = tensor_img.shape[2], tensor_img.shape[3]
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        
        if pad_h > 0 or pad_w > 0:
            tensor_img = torch.nn.functional.pad(tensor_img, (0, pad_w, 0, pad_h), mode='reflect')

        tensor_img = Feature_Processing.normalize(tensor_img)

        # Inference
        try:
            with torch.no_grad():
                output = self.model(tensor_img)
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
            return
            
        # Postprocess
        output = Feature_Processing.denormalize(output)
        
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :h, :w]

        output = output.squeeze(0).cpu().numpy().transpose((1, 2, 0))
        output = np.clip(output, 0, 1) * 255.0
        output = output.astype(np.uint8)
        
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        try:
            out_msg = self.bridge.cv2_to_imgmsg(output_bgr, "bgr8")
            out_msg.header.stamp = self.get_clock().now().to_msg()
            out_msg.header.frame_id = "camera_link"
            self.publisher.publish(out_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error (Publish): {e}")

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = WebcamDehazingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
