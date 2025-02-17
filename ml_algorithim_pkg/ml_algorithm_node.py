import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped
import cv2
from cv_bridge import CvBridge
import torch  # Use TensorFlow if your model is in TF
import numpy as np

class AutonomousDriver(Node):
    def __init__(self):
        super().__init__('autonomous_driver')
        self.bridge = CvBridge()

        # Load the trained model (assuming PyTorch for now)
        self.model = torch.load('model.pth', map_location=torch.device('cpu'))
        self.model.eval()  # Set model to inference mode

        # Subscribers and Publishers
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.control_pub = self.create_publisher(AckermannDriveStamped, '/vesc/commands', 10)

    def image_callback(self, msg):
        # Convert ROS2 Image to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Preprocess image (resize, normalize, convert to tensor)
        processed_frame = self.preprocess_image(frame)

        # Get steering prediction from the model
        steering_angle = self.predict_steering(processed_frame)

        # Publish the control command
        self.publish_control(steering_angle)

    def preprocess_image(self, frame):
        """Resize, normalize, and convert image to tensor."""
        frame = cv2.resize(frame, (200, 66))  # Resize to match training input
        frame = frame / 255.0  # Normalize
        frame = np.transpose(frame, (2, 0, 1))  # Change shape to (C, H, W)
        frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        return frame

    def predict_steering(self, frame):
        """Use the ML model to predict steering angle."""
        with torch.no_grad():
            output = self.model(frame)  # Run inference
        return float(output[0])  # Convert tensor output to float

    def publish_control(self, steering_angle):
        """Publish the predicted steering command."""
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = steering_angle
        msg.drive.speed = 1.0  # Adjust speed as needed
        self.control_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
