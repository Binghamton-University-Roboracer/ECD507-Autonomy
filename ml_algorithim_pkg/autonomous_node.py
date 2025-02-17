import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 64 * 64, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class AutonomousNode(Node):
    def __init__(self):
        super().__init__('autonomous_node')
        
        # Subscribers
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        
        self.bridge = CvBridge()
        self.latest_odom = None
        self.latest_lidar = None
        
        self.model = SimpleCNN()
        self.get_logger().info("Autonomous Node Initialized")
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.process_image(cv_image)
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
    
    def process_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        tensor_image = torch.tensor(gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        output = self.model(tensor_image)
        self.get_logger().info(f"Neural Network Output: {output.detach().numpy()}")
        
        cv2.imshow('Edges', edges)
        cv2.waitKey(1)
    
    def odom_callback(self, msg):
        self.latest_odom = msg
        self.get_logger().info(f"Received Odometry: Position - ({msg.pose.pose.position.x}, {msg.pose.pose.position.y})")
    
    def lidar_callback(self, msg):
        self.latest_lidar = msg.ranges
        self.get_logger().info(f"Received Lidar Scan with {len(msg.ranges)} points")
    
    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()
        

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
