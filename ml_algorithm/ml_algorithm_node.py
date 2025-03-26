import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from ackermann_msgs.msg import AckermannDriveStamped
import torch.nn as nn
import cv2
from cv_bridge import CvBridge
import torch  # Use TensorFlow if your model is in TF
import numpy as np
import os
import torch.nn.functional as F


# Define CNN model
class ResNet15(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3,64,kernel_size = 7, stride = 1, padding = 'same')
        self.conv1_maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv1_bn = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64,64,kernel_size = 3, stride = 1, padding = 'same')
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64,64,kernel_size = 3, stride = 1, padding = 'same')
        self.conv3_bn = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64,64,kernel_size = 3, stride = 1, padding = 'same')
        self.conv4_bn = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64,64,kernel_size = 3, stride = 1, padding = 'same')
        self.conv5_bn = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64,128,kernel_size = 3, stride = 1, padding = 'same')
        self.conv6_maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv6_bn = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128,128,kernel_size = 3, stride = 1, padding = 'same')
        self.conv7_bn = nn.BatchNorm2d(128)
        self.conv7_shortcut = nn.Conv2d(64,128, kernel_size = 2, stride = 2)
        self.conv7_shortcut_bn = nn.BatchNorm2d(128)

        self.conv8 = nn.Conv2d(128,128,kernel_size = 3, stride = 1, padding = 'same')
        self.conv8_bn = nn.BatchNorm2d(128)

        self.conv9 = nn.Conv2d(128,128,kernel_size = 3, stride = 1, padding = 'same')
        self.conv9_bn = nn.BatchNorm2d(128)

        self.conv10 = nn.Conv2d(128,256,kernel_size = 3, stride = 1, padding = 'same')
        self.conv10_maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv10_bn = nn.BatchNorm2d(256)

        self.conv11 = nn.Conv2d(256,256,kernel_size = 3, stride = 1, padding = 'same')
        self.conv11_bn = nn.BatchNorm2d(256)
        self.conv11_shortcut = nn.Conv2d(128,256, kernel_size = 2, stride = 2)
        self.conv11_shortcut_bn = nn.BatchNorm2d(256)

        self.conv12 = nn.Conv2d(256,256,kernel_size = 3, stride = 1, padding = 'same')
        self.conv12_bn = nn.BatchNorm2d(256)

        self.conv13 = nn.Conv2d(256,256,kernel_size = 3, stride = 1, padding = 'same')
        self.conv13_bn = nn.BatchNorm2d(256)

        self.conv14 = nn.Conv2d(256,256,kernel_size = 3, stride = 1, padding = 'same')
        self.conv14_bn = nn.BatchNorm2d(256)

        self.FC1 = nn.Linear(256*4*12,2)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv1_maxpool(x)
        x = F.relu(self.conv1_bn(nn.MaxPool2d(kernel_size=3,stride = 2,padding = 1)(x)))

        # first block
        x1 = F.relu(self.conv2_bn(self.conv2(x)))
        x1 = F.relu(self.conv3_bn(self.conv3(x1)))
        x = x1 + x 
        x1 = F.relu(self.conv4_bn(self.conv4(x)))
        x1 = F.relu(self.conv5_bn(self.conv5(x1)))
        x = x1 + x

        #second block
        x2 = F.relu(self.conv6_bn(self.conv6_maxpool(self.conv6(x))))
        x2 = F.relu(self.conv7_bn(self.conv7(x2)))
        # print(x2.shape)
        # print(self.conv7_shortcut_bn(self.conv7_shortcut(x)).shape)
        x = F.relu(x2 + self.conv7_shortcut_bn(self.conv7_shortcut(x)))
        x2 = F.relu(self.conv8_bn(self.conv8(x)))
        x2 = F.relu(self.conv9_bn(self.conv9(x2)))
        x = x2 + x

        x3 = F.relu(self.conv10_bn(self.conv10_maxpool(self.conv10(x))))
        x3 = F.relu(self.conv11_bn(self.conv11(x3)))
        x = F.relu(self.conv11_shortcut_bn(self.conv11_shortcut(x)))
        x3 = F.relu(self.conv12_bn(self.conv12(x3)))
        x3 = F.relu(self.conv13_bn(self.conv13(x3)))
        x3 = F.relu(self.conv14_bn(self.conv14(x3)))
        # print(x.shape)
        x = x + x3
        

        x = x.view(x.size(0),-1)
        x = self.FC1(x)
        return x

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3840, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 2)  # Output: Steering angle and Throttle
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class AutonomousDriver(Node):
    def __init__(self):
        super().__init__('autonomous_driver')
        self.bridge = CvBridge()

        # Load the trained model (assuming PyTorch for now)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNet15().to(self.device)
        checkpoint = torch.load('/media/ecd507/JetsonOrinNano/home/ecd507/training/model.pth')
        
        self.model.load_state_dict(checkpoint)
        self.get_logger().info(f"Model Loaded Successfully")
        
        self.servo_val = 0.0

        #,map_location=torch.device(self.device) - extra argument to previous line
        self.model.eval()  # Set model to inference mode

        # Subscribers and Publishers
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        # self.servo_sub = self.create_subscription(Float64, 'commands/servo/position', self.servo_callback, 10)
        self.control_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        
    # def servo_callback(self, msg):
    #     self.servo_val = msg.data
    
    def image_callback(self, msg):
        # Convert ROS2 Image to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Preprocess image (resize, normalize, convert to tensor)
        processed_frame = self.preprocess_image(frame)

        # Get steering prediction from the model
        control_values = self.predict_control(processed_frame)[0]
        

        # Publish the control command
        self.publish_control(control_values)

    def preprocess_image(self, frame):
        """Resize, normalize, and convert image to tensor."""
        frame = cv2.resize(frame, (200, 66))  # Resize to match training input
        frame = frame / 255.0  # Normalize
        frame = np.transpose(frame, (2, 0, 1))  # Change shape to (C, H, W)
        frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).to(self.device)  # Add batch dimension
        return frame

    def predict_control(self, frame):
        """Use the ML model to predict steering angle."""
        with torch.no_grad():
            output = self.model(frame)  # Run inference
        return output.tolist() # Convert tensor output to float

    def publish_control(self, control_values):
        """Publish the predicted steering command."""
        steering_angle, throttle = control_values[0], control_values[1]
        msg = AckermannDriveStamped()
        # if (self.servo_val == 0.0):
        #     steering_angle = 0.5192
        steering_angle = (steering_angle - 0.5192)/-1.2
        msg.drive.steering_angle = steering_angle
        msg.drive.speed = (throttle * 23250.0)/4614.0 # de-normalize predictions - change to 23250 after next train
        # self.get_logger().info(f"{msg.drive.steering_angle,msg.drive.speed }")
        self.control_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
