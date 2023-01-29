"""tiago_pick_up controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
import time
import numpy as np
import cv2

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# Enable camera
camera = robot.getDevice("camera")
camera.enable(timestep)

# Create a list of rotational motor devices names in order to get the devices from the TIAGo robot
rotational_motor_names = ["arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint", "arm_6_joint", "arm_7_joint", "wheel_left_joint", "wheel_right_joint"]

# Create a list of linear motor devices names in order to get the devices from the TIAGo robot
linear_motor_names = ["gripper_left_finger_joint", "gripper_right_finger_joint"]

# Create a list of PositionSensor devices names in order to get the devices from the TIAGo robot
position_sensor_names = ["arm_1_joint_sensor", "arm_2_joint_sensor", "arm_3_joint_sensor", "arm_4_joint_sensor", "arm_5_joint_sensor", "arm_6_joint_sensor", "arm_7_joint_sensor", "head_1_joint_sensor", "head_2_joint_sensor", "wheel_left_joint_sensor", "wheel_right_joint_sensor", "gripper_left_finger_joint_sensor", "gripper_right_finger_joint_sensor", "torso_lift_joint_sensor"]

# Get devices from the TIAGo robot
rotational_motors = []
linear_motors = []
position_sensors = []

for name in rotational_motor_names:
    rotational_motors.append(robot.getDevice(name))

for name in linear_motor_names:
    linear_motors.append(robot.getDevice(name))

for name in position_sensor_names:
    position_sensors.append(robot.getDevice(name))

# Enable rotational motors with zero position get max velocity for each motor and set velocity to 0.8 of max velocity
for motor in rotational_motors:
    motor.setPosition(0.07)
    motor.setVelocity(0.8*motor.getMaxVelocity())


# Enable linear motors with 0.045 position
for motor in linear_motors:
    motor.setPosition(0.045)
    motor.setVelocity(0.6*motor.getMaxVelocity())
    old_left = 0.044
    old_right = 0.044
# Enable position sensors
for sensor in position_sensors:
    sensor.enable(timestep)


def neutral_position(rotational_motors):
    rotational_motors[4].setPosition(1.57)
    # Rotate the arm1 joint 90 degrees
    rotational_motors[0].setPosition(1.60)
    # Rotate the arm2 joint
    rotational_motors[1].setPosition(0.1)
    # Rotate the arm4 joint 
    rotational_motors[3].setPosition(0.1)
    #Rotate the arm6 joint 
    rotational_motors[5].setPosition(-0.05) 
       
def pick_object(rotational_motors,linear_motors,old_left,old_right,object_picked):
    # Rotate the arm4 joint 
    linear_motors[0].setPosition(0.001)
    linear_motors[1].setPosition(0.001)    
    left_grip = position_sensors[11].getValue()
    right_grip = position_sensors[12].getValue()
    if abs(left_grip-old_left)<10e-5 and abs(right_grip-old_right)<10e-5 : 
        object_picked = True        
    if object_picked == True:  
        rotational_motors[3].setPosition(-0.32)
    return left_grip, right_grip,object_picked
# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)
sim_time = 0
object_picked = False
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    sim_time = sim_time + timestep*0.001
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()
    neutral_position(rotational_motors)
    if sim_time > 3:
        image = camera.getImageArray()
        image = np.asarray(image, dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        # image = cv2.resize(image,(1280,720))
        image = cv2.flip(image, 1)
        
        old_left,old_right,object_picked = pick_object(rotational_motors,linear_motors,old_left,old_right,object_picked)
    

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

# Enter here exit cleanup code.
