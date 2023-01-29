"""tiago_pick_up controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
import time
import numpy as np
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
import os 
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten,Input
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans,MiniBatchKMeans
import glob
import time
from gui import GUI
import tkinter as tk
import warnings
warnings.filterwarnings("ignore")
# Compile the model

resnet_model = Sequential()
pretrained_model= tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3),
    pooling='avg',
    classes=2)

for layer in pretrained_model.layers:
  layer.trainable = False
resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(12,activation='relu'))
resnet_model.add(Dense(2,activation='softmax'))

resnet_model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

inputs = Input(shape=(None, 224, 224, 3))
layer_output = resnet_model.get_layer("resnet50").output
model = Model(inputs=[resnet_model.input,pretrained_model.input], outputs=layer_output)





## FUNCTIONS
#--------------------------
# Returns feature vector for the given image
def generate_feature_vector(img):
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    pred = model.predict([x,x])
    return pred
    
# Normalization for t-SNE
# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
 
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
 
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range
 
# extract x and y coordinates representing the positions of the images on T-SNE plot

def tsne_process(tsne,tsne_vis=False):
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    if tsne_vis == True:
        print("The Plot")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        labels = ['Apples', 'Oranges']
        labels = labels * 16
        for i in range(len(preds)-32):
            labels.append('Unknown')
        colors_per_class = {
            'Apples': (0, 255, 0),  # green
            'Oranges': (255, 128, 0),  # orange
            'Unknown': (0,0,0)
        }
        # for every class, we'll add a scatter plot separately
        for label in colors_per_class:
            # find the samples of the current class in the data
            indices = [i for i, l in enumerate(labels) if l == label]
         
            # extract the coordinates of the points of this class only
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)
         
            # convert the class color to matplotlib format
            color = np.array(colors_per_class[label], dtype=np.float) / 255
         
            # add a scatter plot with the corresponding color and label
            ax.scatter(current_tx, current_ty, c=color, label=label)
           
        
        # build a legend using the labels we set previously
        ax.legend(loc='best')
         
        # finally, show the plot
        plt.show()
    return tx,ty     
    
def all_equal(lst):
    return not lst or lst.count(lst[0]) == len(lst)
    
def check_mini_clustering(label, NUM_SAMPLE,good_list,bad_list):
    confidence = False
    cluster_labels = label[0:2*NUM_SAMPLE]
    control_labels = np.tile(cluster_labels[:2],NUM_SAMPLE)
    new_label = label[-1]
    if np.array_equal(cluster_labels[:2*NUM_SAMPLE], control_labels, equal_nan=False):
        good_labels = [label[i+2*NUM_SAMPLE-2] for i in good_list]
        bad_labels = [label[i+2*NUM_SAMPLE-2] for i in bad_list]
        if all_equal(good_labels) and all_equal(bad_labels):
            confidence =True
        else:
            pass             
        if new_label == cluster_labels[0]:
            predicted_label = 1
        elif new_label == cluster_labels[1]:
            predicted_label = 2
        elif confidence:
            if new_label == good_labels[0]:
                predicted_label = 3
            elif new_label == bad_labels[0]:
                predicted_label =4
            elif new_label == cluster_labels[0]:
                predicted_label = 1
            elif new_label == cluster_labels[1]:
                predicted_label = 2
                
        else:
            predicted_label = 0
    else:
        predicted_label = 0
    return predicted_label, confidence
    
def check_clustering(label, NUM_SAMPLE):
    cluster_labels = label[0:2*NUM_SAMPLE]
    control_labels = np.tile(cluster_labels[:2],NUM_SAMPLE)
    new_label = label[-1]
    if np.array_equal(cluster_labels[:2*NUM_SAMPLE], control_labels, equal_nan=False):
        if new_label == cluster_labels[0]:
            predicted_label = 1
        elif new_label == cluster_labels[1]:
            predicted_label = 2
        else:
            predicted_label = 3
    else:
        predicted_label = 0
    return predicted_label
                   
 # Prepare data
apples = glob.glob('Webots/Granny_Smith'+'/' + '*.' + 'jpg')
oranges = glob.glob('Webots/Orange'+'/' + '*.' + 'jpg')
arrays=[]
for i in range(16):
  img = image.load_img(apples[i], target_size=(224,224))
  x = image.img_to_array(img)
  x = np.expand_dims(x,axis=0)
  x = preprocess_input(x)
  arrays.append(x)
  img2 = image.load_img(oranges[i], target_size=(224,224))
  y = image.img_to_array(img2)
  y = np.expand_dims(y,axis=0)
  y = preprocess_input(y)
  arrays.append(y)
data = np.concatenate(arrays, axis=0)
preds = model.predict([data,data])
tsne = TSNE(n_components=2).fit_transform(preds)
tx,ty = tsne_process(tsne)
mini = MiniBatchKMeans(n_clusters=2, random_state=0).fit(tsne)
kmeans = KMeans(n_clusters=2, random_state=0).fit(tsne)        
old_inertia_mini = mini.inertia_
old_inertia = kmeans.inertia_
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


def neutral_position(rotational_motors,linear_motors):
    rotational_motors[4].setPosition(1.57)
    # Rotate the arm1 joint 90 degrees
    rotational_motors[0].setPosition(1.63)
    # Rotate the arm2 joint
    rotational_motors[1].setPosition(0.9)
    # Rotate the arm4 joint 
    rotational_motors[3].setPosition(1)
    #Rotate the arm6 joint 
    rotational_motors[5].setPosition(-0.44)
    for motor in linear_motors:
        motor.setPosition(0.045)
        motor.setVelocity(motor.getMaxVelocity())
    old_left = 0.044
    old_right = 0.044 
       
def pick_object(rotational_motors,linear_motors,old_left,old_right):
    # Rotate the arm4 joint 
    linear_motors[0].setPosition(0.001)
    linear_motors[1].setPosition(0.001)    
    left_grip = position_sensors[11].getValue()
    right_grip = position_sensors[12].getValue()   
    return left_grip, right_grip
   
# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)
sim_time = 0
object_picked = False
cluster_num = 2
# Main loop:
# - perform simulation steps until Webots is stopping the controller
i = 1
loop_counter = -1
confidence = False
neutral_position(rotational_motors,linear_motors)
good_object = []
bad_object = []
while robot.step(timestep) != -1:
    loop_counter -=1
    sim_time = sim_time + timestep*0.001
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()
 
    if sim_time > 2:
        img = camera.getImageArray()
        img = np.asarray(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        # image = cv2.resize(image,(1280,720))
        img = cv2.flip(img, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # count the number of pixels in the image that are hue values between 110 and 125
        mask = cv2.inRange(hsv, (110, 0, 0), (125, 255, 255))
        # count the number of pixels in the mask
        conveyor_pixels = np.count_nonzero(mask)
        if conveyor_pixels < 50000:              
            cv2.imwrite(f'image{i}.jpg',img)
            img_path = f'image{i}.jpg'  
            time.sleep(0.1)
            img2 = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img2)           
            new_fv = generate_feature_vector(x)
            preds = np.vstack([preds,new_fv])
            tsne = TSNE(n_components=2).fit_transform(preds)
            tx, ty = tsne_process(tsne)
            mini = MiniBatchKMeans(n_clusters=4, random_state=0).fit(tsne)
            kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(tsne)      
            if cluster_num == 2:
                new_inertia = kmeans.inertia_
            if new_inertia > 3*old_inertia:
                root = tk.Tk()
                my_gui = GUI(root,"Unknown Object Detected", "I believe this is neither an apple nor an orange.\n Am I right?")               
                root.mainloop()
                if my_gui.value:
                    cluster_num = 3                   
                else:
                    pass
            else:
                k_lab = check_clustering(kmeans.labels_,16)
                print(k_lab)
                if min(len(good_object),len(bad_object)) >= 2:
                    m_lab, confidence = check_mini_clustering(mini.labels_,16,good_object,bad_object)
                    if confidence:
                        if m_lab == 1:
                            print("This is an apple")
                        elif m_lab == 2:
                            print("This is an orange")
                        elif m_lab == 3:
                            print("This is a new object I can pick up!")
                            loop_counter = 150                            
                        else:
                            print("This is a new object that I can't pick up, HELP!")
                        i += 1
                        sim_time = 0
                        continue
                if confidence == False:
                    if k_lab == 0:
                        print("Something is wrong")
                    elif k_lab == 1:
                        print("This is an apple")
                    elif k_lab == 2:
                        print("This is an orange")                       
                    elif k_lab == 3 and confidence == False:
                        loop_counter = 150
                    #root = tk.Tk()
                    #my_gui2 = GUI(root,"Object Grasping Attempt", "I am trying to grasp this object.\n Did I succeed?")
                    #root.mainloop()
            
            old_inertia = new_inertia       
            i += 1
            sim_time = 0
        if loop_counter == 155:
            # ADD HELP MOTION HERE
            loop_counter = -1
        if loop_counter == 0:
            print('Picking Up')
            old_left,old_right = pick_object(rotational_motors,linear_motors,old_left,old_right)
        elif loop_counter == -40:
            if confidence==False:
                root = tk.Tk()
                my_gui2 = GUI(root,"Object Grasping Attempt", "I am trying to grasp this object.\n Did I succeed?")
                root.mainloop()
                if my_gui2.value:
                    good_object.append(i)
                else:
                    bad_object.append(i)
            neutral_position(rotational_motors,linear_motors)

        # old_left,old_right,object_picked = pick_object(rotational_motors,linear_motors,old_left,old_right,object_picked)
    

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass


