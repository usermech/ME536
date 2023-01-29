 
from controller import Supervisor
from random import random
import numpy as np
### CONSTANTS #############################################################

# how many boxes to spawn
BOX_COUNT=10
# box size limits
SIZELIMX=[0.05, 0.11]
SIZELIMY=[0.02, 0.04]
SIZELIMZ=[0.05, 0.2]
# box position limits
POSYLIM=[-0.5, 0.5]

### FUNCTIONS #############################################################

# returns a random float between min and max.
def rand(min, max):
    return ((max-min)*random())+min
    
def spawnapple(translation):
    # set the box attributes
    apple_properties=(
         "DEF Apple Apple {"
        f" translation {translation[0]} {translation[1]} {translation[2]}"
        "}}"
        )
    # spawn it
    children.importMFNodeFromString(-1, apple_properties)
    
def spawnorange(translation):
    # set the box attributes
    orange_properties=(
         "DEF Orange Orange {"
        f" translation {translation[0]} {translation[1]} {translation[2]}"
        "}}"
        )
    # spawn it
    
    children.importMFNodeFromString(-1, orange_properties)

# spawn a box at the bottom of the scene tree.
# translation, size and color are 3-element arrays describing the box.
def spawnbox(translation, size, color):
    # set the box attributes
    box_properties=(
         "DEF BOX SolidBox {"
        f" translation {translation[0]} {translation[1]} {translation[2]}"
        f" size {size[0]} {size[1]} {size[2]}"
         " appearance MattePaint {"
        f"baseColor {color[0]} {color[1]} {color[2]}"
        "}"
        "physics Physics {"
        f" density 100"
        "}}"
        )
    # spawn it
    
    children.importMFNodeFromString(-1, box_properties)

### CODE ##################################################################

supervisor=Supervisor()
timestep = int(supervisor.getBasicTimeStep())
counter = 0
position=[-0.1,
          0.005,
          0.7]
# get the scene tree
children=supervisor.getRoot().getField("children")
while supervisor.step(timestep) != -1:     
    box_node = supervisor.getFromDef('BOX')
    orange_node = supervisor.getFromDef('Orange')
    apple_node = supervisor.getFromDef('Apple')
    if box_node is not None:
        box_pos = box_node.getPosition()
        if box_pos[0] > 1.8:
            box_node.remove()
    if orange_node is not None:
        orange_pos = orange_node.getPosition()
        if orange_pos[0] > 1.8:
            orange_node.remove()
    if apple_node is not None:
        apple_pos = apple_node.getPosition()
        if apple_pos[0] > 1.8:
            apple_node.remove()
    if counter == 250:
        counter = 0
    if counter == 0:
        object_choice = rand(0,4)
        if object_choice < 0.8:          
            spawnapple(position)
        elif object_choice <1.6:
            spawnorange(position)
        else:
            good_or_bad = np.random.randint(5)
            if good_or_bad <=2 :
                size=[rand(SIZELIMX[0], SIZELIMX[1]),
                      rand(SIZELIMY[0], SIZELIMY[1]),
                      0]
                size[2] = 0.165 - size[1]
            else:
                size=[rand(SIZELIMX[0], SIZELIMX[1]),
                      rand(0.2-SIZELIMY[0], 0.2-SIZELIMY[1]),
                      0]
                size[2] = 0.19 - size[1]
                      
                
                
            
            
             # burayi atabilirsin
            color=[rand(0.3, 0.6) for j in range(0,3)]
            color[0] = rand(0.6, 0.8) 
            #print(size[1])
            # spawn in a box:            
            spawnbox(position, size, color)
    counter +=1
        