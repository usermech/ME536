
from controller import Supervisor
from random import random
import numpy as np
### CONSTANTS #############################################################

# how many boxes to spawn
BOX_COUNT=10
# box size limits
SIZELIMX=[0.05, 0.10]
SIZELIMY=[0.02, 0.08]
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
    print("An apple has been spawned")
    children.importMFNodeFromString(-1, apple_properties)
    
def spawnorange(translation):
    # set the box attributes
    orange_properties=(
         "DEF Orange Orange {"
        f" translation {translation[0]} {translation[1]} {translation[2]}"
        "}}"
        )
    # spawn it
    
    print("An orange has been spawned")
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
    
    print("A box has been spawned")
    children.importMFNodeFromString(-1, box_properties)

### CODE ##################################################################

supervisor=Supervisor()
timestep = int(supervisor.getBasicTimeStep())
counter = 0
position=[0,
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
        if box_pos[0] > 1.3:
            box_node.remove()
    if orange_node is not None:
        orange_pos = orange_node.getPosition()
        if orange_pos[0] > 1.3:
            orange_node.remove()
    if apple_node is not None:
        apple_pos = apple_node.getPosition()
        if apple_pos[0] > 1.3:
            apple_node.remove()
    if counter == 250:
        counter = 0
    if counter == 0:
        object_choice = rand(0,5)
        print(object_choice)
        if object_choice < 2:          
            spawnorange(position)
        elif object_choice <4:
            spawnorange(position)
        else:
            size=[rand(SIZELIMX[0], SIZELIMX[1]),
                  rand(SIZELIMY[0], SIZELIMY[1]),
                  0]
                  #rand(SIZELIMZ[0], SIZELIMZ[1])]
            size[2] = 0.15 - size[1]; # burayi atabilirsin
            color=[rand(0.5, 1) for j in range(0,3)]
            #print(size[1])
            # spawn in a box:            
            spawnbox(position, size, color)
    counter +=1
        