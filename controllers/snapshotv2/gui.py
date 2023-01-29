"""
Create a pop up window with a message and two buttons.
"""
import tkinter as tk
from tkinter import messagebox


class GUI:
    def __init__(self, master,title, message):
        self.master = master
        master.title(title)

        self.message = message
        self.label = tk.Label(master, text=self.message)
        self.label.pack()
        self.yes_button = tk.Button(master, text="Yes", command=self.yes)
        self.yes_button.pack()

        self.no_button = tk.Button(master, text="No", command=self.no)
        self.no_button.pack()
        self.value = False        
    

    def yes(self):
        self.value = True        
        self.master.destroy()

    def no(self):
        self.value = False  
        self.master.destroy()
        
  


'''
for i in range(0,10):
    box = False
    if i >-1:        
        root = tk.Tk()
        my_gui = GUI(root,"Unknown Object Detected", "I believe this is neither an apple nor an orange.\n Am I right?")               
        root.mainloop()
        print(my_gui.value)
        # print("Wow, you are good at this game!")
    else:
        root = tk.Tk()
        my_gui2 = GUI(root,"Object Grasping Attempt", "I am trying to grasp this object.\n Did I succeed?")   
        root.mainloop()
'''