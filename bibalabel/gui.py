#!/usr/bin/python3
import tkinter as tk
from tkcolorpicker import askcolor
from bladerTools import *
import copy
from statisticalMeasurements import *

class PredictionEvaluation():
    def __init__(self, main):
        
        main.title('Prediction Evaluation')
        
        #Required list and vars
        self.count = -1
        self.oldImageList = readImageList()
        self.newImageList = copy.deepcopy(self.oldImageList)
        notGoodList = notGoodImageList(self.oldImageList)
        self.notGoodListSize = len(notGoodList)
        self.flatList = [item[0] for item in notGoodList]
        self.listSize = len(self.oldImageList)
        
        self.main = main
        self.main.resizable(False, False)
        main.protocol("WM_DELETE_WINDOW", self.onClosing)
        
        #checkbox1
        self.x1 = tk.IntVar()
        txt="Show only the unfinished labels"
        self.check1 = tk.Checkbutton(main, text = txt, variable=self.x1)
        self.check1.grid(row=0, columnspan = 2, column=0, sticky="W")
        self.x1.set(1)
        
        #checkbox2
        self.x2 = tk.IntVar()
        txt="Save changes on exit"
        self.check2 = tk.Checkbutton(main, text = txt, variable=self.x2)
        self.check2.grid(row=0, columnspan = 2, column=2, sticky="W")
        self.x2.set(1)

        #Color Entry
        self.clr = (180,0,255)
        self.clrEntry = tk.Label(main, text='    ', bg="#FF00B7", height = 1)
        self.clrEntry.grid(row=0, column=3)
        self.clrEntry.bind("<Button-1>", self.colorPicker)

        # checkbox3
        self.x3 = tk.IntVar()
        txt = "Show only the good labels"
        self.check3 = tk.Checkbutton(main, text=txt, variable=self.x3)
        self.check3.grid(row=1, columnspan=2, column=0, sticky="W")
        self.x3.set(0)

        #canvas for Images
        self.canvas = tk.Canvas(main, width=512, height=512)
        self.canvas.grid(row=2, columnspan = 4)

        # button to go backward
        self.button1 = tk.Button(main, text="<--",
                                 command=self.lastImage)
        self.button1.grid(row=3, column=0)
        
        #BAD Button
        self.v = tk.IntVar()
        self.radio1 = tk.Radiobutton(main,
                                     text="BAD",
                                     indicatoron = 0,
                                     width = 20,
                                     height=2,
                                     selectcolor = "red",
                                     padx = 20,
                                     variable=self.v,
                                     command = self.badImage,
                                     value=1)
        self.radio1.grid(row=3, column=1)
        
        #GOOD Button
        self.radio2 = tk.Radiobutton(main, text="GOOD",
                                    indicatoron = 0,
                                    width = 20,
                                    height=2,
                                    selectcolor = "green",
                                    padx = 20,
                                    variable=self.v,
                                     command = self.goodImage,
                                    value=2)
        self.radio2.grid(row=3, column=2)

        # button to go forward
        self.button2 = tk.Button(main, text="-->",
                                 command=self.nextImage)
        self.button2.grid(row=3, column=3)
        
        #Keyboard Shortcuts
        main.bind("<Key>", self.key)
        main.bind("<Left>", self.Left)
        main.bind("<Right>", self.Right)
        main.bind("<Up>", self.Up)
        main.bind("<Down>", self.Down)
        
        #StatusBar
        self.t = tk.StringVar()
        self.mytext = tk.Label(main, textvariable=self.t)
        self.mytext.grid(row=4, columnspan=4)

        
        # loading the first image placeholder
        self.image_on_canvas = self.canvas.create_image(0,
                                                        0,
                                                        anchor = tk.NW,
                                                        image = None)
        self.nextImage()
        self.newStatusImage()
        
    #----------------
    
    def key(self, event):
        #list of Keyboard Shortcuts
        if event.char =='g' or event.char =='G':
            self.goodImage()
            self.newStatusImage()
        if event.char =='b' or event.char =='B':
            self.badImage()
            self.newStatusImage()
            
    def Left(self, event):
        self.lastImage()
    
    def Right(self, event):
        self.nextImage()
    
    def Up(self, event):
        self.goodImage()
        self.newStatusImage()
        
    def Down(self, event):
        self.badImage()
        self.newStatusImage()

    def colorPicker(self, event):
        b,g,r = self.clr
        m , n = askcolor((r,g,b), root)
        if m:
            r,g,b = m
            self.clr = (b,g,r)
            self.clrEntry.config(bg=n)
            self.imgtk = readOverlay(self.newImageList[self.count][0], self.clr)
            self.canvas.itemconfig(self.image_on_canvas,
                                   image = self.imgtk)

    def findCount(self, code):

        # finds the next not-finished image and prevents infinit loops
        if (self.x1.get() == 1 and self.x3.get() == 0):
            noLoop = self.listSize
            while noLoop > 0:
                noLoop -= 1
                self.count += code
                if abs(self.count) > abs(self.listSize - 1):
                    self.count = 0
                status = self.oldStatusImage()
                if status < 1: break

        # finds the next good-labeled image and prevents infinit loops
        if (self.x1.get() == 0 and self.x3.get() == 1):
            noLoop = self.listSize
            while noLoop > 0:
                noLoop -= 1
                self.count += code
                if abs(self.count) > abs(self.listSize - 1):
                    self.count = 0
                dice = float(self.newImageList[self.count][2]) if len(self.newImageList[0])>2 else diceCoefficient(self.oldImageList[self.count][0][:-4])
                status = 0 if dice > STRICTNESS else 1
                if status < 1: break


        if (self.x1.get() == 1 and self.x3.get() == 1):
            noLoop = self.listSize
            while noLoop > 0:
                noLoop -= 1
                self.count += code
                if abs(self.count) > abs(self.listSize - 1):
                    self.count = 0
                dice = float(self.newImageList[self.count][2]) if len(self.newImageList[0])>2 else diceCoefficient(self.oldImageList[self.count][0][:-4])
                status = 0 if (dice > STRICTNESS and self.oldStatusImage()==0) else 1
                if status < 1: break

        if (self.x1.get() == 0 and self.x3.get() == 0):
            self.count += code


    def changeImage(self,code):

        # change to a different image based on the code value
        self.findCount(code)

        if abs(self.count) > abs(self.listSize-1):
            self.count = 0

        # changes image and its status button    
        self.newStatusImage()
        self.imgtk = readOverlay(self.newImageList[self.count][0], self.clr)
        self.canvas.itemconfig(self.image_on_canvas,
                               image = self.imgtk)
        
        #changes title and statusbar
        underText = self.statusText()
        self.main.title('Prediction Evaluation'+
                        '--'+self.newImageList[self.count][0])
        self.t.set(underText)
        
    def lastImage(self):
        # previous image
        code = -1
        self.changeImage(code)

    def nextImage(self):
        # next image
        code = +1
        self.changeImage(code)

    def badImage(self):
        #changes status of the image in the self.newImageList to Bad
        self.newImageList[self.count][1] = '1'
        backupImageList = copy.deepcopy(self.newImageList)
        writeImageList(backupImageList,0) #takes a backup  
        maskDraftDelete(self.newImageList[self.count][0])
        
    
    def goodImage(self):
        #changes status of the image in the self.newImageList to Good
        self.newImageList[self.count][1] = '0'
        backupImageList = copy.deepcopy(self.newImageList)
        writeImageList(backupImageList,0) #takes a backup  
        if self.oldImageList[self.count][1] != '0':
            maskSave(self.newImageList[self.count][0], draft =1)
        
    def oldStatusImage(self):
        #reterns status of the image (bad or good) before modification
        oldStatus = self.oldImageList[self.count][1]
        if oldStatus=='1':
            result = 0
        elif oldStatus=='0':
              result = 1
        else:
            result = -1
        return result
    
    def newStatusImage(self):
        #returns status of the image in the current session and updates
        #the radioButtons relatively 
        newStatus = self.newImageList[self.count][1]
        if newStatus=='1':
            self.radio1.select()
            self.radio2.deselect()
            result = 0
        elif newStatus=='0':
            self.radio1.deselect()
            self.radio2.select()
            result = 1
        else:
            self.radio1.deselect()
            self.radio2.deselect()
            result = -1
        return result
    
    def statusText(self):
        #produces statusbar text
        if self.count<0: number = str(self.listSize+self.count+1)
        else: number = str(self.count+1)
        try:
            badNumber = self.flatList.index(self.newImageList[self.count][0])
            badNumber = str(badNumber+1)
        except:
            badNumber = '---'
        dice = str(100*diceCoefficient(self.newImageList[self.count][0][:-4],4))
        dice = 'dice='+dice[:5]
        myText = ('       ('+number+' of '+str(self.listSize)+' total)'
                   '      ('+badNumber+' of '
                   ' '+str(self.notGoodListSize)+' in progress)')
        return myText
    
    def onClosing(self):
        #handels destroy window
        if self.x2.get()==1:
            writeImageList(self.newImageList)
            maskDraftMove()
        self.main.destroy()

#----------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    PredictionEvaluation(root)
    root.mainloop()
