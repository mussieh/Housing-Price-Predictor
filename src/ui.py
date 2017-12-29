# Adair County Housing Prediction Program User Interface
# Group Members : Ronit Das, Mussie Habtemichael, Jerry Lin, Zorig Magnaituvshin


from tkinter import *
from tkinter import messagebox
import re
from housing_predictor import PricePredictor

# The window of the program
class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

	# Creation of the init_window
    def init_window(self):

        inputValues = []

    	# changing the title of our master widget
        self.master.title("Adair County Housing Price Prediction Program")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # Main Page Caption
        title = Label(self, text="Input Values")
        title.place(x=160, y=10)

        bedrooms = Label(self, text="Bedrooms")
        bedrooms.place(x=20, y=40)

        bedroomsIn = Entry(self, bd=5)
        bedroomsIn.place(x=20, y=70)
        inputValues.append(bedroomsIn)

        baths = Label(self, text="Baths")
        baths.place(x=20, y=100)

        bathsIn = Entry(self, bd=5) # bath input
        bathsIn.place(x=20, y=130)
        inputValues.append(bathsIn)

        age = Label(self, text="Age")
        age.place(x=20, y=160)

        ageIn = Entry(self, bd=5) # age input
        ageIn.place(x=20, y=190)
        inputValues.append(ageIn)

        floorsize = Label(self, text="Floor Size")
        floorsize.place(x=20, y=220)

        floorsizeIn = Entry(self, bd=5) #floor size input
        floorsizeIn.place(x=20, y=250)
        inputValues.append(floorsizeIn)

        lotsize = Label(self, text="Lot Size")
        lotsize.place(x=20, y=280)

        lotsizeIn = Entry(self, bd=5) # lot size input
        lotsizeIn.place(x=20, y=310)
        inputValues.append(lotsizeIn)

        pricepersqft = Label(self, text="Price Per Sq. ft")
        pricepersqft.place(x=20, y=340)

        pricepersqftIn = Entry(self, bd=5) # Price per square foot input
        pricepersqftIn.place(x=20, y=370)
        inputValues.append(pricepersqftIn)

        # text to be used in prediction box
        priceText = StringVar()

        predictedPrice = Label(self, text="Predicted Price: $")
        predictedPrice.place(x=210, y=375)
        predictedPriceOut = Entry(self, width=30, state="readonly", textvariable=priceText)
        predictedPriceOut.place(x=320, y=374)

        # creating a predict button instance
        predictButton = Button(self, text="Predict", command= lambda: self.predict(inputValues, priceText))

        # placing the button on the window
        predictButton.place(x=20, y=410)

        # The price object to run the machine learning prediction on
        self.priceObj = PricePredictor() # construct the price predictor object

    # Prediction function that outputs the predicted price
    def predict(self, inputValues, priceText):
            self.priceObj.setParameters(inputValues[0], # the input values from the user
                    inputValues[1],inputValues[2],inputValues[3],
                    inputValues[4],inputValues[5])
            priceText.set(self.priceObj.predict()) # setting the price text to the prediction


root = Tk()
root.geometry("600x600") # window size
app = Window(root)
root.mainloop()
