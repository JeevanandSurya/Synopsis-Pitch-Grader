import tkinter
from tkinter import *
import pickle
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as es
from sklearn.feature_extraction.text import TfidfVectorizer

with open('rating_model_pickle', 'rb') as f:
  model = pickle.load(f)

vectorizer = TfidfVectorizer(stop_words = es, lowercase=False)

master = Tk()
Label(master, text='Enter the synopsis').grid(row=0)
text = Text(master, height=8)
text.grid(column=0, row=1)

master.title('Pitching Idea')

def CallBack():
   inp = text.get(1.0, "end-1c")
   pred = model.predict(vectorizer.transform([inp]))
   grade.config(text = pred)

button = tkinter.Button(master, text ="Submit", command = CallBack)
button.grid(column=0, row=2)

grade = Label(master, text = "")
grade.grid(column = 0, row = 3)

master.mainloop()
