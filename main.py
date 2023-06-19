from typing import Union
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import speech_recognition as sr
from tmp import the_machine
r = sr.Recognizer()

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}



@app.get("/getAudio")
def read_root():
    with sr.Microphone() as source:
        print("Speak now...")
        audio = r.listen(source)

    try:
        #        print("You said: " + r.recognize_google(audio))
        return {"data": the_machine(r.recognize_google(audio))}
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio.")
        return {"data": "---"}
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return {"data": "---"}
