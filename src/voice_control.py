import speech_recognition as sr
import os

r = sr.Recognizer()

def listen_command():
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)

    try:
        command = r.recognize_google(audio).lower()
        print("You said:", command)
        return command
    except:
        return ""

def open_app(command):
    if "chrome" in command:
        os.system("start chrome")
    elif "notepad" in command:
        os.system("start notepad")
    elif "calculator" in command:
        os.system("start calc")