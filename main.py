from os import path, listdir, startfile
from subprocess import call


exercises_dir = "exercises\practice-python"
exercises = [f for f in listdir(exercises_dir) if f.endswith(".py")]
exercises.sort()

last_exercise = exercises.pop()

choice = "y"
while choice == "y":
    print("Executing " + last_exercise)
    call(path.join(exercises_dir, last_exercise), shell=True)

    choice = input("Execute {} again? ".format(last_exercise))
