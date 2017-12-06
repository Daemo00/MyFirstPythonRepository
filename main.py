from os import path, listdir, system


exercises_dir = path.join('exercises', 'practice-python')
exercises = [f for f in listdir(exercises_dir) if f.endswith(".py")]
exercises.sort()

last_exercise = exercises.pop()

choice = "y"
while choice == "y":
    print("Executing " + last_exercise)
    file_path = path.join(exercises_dir, last_exercise)
    system('python {}'.format(file_path))
    choice = input("Execute {} again? ".format(last_exercise))
