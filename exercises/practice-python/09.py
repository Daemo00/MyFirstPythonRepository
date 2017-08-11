"""Generate a random number between 1 and 9 (including 1 and 9). Ask the user to guess the number, then tell them whether they guessed too low, too high, or exactly right. (Hint: remember to use the user input lessons from the very first exercise)

Extras:

Keep the game going until the user types "exit"
Keep track of how many guesses the user has taken, and when the game ends, print this out."""

import random


playing = True
while playing:
    n = random.randint(1, 9)
    while True:
        guess = int(input("Make a guess: "))
        if guess > n:
            print("Too high")
        elif guess < n:
            print("Too low")
        else:
            playing = input("Got it! Play again? (Type exit to stop) ") != "exit"
            break
