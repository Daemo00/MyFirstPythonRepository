"""Make a two-player Rock-Paper-Scissors game. (Hint: Ask for player plays (using input), compare them, print out a message of congratulations to the winner, and ask if the players want to start a new game)

Remember the rules:

Rock beats scissors
Scissors beats paper
Paper beats rock"""

game = ["r", "s", "p"]

playing = True
while playing:
    playerA = input("Player A choose: r, s, p: ")
    playerB = input("Player B choose: r, s, p: ")
    
    if (game.index(playerB) - game.index(playerA)) % 3 == 1:
        print("Player A wins!")
    elif (game.index(playerB) - game.index(playerA)) % 3 == 0:
        print("It's a tie!")
    else:
        print("Player B wins!")
    
    playing = input("Another round? ") == "y"
