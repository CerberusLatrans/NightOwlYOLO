import math

secret_number = input("Pick a secret number between 1 and 100 (inclusive). I will guess it within seven tries.")
correct = False
UL = 100
LL = 0
while correct==False:
    guess = math.floor((UL+LL)/2)
    feedback = input(f"Is {guess} 'too high' or 'too low' or 'just right?'")
    if feedback == "too high":
        UL = guess
    elif feedback ==  "too low":
        LL = guess
    elif feedback ==  "just right":
        correct = True
        print("yay")
    else:
        feedback = input("You're not making any sense! Is this guess 'too high' or 'too low' or 'just right?'")
