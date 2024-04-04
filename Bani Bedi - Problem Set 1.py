# Exercise 0

def github() -> str:

    """
    This function will return Bani Bedi's GitHub page for Problem Set 1.
    """

    return "https://github.com/banibedi/Econ481.git"

# Exercise 1 - Installing Packages
import numpy
import pandas
import scipy
import matplotlib
import seaborn

# Exercise 2 

def evens_and_odds(n: int) -> dict: # Defines the evens_and_odds function that takes 'n' and returns a dictionary 
    """
    Return a dictionary with two keys: "evens" being the sum of all the even natural numbers less than n while "odds" is the sum of all natural numbers less than n.
    """

    # Sum of all "even" natural numbers less than n
    total_even = sum(x for x in range(2, n, 2)) # Starts from 2 up to n with a step of 2

    # Sum for all "odd" natural numbers less than n
    total_odd = sum(range(1, n, 2)) # Starts from 1 up to m with a step of 1

    return {'evens': total_even, 'odds': total_odd} # Returns a dictionar with two keys: 'evens' and 'odds'

def main(): # This function allows the user to input his/her desired values
    while True:
        n = int(input("Please input a number: ")) # Prompts user to input a number that is stored as the variable 'n'
        result = evens_and_odds(n) # Stores the result
        additional_numbers = input("Would you like to enter another number? (y/n)") # Asks the user if they want to nput another number
        if additional_numbers == "y": # If the user enters 'y', the loop continues
            continue
        else:
            # Print out the sums of even and odd numbers kess than 'n'
            print(f"Sum of even numbers less than {n}: {result['evens']}")
            print(f"Sum of odd numbers less than {n}: {result['odds']}")
            break # Exits loop

main()

from typing import Union
from datetime import datetime # Using datetime package as assignment instructed

# States that the first and second date arguments need to be strings; the third argument specifies the output format as a string
# Union function states that multiple types are possible for the output value: a string or a float
def time_diff(date_1: str, date_2: str, out: str = "str") -> Union[str, float]: 
    """
    This function takes two strings in the format 'YYYY-MM-DD' and a keyword 'float' or 'string'.
    It returns the absolute difference in days between the two dates as a float or a string, based on the keyword.
    """

    # Converting strings into date time objects
    date_1 = datetime.strptime(date_1, '%Y-%m-%d') # Converts the string date_1 into a date time object using the format YYYY-MM-DD
    date_2 = datetime.strptime(date_2, '%Y-%m-%d') # Converts the string date_2 into a date time object using the format YYYY-MM-DD

    # Calculating the period between two days
    date_format = abs((date_2 - date_1).days) # Takes the absolute value of the difference between the two dates

    # Result
    if out == "float": # If the output is a float
        return float(date_format) # Return the date_float value
    elif out == "string": # If the output is a string
        return f"There are {date_format} days between the two dates" # Return the date_float value with additional text
    else:
        raise ValueError("Invalid Entry. Please input either 'float' or 'string' as the keyword.") # If it's neither, produce an error

# Test values
print(time_diff('2024-04-02', '2024-04-06', 'float'))  # Should return "4.0"
print(time_diff('2024-04-02', '2024-04-06', 'string'))  # Should return "There are 4 days between the two dates"

# Exercise 4

def reverse(in_list: list) -> list: # Defines the reverse function that takes a list and returns a list

    """
    This function called 'reverse' takes a list and returns a list in reverse order.
    """
    return in_list[::-1] # Uses slicing to create a new list that is the reverse of the original list; it strts at the end of the list and moves backwards

def main():
    keywords = [] # Declares variable 'keywords' as a list
    while True:
        main_command = input("Please input a keyword: ") # Asks the user to input a keyword
        keywords.append(main_command) # Adds the inputted keyword to the 'keywords' list
        additional_keywords = str(input("Would you like to enter another keyword? (y/n): ")) # Asks the user to input another keyword, if desired
        if additional_keywords == "y": # If user says yes
            continue # Continue with the loop
        #if additional_keywords == "n": # If user says no
        else:
            reverse_keywords = reverse(keywords) # Calls the reverse function to reverse the 'keywords' list
            print(f"The original list is {keywords} and the reversed list is {reverse_keywords}.") # Prints out both the original and the reversed 'keywords' list
            break # Exits the loop
        
main()

# Exercise 5

def prob_k_heads(n: int, k: int) -> float:
    """
    This function takes natural numbers 'n' and 'k' with 'n>k' and returns the probability of getting 'k' heads from 'n' flips.
    """
    
    # Calculate the factorial of n
    factorial_n = 1
    for i in range(1, n + 1):
        factorial_n *= i

    # Calculate the factorial of k
    factorial_k = 1
    for i in range(1, k + 1):
        factorial_k *= i

    # Calculate the factorial of (n - k)
    factorial_n_k = 1
    for i in range(1, n - k + 1):
        factorial_n_k *= i

    # Calculate the binomial coefficient (n choose k)
    binomial_coefficient = factorial_n / (factorial_k * factorial_n_k)

    # Calculate the probability
    probability = binomial_coefficient * (0.5 ** n)

    return probability

def main():
    while True:
        n = int(input("Please input the number of coin flips desired: "))
        k = int(input("Please input the number of heads desired: "))
        if n >= k:
            break
        else:
            print("Please ensure that the number of coin flips is greater than or equal to the number of heads desired.")

    probability = prob_k_heads(n, k)
    print(f"The probability of getting {k} heads from {n} flips is {probability}")

main()

