# Exercise 0

def github() -> str:
    """
    This function will return Bani Bedi's GitHub page for Problem Set 5.
    """

    return "https://github.com/banibedi/Econ481.git"

# Exercise 1

import requests # Makes HTTP requests in Python
from bs4 import BeautifulSoup # Python library that parses HTML and XML documents
import os # Will help read or write the file into a system

def scrape_code(url: str, filename: str) -> str: # Defines the 'scrape_code' function; it takes the URL (a string) and filename (a string), and indicates that the result will also be a string
    """
    The function 'scrape_code' takes a lecture's URL and a filename, then scrapes all the Python code,
    saves it to a file on the user's desktop, and handles errors that might occur during the process.
    """
    try:
        # Create a path to the user's desktop for file to save
        desktop = os.path.join(os.path.expanduser("~"), "Desktop") # Finds home directory of the current user; creates a path to Desktop folder
        full_path = os.path.join(desktop, filename) # Combines desktop name with filename; creates the full file path where scrapped code will be saved

        # Send a GET request to the URL
        response = requests.get(url) # Sends GET request to the specified URL and stores it in "response"
        response.raise_for_status()  # Checks for HTTP errors

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser') # Parses HTML

        # Find elements that contain Python code
        code_blocks = soup.find_all('pre')  # Finds all <pre? tags in HTML

        # Extract and compile code
        python_code = "" # Initializes string
        for block in code_blocks: # Loops over each block
            code = block.text.strip() # Extracts text from the block and strips any leading or trailing whitespace
            # Filter out relevant lines
            code_lines = [line for line in code.split('\n') if not line.startswith('%')] # Filters any lines that start with %
            filtered_code = '\n'.join(code_lines) # Joins the filtered lines back into a single string
            
            python_code += filtered_code + '\n\n' # Adds new lines for separation between code and blocks

        # Save the code to a Python file on the desktop
        with open(full_path, 'w') as file: # Opens the file and overwrites, or creates one if it doesn't exist
            file.write(python_code) # Writes the python code into the file

        return f"Successfully saved the code to {full_path}" # Returns a message that states that the code has been successfully saved in the file

    except Exception as e: # Calculates any exceptions that may have been raised in the try block
        return f"An error occurred: {e}" # Returns an error message

# Test URL
url = "https://lukashager.netlify.app/econ-481/01_intro_to_python"
filename = "scraped_code.py"
print(scrape_code(url, filename))
