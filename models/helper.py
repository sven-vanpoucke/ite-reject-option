from datetime import datetime


def helper_output():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'results_{timestamp}.txt'
    folder_path = 'output/'
    file_path = folder_path + filename

    with open(file_path, 'a') as file:
        file.write(f"CHAPTER 1: INIT\n\n")
        file.write(f"This file has been generated on: {timestamp}\n\n")

    return timestamp, filename, file_path

