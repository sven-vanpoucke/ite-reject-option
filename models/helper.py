from datetime import datetime


def helper_output(folder_path='output/'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'results_{timestamp}.txt'
    file_path = folder_path + filename

    with open(file_path, 'a') as file:
        file.write(f"CHAPTER 1: INIT\n\n")
        file.write(f"This file has been generated on: {timestamp}\n\n")

    return timestamp, filename, file_path


def improvement(old_value, new_value):
    old_value = float(old_value)
    new_value = float(new_value)

    improvement = ((new_value-old_value)/new_value*100)# .round(2)
    return improvement