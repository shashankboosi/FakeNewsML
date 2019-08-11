import pickle
import csv


def output_file(obj, filename):
    f = open(filename, 'wb')
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def input_file(filename):
    f = open(filename, "rb")
    return pickle.load(f)


def write_to_csv(filename, heading, content):
    """
    Function to write desired output to csv file
    :param filename: the name of the file
    :param heading: a Python list of the headers for the csv ['Headline','Body ID','Stance']
    :param content: a Python list of lists where each list has the same structure as the header
    :return: created csv filepath
    """
    with open(filename, mode='w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(heading)
        for row in content:
            csv_writer.writerow(row)
    print(filename, " created")
    return filename


def count_stances(data):
    agree = 0
    disagree = 0
    discuss = 0
    unrelated = 0

    for stance in data:
        if stance == "agree":
            agree += 1
        elif stance == "disagree":
            disagree += 1
        elif stance == "discuss":
            discuss += 1
        else:
            unrelated += 1

    print("Agree", agree)
    print("Disagree", disagree)
    print("Discuss", discuss)
    print("Unrelated", unrelated)
