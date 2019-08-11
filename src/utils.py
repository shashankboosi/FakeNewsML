import pickle
import csv


def output_file(obj, filename):
    f = open(filename, 'wb')
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def input_file(filename):
    f = open(filename, "rb")
    return pickle.load(f)


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


def write_to_csv(filename, content):
    with open(filename, mode='w') as file:
        csv_writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        csv_writer.writerow(content)
    print(filename, " created")
    file.close()
    return filename


def read_from_csv(filename):
    with open(filename, mode='r') as file:
        csv_reader = csv.reader(file)
        results = list(csv_reader)

    file.close()
    return results[0]
