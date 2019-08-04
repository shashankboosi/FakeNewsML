from csv import DictReader

'''
This function returns the instances and the article bodies
Input: The stance path and the body path

Output: 'stance' which is an ordered 2 dimensional dictionary 
        Example: To retrieve the first headline use
                self.stance[0]['headline']
        'articleBody' which is a 1 dimensional dictionary
        Example: To retrieve the article body with ID 500 
                self.articleBody[500]
'''
class FakeNewsData:

    def __init__(self, stancesPath, bodiesPath):
        self.stancesPath = stancesPath
        self.bodiesPath = bodiesPath
        self.stances = self.read(self.stancesPath)
        bodies = self.read(self.bodiesPath)
        self.articleBody = {}

        for body in bodies:
            self.articleBody[int(body['Body ID'])] = body['articleBody']

    '''
    This function reads the csv file and stores it in a list
    '''
    def read(self, path):
        row = []
        with open(path, 'r', encoding='utf-8') as csvTable:
            csvLine = DictReader(csvTable)
            for line in csvLine:
                row.append(line)

        return row

