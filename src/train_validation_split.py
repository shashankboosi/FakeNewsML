class DataSplit:

    def __init__(self, ids, headline, split_size=0.8):
        self.ids = ids
        self.headline = headline
        self.training_split_size = split_size

    def split(self):
        train_ids = self.ids[:int(self.training_split_size * len(self.ids))]
        validation_ids = self.ids[int(self.training_split_size * len(self.ids)):]
        train_stances = []
        validation_stances = []
        for stance in self.headline:
            if int(stance['Body ID']) in train_ids:
                train_stances.append(stance)
            elif int(stance['Body ID']) in validation_ids:
                validation_stances.append(stance)

        return train_stances, validation_stances
