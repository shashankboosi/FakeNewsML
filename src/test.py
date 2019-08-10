import nltk, re
import contractions

lemmatizer = nltk.WordNetLemmatizer()

'''
def preprocess(string):
    # to lowercase, non-alphanumeric removal
    step1 = re.findall(r'[^\w\s]', '', string)
    #step1 = " ".join(re.findall(r'\w+', string)).lower()
    # step2 = [lemmatizer.lemmatize(t).lower() for t in nltk.word_tokenize(step1)]

    return step1
'''
s = ['Last', 'week', 'we', 'hinted', 'at', 'what', 'was', 'to', 'come', 'as', 'Ebola', '78698', '90', 'fears', 'spread',
     'across', 'America', '.', 'Today', ',', 'we', 'get', 'confirmation', '.', 'As', 'The', 'Daily', 'Caller',
     'reports', ',']
ss = "Last week we hinted at what was to come as Ebola 78698 90 fears spread across America. Today, we get confirmation. As The Daily Caller reports, one passenger at Dulles International Airport outside Washington, D.C. is apparently not taking any chances. A female passenger dressed in a hazmat suit - complete with a full body gown, mask and gloves - was spotted Wednesday waiting for a flight at the airport. Source: The Daily Caller We particularly liked the JCPenney bag - maybe that's a new business line for the bankrupt retailer... *  *  * On a side note, try Halloween stores if you need a Haz-Mat suit in a hurry..."


sh = "Police find mass graves with at least '15 bodies' near Mexico town where 43 students disappeared after"
print(preprocess(sh))
