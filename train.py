import pickle

with open('./data/scitechnews.pkl', 'rb') as file:
    dataset =  pickle.load(file, encoding="latin1")
    
print(dataset)