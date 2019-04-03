# nearest-neighbour-classifier

This is an implementation of the K nearest neighbours algorithm where K is set to 1 by default (thus just anearest Neighbour!).
Beyond the actual algorithm, use the script to make predictions after training the model on any (numerical, and labeled) dataset you have!

# Try It
1. Clone the repository 
2. Move to the directory 
2. Make sure you have the virtual environemnt running 
3. Run the script (script.py)

```bash 
git clone https://github.com/simarpreet-singh/nearest-neighbour-classifier.git
cd nearest-neighbour-classifier 
source oneenv/bin/activate
python script.py
```

To use a sample data, I've included the iris dataset as a csv where the species have been labeled as such:
* 0 = Setosa 
* 1 = Versicolor
* 2 = Virginica 

Note: (1) The filename for the dataset is iris.csv and (2) the labels column is called species.
