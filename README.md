# k-means-data-mining
#### Nora Myer
#### April 2019

### Dependencies
- Runs with python 2.7 or python 3.7.x
- numpy compatible with python 2.7 or python 3.7.x
- tested on Mac and Linux environments

### Running the program
To run the k-means clustering algorithm, use the command
```
python kmeans.py -database_file database.txt -k 5 -max_iters 30 -eps .3 -output_file output.txt
```

### Data Representation
To represent the data, I used a numpy matrix where each row represented one data point in the database and the columns represented features of the data. This made is easy to access values within the matrix and it was the simplest and easiest way to represent the data. Since I needed to do a lot of operations based on rows, columns, column averages, etc. using numpy was a good and efficient decision.

### Output File
The final cluster assignments can be found in the output file based on the file parameter, which based on the command above is output.txt. The assignments are of the form
```
0: 0 4 6 7 8 9
1: 1 2 5 10 12
2: 3 11 13
```
in C-style indexing (numerical sorted order) where the index of the cluster points to all the indices of the points assigned to that cluster.

### **Bonus**
For the bonus, I initialized the centroids based on the k-means++ algorithm. This means I selected a point at random to be the first cluster, and then until I reached the desired number of clusters, I found the probability proportion based on the min distance from the current centroids already initialized and chose a point to become the next cluster based off those weighted probabilities. This is an improvement on randomly generating centroids initially.
