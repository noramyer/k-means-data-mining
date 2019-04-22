import argparse
import numpy as np

#Author: Nora Myer
#Date: 4/10/19

args = ""
columns = {}
num_transactions = 0
num_items = 0

#Sets up and reads in args based in from the command line
def parse_data_file_args():
    global args

    #set arg labels
    args_labels = ["-database_file", "-k", "-max_iters", "-eps", "-output_file"]
    parser = argparse.ArgumentParser()

    #build the parser with labels
    for arg in args_labels:
        parser.add_argument(arg)

    #set global args
    args = parser.parse_args()

#K-means clustering algorithm based on parameters
def genKmeans(database, k, n, e, output_file):
    iteration = 0

    #select k points as the initial centroids
    centroids = assign_centroids(k, database)

    #while less than max number of iterations
    while iteration < n:
        oldcentroids = centroids
        print(iteration)
        #assign points to clusters
        assignments = assign_points(database, centroids)

        #update cluster centroids
        centroids = update_centroids(centroids, database, assignments)

        #if the new centroid and old centroid have converged now, break the while loop
        if converged(oldcentroids, centroids, e):
            break
            print("break")

        iteration += 1

    #assign points with final centroids
    assignments = assign_points(database, centroids)
    #print clusters to output file
    output_clusters(output_file, assignments)

#initialize centroids based on k-means ++ algorithm
#****BONUS****: initializing incrementally based on min distance proporional probilities from k++ algorithm
def assign_centroids(k, database):
    #begin by initializing one cluster to a random point in the database
    centroids = {}
    centroids[0] = database[np.random.randint(0, database.shape[0])]
    #then, for i < total number of clusters
    for i in range(1, k):
        distances = np.zeros([database.shape[0], len(centroids)])
        #get the distances to the existing centroids from each points
        for c in range(i):
            for row in range(database.shape[0]):
                #calculate eucliedian distance for each
                distances[row][c] = np.linalg.norm(database[row]-centroids[c])

        #get index of the minimal distance to a single cluster
        if i > 1:
            mins = np.argmin(distances, axis=1)
            d = [distances[x][mins[x]] for x in range(database.shape[0])]
        else:
            d = distances

        #get the proportion of the min distance to a cluster to the sum of all distances
        probabilities = d / np.sum(d)

        #get a random number from 0 - 1.0
        number = np.random.random()

        sum = 0.0
        index = 0
        #Treat the probailities matrix as a distribution
        #find where the random number falls in the distribution
        #and pick that index for the next centroid point.
        #This serves as picking a random point based on the weighted probailities
        for j in range(database.shape[0]):
            if number > sum and number < sum + probabilities[j]:
                index = j
                break
            sum = sum + probabilities[j]

        #assign new centroid to that point found
        centroids[i] = database[index]

    return centroids

#assign points based on centroid values
def assign_points(database, centroids):
    #initialize distance matrix
    distances = np.zeros([database.shape[0], len(centroids)])
    assignments = {}

    #for each centroid and each point, calcuate the euclidean distance
    for c in range(len(centroids)):
        for row in range(database.shape[0]):
            #calculate eucliedian distance for each
            distances[row][c] = np.linalg.norm(database[row]-centroids[c])

    #get index of the minimal distance for each row
    min_distances = np.argmin(distances, axis=1)

    #for each cluster, map assigned data points
    for row in range(database.shape[0]):
        if not min_distances[row] in assignments:
            assignments[min_distances[row]] = set()
        assignments[min_distances[row]].add(row)

    #return assigned points in the form of a map
    return assignments

#update centroid values based on previous point assignments
def update_centroids(centroids, database, assignments):
    newcentroids = {}
    #for each cluster, get the mean of assigned points
    for index in range(len(centroids)):
        if index in assignments:
            #get points assigned to that cluster
            rows = assignments[index]
            clustered = np.take(database, list(rows), axis=0)

            #assign the centroid to the average of assigned points based on column values
            newcentroids[index] = np.mean(clustered, axis=0)
        else:
            #if points have no been assigned to that cluster this round, dont update centroid
            newcentroids[index] = centroids[index]
    return newcentroids

#print clustered point to output file
def output_clusters(output_file, assignments):
    #open file
    f = open(output_file, "w+")

    #for each cluster, print the assigned point indices
    for cluster in assignments.keys():
        f.write(str(cluster) + ": " + str(' '.join(str(x) for x in sorted(assignments[cluster]))) + "\n")

#Read in database from args input file
def read_database(database_file):
    #load data into numpy matrix
    input = np.loadtxt(database_file, dtype='f')

    return input

#given old and new centroid values, find if they have converged based on epsilon
def converged(centroids, newcentroids, e):
    sum = 0
    #look if points have converged based on distance
    for i in centroids.keys():
        sum += np.sum(np.linalg.norm(centroids[i] - newcentroids[i], axis = 0))

    #if sum is less than epsilon, converged
    if sum < e:
        print("Converged")
        return True;

    #otherwise, it has not converged
    return False

#main function
def main():
    #get arg parser
    parse_data_file_args()
    #get database
    database = read_database(str(args.database_file))
    genKmeans(database, int(args.k), int(args.max_iters), float(args.eps), str(args.output_file))

if __name__ == "__main__":
    main()
