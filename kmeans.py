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

#initialize centroids based on randomness, std, and mean
def assign_centroids(k, database):
    #for each cluster k, initialize centroid
    centroids = {
        i: np.random.randn(k, database.shape[1]) * np.std(database, axis = 0)
        for i in range(k)
    }
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
