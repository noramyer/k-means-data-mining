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

def genKmeans(database, k, n, e, output_file):
    iteration = 0

    #select k points as the initial centroids
    centroids = assign_centroids(k, database)
    newcentroids = centroids

    while iteration < n and not converged(centroids, newcentroids, e):
        #assign points to clusters
        centroids = newcentroids
        assignments = assign_points(database, centroids)

        #update cluster centroids
        update_centroids(centroids, database, assignments)
        newcentroids = centroids
        iteration += 1

    assignments = assign_points(database, centroids)
    output_clusters(output_file, assignments)

def assign_centroids(k, database):
    #for each cluster k, initialize centroid to be the mean of the columns of all the rows
    centroids = {
        i: np.random.randn(k, database.shape[1]) + np.mean(database,axis = 0)
        for i in range(k)
    }
    return centroids

def assign_points(database, centroids):
    distances = np.zeros([database.shape[0], len(centroids)])
    assignments = {}
    for c in range(len(centroids)):
        for row in range(database.shape[0]):
            distances[row][c] = np.linalg.norm(database[row]-centroids[c])

    min_distances = np.argmin(distances, axis=1)

    for row in range(database.shape[0]):
        if not min_distances[row] in assignments:
            assignments[min_distances[row]] = set()
        assignments[min_distances[row]].add(row)

    return assignments

def update_centroids(centroids, database, assignments):
    for index in range(len(centroids)):
        if index in assignments:
            rows = assignments[index]
            clustered = np.take(database, list(rows), axis=0)
            centroids[index] = np.mean(clustered, axis=0)

def output_clusters(output_file, assignments):
    f = open(output_file, "w+")
    for cluster in assignments.keys():
        f.write(str(cluster) + ": " + str(' '.join(str(x) for x in sorted(assignments[cluster]))) + "\n")

#Read in database from args input file
def read_database(database_file):
    input = np.loadtxt(database_file, dtype='f')

    return input

def converged(centroids, newcentroids, e):
    dist = sum(linalg.norm(means - newmeans, axis = 0))
    if dist < e:
        return True;

    return False

#main function
def main():
    #get arg parser
    parse_data_file_args()
    #get database
    database = read_database(str(args.database_file))
    genKmeans(database, int(args.k), int(args.max_iters), float(args.eps), str(args.output_file))
    #centroids = assign_centroids(int(args.k), database)
    #assignments = assign_points(database, centroids)



if __name__ == "__main__":
    main()
