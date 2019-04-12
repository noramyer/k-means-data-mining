import argparse
import numpy as np

#Author: Nora Myer
#Date: 3/26/19

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
    return None
    iteration = 0

    #select k points as the initial centroids
    centroids = assign_centroids(k, database)

    while iteration < n or converged():
        #assign points to clusters
        assignments = assign_points(database, centroids)
        print("Assignments = " + str(assignments))

        #update cluster centroids
        centroids = update_centroids()
        iteration += 1

    assignments = assign_points(database, centroids)
    output_clusters(output_file, assignments)

def assign_centroids(k, database):
    #for each cluster k, initialize centroid to be the mean of the columns of all the rows
    centroids = {
        i: np.mean(database, axis=0)
        for i in range(k)
    }
    return centroids

def assign_points(database, centroids):
    distances = np.zeros([database.shape[1], len(centroids)])
    assignments = {}
    for c in range(len(centroids)):
        for row in database:
            distances[row][c] = linalg.norm(database[row]-curmeans[c], axis = 1)

    min_distances = argmin(distances, axis=1)
    print(min_distances)


def update_centroids(centroids, database, assignments):
    for index in range(len(centroids)):
        rows = assignments[index]
        clustered = np.take(database, rows, axis=0)
        centroids[index] = np.means(clustered, axis=0)

def output_clusters(output_file, assignments):
    f = open(output_file, "w+")
    for cluster in assignments.keys():
        f.write(str(cluster) + ": " + str(' '.join(sorted(assignments[cluster]))) + "\n")

#Read in database from args input file
def read_database(database_file):
    input = np.loadtxt(database_file, dtype='f')

    return input

def converged():
    return False

#main function
def main():
    #get arg parser
    parse_data_file_args()
    #get database
    database = read_database(str(args.database_file))

    assignments = assign_centroids(int(args.k), database)
    print(len(assignments[0]))



    #x = np.arange(20).reshape(5, 4)
    #rows = [1, 3]
    ##print(x)
    #print("\n")
    #print(np.take(x, rows, axis=0))


if __name__ == "__main__":
    main()
