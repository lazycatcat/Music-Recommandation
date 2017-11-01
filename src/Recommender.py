import findspark
findspark.init('/usr/local/bin/spark-1.3.1-bin-hadoop2.6')

from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

print "good"

# Helper functions
# artist alias 
def getArtistAlias(line):
    data = line.split()
    try:
        return [int(data[0]), int(data[1])]
    except:
        return None

# user artist data
def mapRatingData(line):
    data = line.split()
    artistID = int(data[1])
    ArtistAlias = bArtistAlias.value
    if artistID in ArtistAlias:
        artistID = ArtistAlias[artistID]
    return Rating(int(data[0]),artistID,int(data[2]))
	
# Helper function for artist data
def getArtistData(line):
    data = line.split(None,1)
    try:
        return [int(data[0]), data[1].strip(' ')]
    except:
        return None
        
# Initilize spark
conf = SparkConf().setAppName('RecommendationTrainer').setMaster('local')
sc = SparkContext(conf=conf)


# Preprocess data
rawUserArtistData = sc.textFile('user_artist_data.txt')

rawArtistData = sc.textFile('artist_data.txt')
artistByID = rawArtistData.map(lambda line: getArtistData(line)).filter(lambda line: line is not None).cache()

rawArtistAlias = sc.textFile('artist_alias.txt')
artistAlias = rawArtistAlias.map(lambda line: getArtistAlias(line)).filter(lambda line: line is not None).collectAsMap()


# Setup Model
bArtistAlias = sc.broadcast(artistAlias)
trainData = rawUserArtistData.map(lambda line: mapRatingData(line)).cache()

model = ALS.trainImplicit(trainData,rank = 10, iterations = 5, lambda_ = 0.01, alpha = 1.0)

# First user feature vector
vector = model.userFeatures().take(1)[0]

# top 10 recommendation
recommendations = model.call("recommendProducts", 2093760, 10) 

ProductIDs = []
recommendation_results = []
for r in recommendations:
    ProductIDs.append(r[1])
    recommendation_results.append("Product: " + str(r[1]) + "  rating: " + str(r[2]))

# map to musician names
products = artistByID.filter(lambda line: line[0] in ProductIDs)
name_results = []
for p in products.collect(): 
    name_results.append("Id: " + str(p[0]) + " Name: " + str(p[1]))

# print result
print "Feature vector of the first user: "
print vector
print "------------------------------------------------"
print "Top 10 recommendation for user 2093760: "
for result in recommendation_results:
    print result
print "------------------------------------------------"
print "Names of musicians"
for name in name_results:
    print name
print "------------------------------------------------"