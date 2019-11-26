package com.city.bike.spark.start

import com.city.bike.spark.model.FileType
import com.city.bike.spark.service.SparkSessionWrapper
import org.apache.spark.mllib.linalg.Vectors
import org.apache.log4j.Logger
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.sql.SaveMode

object Main extends App with SparkSessionWrapper {

  val logger = Logger.getLogger(this.getClass.getName)

  logger.trace("Start Application City Bike Clustering!")

  //Parsing application config
  val (inputPath, outputPath, fileType, numIterations, maxNbCluster) = {
    (conf.getString("city-bike.data.input.path"),
      conf.getString("city-bike.data.output.path"),
      FileType.withName(conf.getString("city-bike.data.file.type")),
      conf.getInt("city-bike.cluster.num.iterations"),
      conf.getInt("city-bike.cluster.max.nb.cluster")
    )
  }
  //Set Spark Log to WARN to avoid a lot of info log
  spark.sparkContext.setLogLevel("WARN")

  //Load json file
  val cityBikeDF = if (FileType.JSON.eq(fileType)) {
    logger.info(s"Start loading the file ${inputPath}")
    spark.read.option("multiLine", true).json(inputPath)
  } else {
    logger.error(s"The file ${inputPath} is not a JSON file please change the inputPath in application.conf")
    sys.exit(-1)
  }

  // Check unicity of ID
  val checkIdUnicity = cityBikeDF.select(cityBikeDF.col("id")).rdd.countByValue().filter(_._2 > 1)
  if (!checkIdUnicity.isEmpty) {
    logger.warn(s"There is some ids are not unique in the file ${checkIdUnicity.toString()}")
    checkIdUnicity foreach println
  }

  // Filter and normalize dataset by taking Latitude and Longitude
  val cityBikeRddPair = cityBikeDF.rdd.map(r => {
    val latitude = r.getString(r.fieldIndex("latitude"))
    val longitude = r.getString(r.fieldIndex("longitude"))
    val coordinates = r.getStruct(r.fieldIndex("coordinates"))
    val id = r.getDouble(r.fieldIndex("id"))

    //Regex to take just double number
    val regex = "[-+]?[0-9]*\\.?[0-9]*".r

    val latResult = if (latitude != null && regex.pattern.matcher(latitude).matches) {
      latitude.toDouble
    } else if (coordinates != null && !coordinates.isNullAt(0) && regex.pattern.matcher(coordinates.getDouble(0).toString).matches) coordinates.getDouble(0)
    else Double.NaN
    val longResult = if (longitude != null && regex.pattern.matcher(longitude).matches) {
      longitude.toDouble
    } else if (coordinates != null && !coordinates.isNullAt(1) && regex.pattern.matcher(coordinates.getDouble(1).toString).matches) coordinates.getDouble(1)
    else Double.NaN

    (id, (latResult, longResult))
  })

  //Drop NaN latitude and NaN longitude from dataset
  val cityBikeRddClean = cityBikeRddPair.filter(row => !(row._2._1.isNaN || row._2._2.isNaN))

  //Transform dataset to a vector of (latitude, longitude)
  val cityBikeVectors = cityBikeRddClean.map(row => Vectors.dense(row._2._1, row._2._2))

  // Find the optimal k cluster between 2 and maxNbCluster
  var nbCluster = 2
  var res = -1d

  while (nbCluster <= maxNbCluster && res != 0d) {
    val model = KMeans.train(cityBikeVectors, nbCluster, numIterations)
    val WSSSE = model.computeCost(cityBikeVectors)
    res = BigDecimal(WSSSE).setScale(2, BigDecimal.RoundingMode.DOWN).toDouble
    nbCluster += 1
  }

  nbCluster = if (nbCluster > 3) nbCluster - 2 else if (nbCluster == 3) nbCluster - 1 else nbCluster
  logger.info(s"the optimal number of cluster for this dataset is ${nbCluster}")

  //Start using K-Means clustering on our dataset by using a optimal nb cluster
  logger.info(s"Start using K-Means clustering on our dataset by using those parameters : optimal nb cluster ${nbCluster} and nb iteration ${numIterations}")
  val clusters = KMeans.train(cityBikeVectors, nbCluster, numIterations)
  //clusters.clusterCenters.foreach(println)

  //Build a dataset with clustering result
  val cityBikeClusetring = cityBikeRddClean.map(row => (row._1, row._2._1, row._2._2, clusters.predict(Vectors.dense(row._2._1, row._2._2))))
  val columns = Seq("Id", "Latitude", "Longitude", "Cluster")
  val dfFromRDDCityBike = spark.createDataFrame(cityBikeClusetring).toDF(columns: _*)

  dfFromRDDCityBike.write.mode(SaveMode.Overwrite).csv(outputPath)

  spark.stop()
  logger.trace("the end of Application City Bike Clustering!")
}
