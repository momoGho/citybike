package com.city.bike.spark.service

import com.typesafe.config.ConfigFactory
import org.apache.spark.sql.SparkSession


trait SparkSessionWrapper {

  //Parsing Spark args
  val conf = ConfigFactory.load()
  val (appName, mode) ={
    (conf.getString("city-bike.spark.app.name"),
    conf.getString("city-bike.spark.mode"))
  }

  // Init Spark Session
  lazy val spark: SparkSession = {
    SparkSession
      .builder()
      .master(mode)
      .appName(appName)
      .getOrCreate()
  }
}
