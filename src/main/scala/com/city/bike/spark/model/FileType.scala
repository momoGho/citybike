package com.city.bike.spark.model

object FileType extends Enumeration {
  type FileType = Value
  val JSON, XML, CSV, TXT = Value
}
