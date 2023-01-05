//> using scala "2.13"
//> using lib "org.apache.spark::spark-core:3.3.1"
//> using lib "org.apache.spark::spark-sql:3.3.1"
//> using lib "org.apache.spark::spark-mllib:3.3.1"

// Java17だとうまく動かないことがあるので11で動かすこと
// https://stackoverflow.com/questions/73465937/apache-spark-3-3-0-breaks-on-java-17-with-cannot-access-class-sun-nio-ch-direct

import org.apache.spark.sql.SparkSession

println("Apache Spark Regression Example")

// Scikit-learn、Keras、TensorFlowによる実践機械学習第2版
// pp.23 の線形モデルの実装をApache Sparkで実装してみる

// SparkSessionをまず作成する必要がある。ドキュメントでspark.という表記が登場した場合はこのSparkSessionのことを指している。
val spark = SparkSession
  .builder()
  .appName("Spark-Exercise-Regression")
  .config("spark.master", "local") // 実行するマスターノードを指定するのが必須なのでlocalとする
  .getOrCreate()

// https://homl.info/4
val oecdBli = spark.read
  .format("csv")
  .option("header", "true")
  .load("BLI_03012023082923436.csv")
// https://homl.info/5
val gdpPerCapita =
  spark.read.format("csv").option("header", "true").load("WEOOct2022all.csv")

// oecdBliを'Life satisfaction' かつ 男女を絞り込まない総合点数で絞り込み、 "LOCATION" カラムと "Value" カラムを残す。
val oecdBliFiltered = oecdBli
  .filter("`INDICATOR2` == 'SW_LIFS' AND `INEQUALITY6` = 'TOT'")
  .select("LOCATION", "Value")

// そして WEOは "WEO Subject Code" == "NGDP" でフィルタし、"ISO" カラムと 年のカラムとを残す。
val year = "2015"
val gdp = "NGDPDPC" // GDP per capita, constant price, dollar
val gdpPerCapitaFiltered = gdpPerCapita
  .filter(s"`WEO Subject Code` == '${gdp}'")
  .select("ISO", year, "WEO Subject Code")

// 最後に "Location" == "ISO" でInner JOINする。
import org.apache.spark.sql.Column
val joined = oecdBliFiltered
  .join(
    gdpPerCapitaFiltered,
    oecdBliFiltered.col("Location") === gdpPerCapitaFiltered.col("ISO"),
    "inner"
  )
  .select("ISO", "Value", year)
  .withColumnRenamed("Value", "Satisfaction")
  .withColumnRenamed(year, gdp)
  .withColumns(
    Map(
      "Satisfaction" -> new Column("Satisfaction").cast("double"),
      gdp -> new Column(gdp).cast("double")
    )
  )

joined.show()

// joined.drop("ISO").select("Satisfaction", gdp).write.option("header", true).csv("./result.csv")

// Linear RegressionがFeaturesのためにVectorを要求するので、VectorAssemblerでVectorにカラムを変換する
import org.apache.spark.ml.feature.VectorAssembler
val va = new VectorAssembler().setInputCols(Array(gdp)).setOutputCol("NGDPVec")

import org.apache.spark.ml.regression.LinearRegression
val lr = new LinearRegression()
  .setMaxIter(50)
  .setRegParam(0.1)
  .setFeaturesCol("NGDPVec")
  .setLabelCol("Satisfaction")
val model = lr.fit(va.transform(joined))
val coeffs = model.coefficients.toArray.mkString("[", ", ", "]")
println(s"intercept: ${model.intercept}, coefficients: ${coeffs}")

spark.stop() // 必須
