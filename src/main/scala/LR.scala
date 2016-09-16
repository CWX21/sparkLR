// General purpose library
import scala.xml._

// Spark data manipulation libraries
import org.apache.spark.sql.catalyst.plans._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

// Spark machine learning libraries
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object LR {
	def main(args: Array[String]) {
		val conf = new SparkConf().setAppName("Simple Application").setMaster("local")
				val sc = new SparkContext(conf)
		val fileName = "/home/hadoop/data/sparkdata/Posts.small.xml"
		val textFile = sc.textFile(fileName)
		val postsXml = textFile.map(_.trim).
		filter(!_.startsWith("<?xml version=")).
		filter(_ != "<posts>").
		filter(_ != "</posts>")
		val postsRDD = postsXml.map { s =>
		val xml = XML.loadString(s)

		val id = (xml \ "@Id").text
		val tags = (xml \ "@Tags").text

		val title = (xml \ "@Title").text
		val body = (xml \ "@Body").text
		val bodyPlain = ("<\\S+>".r).replaceAllIn(body, " ")
		val text = (title + " " + bodyPlain).replaceAll("\n", 
				" ").replaceAll("( )+", " ");

		println(id + "cwx")
		Row(id, tags, text)
		}
		val schemaString = "Id Tags Text"
				val schema = StructType(
						schemaString.split(" ").map(fieldName => 
						StructField(fieldName, StringType, true)))
						
            val sqlContext = new SQLContext(sc)
						val postsDf = sqlContext.createDataFrame(postsRDD, schema)
						sc.stop   
	}
}
