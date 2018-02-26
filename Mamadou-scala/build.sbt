name := "mamadou in scala"


fork in run := true
scalaVersion := "2.11.8"
val sparkVersion = "2.1.0"

logLevel := Level.Warn

val spark = Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-streaming" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-streaming-kafka-0-10" % sparkVersion)


val kafka = Seq("net.cakesolutions" %% "scala-kafka-client" % "1.0.0")

libraryDependencies ++= spark ++ kafka
mainClass in (Compile, run) := Some("SparkStreamingClassifier")