package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.evaluation.MulticlassMetrics



object Trainer {

  def main(args: Array[String]): Unit = {


    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()
    import spark.implicits._

    /****************************************************************************************************************
      *                                                                                                             *
      *       TP 3                                                          Implémentation de Matthieu ROUSSEL      *
      *                                                                                                             *
      *       - lire le fichier sauvegarder précédemment                                                            *
      *       - construire les Stages du pipeline, puis les assembler                                               *
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search          *
      *       - Sauvegarder le pipeline entraîné                                                                    *
      *                                                                                                             *
      ***************************************************************************************************************/

    /* IMPORTANT : à mettre à jour avant de lancer sur votre machine */
    // Chemin d'accès aus données
    val strPathData = "/Users/matthieuroussel/OneDrive/Documents/SauvegardeMac/Documents/Telecom Paristech Enseignements/INF729 - Introduction Framework Hadoop/TP Spark/TP3/prepared_trainingset"
    // Chemin d'accès au répertoire de sauvegarde du modèle
    val strModele = """/Users/matthieuroussel/OneDrive/Documents/SauvegardeMac/Documents/Telecom Paristech Enseignements/INF729 - Introduction Framework Hadoop/TP Spark/TP3/LogisticRegressionModel\"""

    println("Démarrage du pipeline")

    /** 1 - CHARGEMENT DES DONNEES **/

    // a) Charger un csv dans dataframe
    val data_df = spark.read.parquet(strPathData)

    // Vue SQL pour faciliter l'exploration des données (étapes non conservées dans cette version finale)
    data_df.createOrReplaceTempView("data_df")
    // Intermédiaire, pour raccourcir les temps de traitement
    //val data_df = spark.sql("SELECT * FROM data_df LIMIT 1000")
    //data_df.select("text").show(30)

    // Nombre de lignes et colonnes
    println(s"Total number of rows: ${data_df.count}")
    println(s"Number of columns ${data_df.columns.length}")
    // Observation du df
    data_df.show(10)
    // Schema du df
    data_df.printSchema()

    /** 2 -	Utiliser les données textuelles **/
    // Stage 0
    // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")
    // Stage 1
    // Enlever les stop words depuis la liste disponible en anglais
    val stopWords= StopWordsRemover.loadDefaultStopWords("english")
    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("text_cleaned")
      .setStopWords(stopWords)
    // Stage 2
    // Vexctoriser
    val cvm = new CountVectorizer()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("text_cleaned_vect")
    // Stage 3
    // Transformation TFIDF
    val idf = new IDF()
      .setInputCol(cvm.getOutputCol)
      .setOutputCol("text_idf")

    /** 3 -	Convertir les catégories en données numériques **/
    // Stage 4
    // Convertir en numérique la variable country
    val indexer1 = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
    // Stage 5
    // Convertir en numérique la variable currency
    val indexer2 = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
    // Stage 6
    // Convertir les variables catégorielles indexées numériquement en flag (variable country)
    val encoder1 = new OneHotEncoder()
      .setInputCol("country_indexed")
      .setOutputCol("countryVec")
    // Stage 7
    // Convertir les variables catégorielles indexées numériquement en flag (variable currency)
    val encoder2 = new OneHotEncoder()
      .setInputCol("currency_indexed")
      .setOutputCol("currencyVec")

    /** 4 -  Mettre les données sous une forme utilisable par Spark.ML **/

    // Stage 8
    // Vectorizer les features avant modélisation
    val assembler = new VectorAssembler()
      .setInputCols(Array("text_idf", "days_campaign", "hours_prepa","goal","countryVec","currencyVec"))
      .setOutputCol("features")
    // Stage 9
    // Model definition : régression logistique
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    // Création du pipeline avec l'ensemble des étapes
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, cvm, idf, indexer1,indexer2,encoder1,encoder2,assembler,lr))

    /** 5 - Entraînement et tuning du modèle **/
    // Splitter les données en Training Set et Test Set
    val splits = data_df.randomSplit(Array(0.9, 0.1), seed=1)
    val (trainingData, testData) = (splits(0), splits(1))

    // Définition de grille pour la "validation croisée"
    //1.  minDF = stage countVectorizer a un paramètre “minDF” qui permet de ne prendre que
    // les mots apparaissant dans au moins le nombre spécifié par minDF de documents
    // de 55 à 95 en pas de 20
    //2. regParam = paramètre de régularisation de la régression logistique
    val paramGrid = new ParamGridBuilder() // No parameter search
      .addGrid(cvm.minDF, Array(55.0,75.0,95.0))
      .addGrid(lr.regParam, Array(10e-2, 10e-4,10e-6,10e-8))
      .build()
    // Définition d'un evaluateur et de la métrique associée (f1)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      // "f1" (default), "weightedPrecision", "weightedRecall", "accuracy"
      .setMetricName("f1")
    // Modele de split train /  validation ("cv light")
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      // 70% des données pour le train, 30% pour la validation.
      .setTrainRatio(0.7)
    // Lancement de l'évaluation des modélisations avec l'esnemble des paramètres de la grille
    val model = trainValidationSplit.fit(trainingData)

    // Application sur l'échantillon de test des prévisions en utilisant le meilleur modèle
    val predictions =  model.transform(testData)
    // Affichage du score f1 obtenu sur l'échantillon de test
    val score_f1 = evaluator.evaluate(predictions)
    println("Test Error = " + (score_f1))
    // Affichage de la Matrice de confusion sur le test
    val predictions_rdd = predictions.select("final_status", "predictions").as[(Double, Double)].rdd
    val metrics = new MulticlassMetrics(predictions_rdd);
    println("Confusion Matrix:")
    println(metrics.confusionMatrix)
    // Récupération des paramètres optimaux
    // Récupération du pipeline correspondant
    val bestPipelineModel = model.bestModel.asInstanceOf[PipelineModel]
    // Récupération des étapes associées
    val stages = bestPipelineModel.stages
    // Récupération du paraètre regParam de l'étape de modélisation (régression logistique)
    val lrStage = stages(9).asInstanceOf[LogisticRegressionModel]
    println("regParam = " + lrStage.getRegParam)
    // Récupération du paramètre minDF pour l'étape de comptage des mots
    val cvmStage = stages(2).asInstanceOf[CountVectorizerModel]
    println("minDF = " + cvmStage.getMinDF)

    // Sauvegarde du modèle de régression logistique uniquement (on aurait pu sauvegarder ll'ensemble du popie du meilleur modèle)
    lrStage.write.overwrite().save(strModele)
    // Réaffichage de la matrice de confusion, par un autre moyen, plus simple
    println(predictions.groupBy("final_status", "predictions").count.show())
  }
}
