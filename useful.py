
# coding: utf-8

# In[25]:


import findspark
findspark.init('/home/ubuntu/spark-2.1.1-bin-hadoop2.7')
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('basics').getOrCreate()


# In[26]:


df = spark.read.csv('Book3.csv', inferSchema=True, header=True)

# Let's see the data.
df.show()

# And the data schema.
df.printSchema()


# In[27]:


df.head(1)


# In[28]:


from pyspark.sql.functions import col, when
df = df.select(col("Time"),col("air_temperature"),
col("dewpoint"),col("wind_direction"),col("`PM2.5`").alias("PM25"),
col("wind_speed"), col("ambient_pressure"),
col("PM10"))
df.describe().show()


# In[29]:


df= df.withColumn('air_temperature', when(col('air_temperature') != 'NULL',col('air_temperature')))
df= df.withColumn('dewpoint', when(col('dewpoint') != 'NULL',col('dewpoint')))
df= df.withColumn('wind_direction', when(col('wind_direction') != 'NULL',col('wind_direction')))
df= df.withColumn('wind_speed', when(col('wind_speed') != 'NULL',col('wind_speed')))
df= df.withColumn('PM25', when(col('PM25') != 'NULL',col('PM25')))
df.show()


# In[30]:


from pyspark.sql.functions import when, count, col

df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()


# In[31]:


from pyspark.sql.types import FloatType
df = df.withColumn("air_temperature",df["air_temperature"].cast(FloatType()))
df = df.withColumn("dewpoint",df["dewpoint"].cast(FloatType()))
df = df.withColumn("wind_direction",df["wind_direction"].cast(FloatType()))
df = df.withColumn("wind_speed",df["wind_speed"].cast(FloatType()))

df.printSchema() 


# In[32]:


from pyspark.sql.functions import mean

# Let's collect the average. You'll notice that the collection returns the average in an interesting format.
mean_sales1 = df.select(mean(df['air_temperature'])).collect()
mean_sales2 = df.select(mean(df['dewpoint'])).collect()
mean_sales3 = df.select(mean(df['wind_direction'])).collect()
mean_sales4 = df.select(mean(df['wind_speed'])).collect()


# In[33]:


mean_sales1[0][0]
mean_sales_val1 = mean_sales1[0][0]
mean_sales2[0][0]
mean_sales_val2 = mean_sales2[0][0]
mean_sales3[0][0]
mean_sales_val3 = mean_sales3[0][0]
mean_sales4[0][0]
mean_sales_val4 = mean_sales4[0][0]


# In[34]:


df = df.na.fill(mean_sales_val1, subset=['air_temperature'])
df = df.na.fill(mean_sales_val2, subset=['dewpoint'])
df = df.na.fill(mean_sales_val3, subset=['wind_direction'])
df = df.na.fill(mean_sales_val4, subset=['wind_speed'])


# In[35]:


df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()


# In[36]:


from pyspark.sql.functions import substring
df = df.withColumn("Year", substring(df.Time, 7,4))

df.show()


# In[37]:




df = df.drop("Time")
df = df.drop("PM10")
df. show()


# In[38]:


df = df.withColumn("pressure2", substring(df.ambient_pressure, -2,2))
df = df.withColumn("pressure1", substring(df.ambient_pressure, 0,3))
from pyspark.sql.functions import concat, col, lit
df = df.withColumn("pressure", concat(df.pressure1, df.pressure2))
df.show()


# In[39]:


df = df.drop("pressure1")
df = df.drop("pressure2")
df = df.drop("ambient_pressure")

df = df.withColumn("pressure",df["pressure"].cast(FloatType()))
df = df.withColumn("year",df["year"].cast(FloatType()))
df = df.withColumn("PM25",df["PM25"].cast(FloatType()))
df.printSchema() 


# In[40]:


mean_sales5 = df.select(mean(df['pressure'])).collect()
df = df.na.fill(mean_sales_val1, subset=['pressure'])
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()


# In[41]:


df =df.na.drop()
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()


# In[42]:


df= df.withColumn('PM25', when(col('PM25') > 0, col('PM25')))
df =df.na.drop()


# In[43]:


df.count()


# In[44]:


df = df.withColumn("wind_d",                                  when(col("wind_direction") <= 90, 1).                                 when(col("wind_direction") <= 180, 2).                                 when(col("wind_direction") <= 270, 3).                                 otherwise(4))

df = df.drop("wind_direction")


# In[45]:


df.show()
df.printSchema()


# In[46]:



df = df.withColumn("PM25_level",                                  when(col("PM25") <= 10, 1).                                 when(col("PM25") <= 17, 2).                                 otherwise(3))


# In[47]:


df.show()


# In[48]:


from pyspark.ml import Pipeline
from pyspark.ml.classification import (RandomForestClassifier, GBTClassifier, DecisionTreeClassifier)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

cols_now = ['air_temperature',
 'dewpoint',
 'wind_speed',
 'year',
 'pressure',
 'wind_d']
assembler = VectorAssembler(inputCols=cols_now, outputCol='features')
data = assembler.transform(df)
data = data.withColumn("label", df.PM25).drop('air_temperature',
 'dewpoint',
 'wind_d',
 'wind_speed',
 'year',
 'pressure', 
  'PM25', 'PM25_level'                                    )
data.show()


# In[49]:


import findspark
findspark.init('/home/ubuntu/spark-2.1.1-bin-hadoop2.7')
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('linear_regression_docs').getOrCreate()
from pyspark.ml.regression import LinearRegression
(training, test) = data.randomSplit([0.7, 0.3])
lr = LinearRegression(featuresCol='features', labelCol='label', predictionCol='prediction')
lrModel = lr.fit(training)
print("Coefficients: {}".format(str(lrModel.coefficients))) # For each feature...
print('\n')
print("Intercept:{}".format(str(lrModel.intercept)))


# In[50]:


test_results = lrModel.evaluate(test)
print("RMSE: {}".format(test_results.rootMeanSquaredError))


# In[51]:


train, test = df.randomSplit([0.7, 0.3], seed = 2018)


# In[52]:


import findspark
findspark.init('/home/ubuntu/spark-2.1.1-bin-hadoop2.7')
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('tree_methods_doc').getOrCreate()

from pyspark.ml import Pipeline
from pyspark.ml.classification import (RandomForestClassifier, GBTClassifier, DecisionTreeClassifier)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


cols_now = ['air_temperature',
 'dewpoint',
 'wind_speed',
 'year',
 'pressure',
 'wind_d']
assembler = VectorAssembler(inputCols=cols_now, outputCol='features')
data = assembler.transform(df)
data = data.withColumn("label", df.PM25_level).drop('air_temperature',
 'dewpoint',
 'wind_d',
 'wind_speed',
 'year',
 'pressure', 
  'PM25', 'PM25_level'                                    )
data.show()


# In[53]:


data.show()
(trainingData, testData) = data.randomSplit([0.7, 0.3])


# In[54]:


from pyspark.ml import Pipeline
from pyspark.ml.classification import (RandomForestClassifier, GBTClassifier, DecisionTreeClassifier)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

(trainingData, testData) = data.randomSplit([0.7, 0.3])
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=20)
model_rf = rf.fit(trainingData)


# In[55]:


prediction_rf = model_rf.transform(testData)


# In[56]:


prediction_rf.show()
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(prediction_rf)
accuracy


# In[57]:


from pyspark.ml import Pipeline
from pyspark.ml.classification import (RandomForestClassifier, GBTClassifier, DecisionTreeClassifier)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

(trainingData, testData) = data.randomSplit([0.7, 0.3])
dt = DecisionTreeClassifier()
model_dt = dt.fit(trainingData)
prediction_dt = model_dt.transform(testData)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(prediction_dt)
accuracy


# In[58]:


model_dt.featureImportances


# In[59]:


model_rf.featureImportances


# In[70]:


import matplotlib.pyplot as plt
# Show histogram of the 'C1' column
bins, counts = df.select('air_temperature').rdd.flatMap(lambda x: x).histogram(20)
plt.hist(bins[:-1], bins=bins, weights=counts)


# In[71]:


bins, counts = df.select('PM25').rdd.flatMap(lambda x: x).histogram(20)

plt.hist(bins[:-1], bins=bins, weights=counts)


# In[64]:


bins, counts = df.select('wind_speed').rdd.flatMap(lambda x: x).histogram(20)

# This is a bit awkward but I believe this is the correct way to do it 
plt.hist(bins[:-1], bins=bins, weights=counts)

