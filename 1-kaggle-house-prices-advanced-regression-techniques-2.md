## 1.【Kaggle】House Prices: Advanced Regression Techniques（下）

[Predict sales prices and practice feature engineering, RFs, and gradient boosting](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

### 1.6 构建baseline

<video width=80%  controls >
	<source type="video/mp4" src="1-kaggle-house-prices-advanced-regression-techniques/1-6-1.mp4" />
</video>

<video width=80%  controls >
	<source type="video/mp4" src="1-kaggle-house-prices-advanced-regression-techniques/1-6-2.mp4" />
</video>

**任务名称：**构建baseline

**任务简介：**完成一个基本的baseline提交到kaggle上然后有成绩

**详细说明：**本节将会向大家介绍利用python数据清洗和数据预处理以及模型的构建，拟合数据，进行对test数据集进行预测，提交到成绩有排名。会先从理论讲起，再到实际的的一个操作。

数据清洗和数据处理是比赛以及任何一种机器学习模型的必须要经过的过程，而且极为重要，这里只是给大家介绍一下数据清洗的几种常见的知识，包括可以利用pandas和sklearn库来进行，对数据的空值的填充，以及数据归一化，独热编码，标签编码等数据处理方面的问题，以及模型的构建问题，如何进行训练以及这个预测提交的问题。在这个过程中可能有很多同学对于很多知识不是很熟悉，那么就需要自己多多面向谷歌或者组队讨论，出现问题的时候多思考以及多查阅资料。

**代码下载：**

链接：https://pan.baidu.com/s/11hmFMnKqnA1j_5NnpoGzVQ 

提取码：wxr6 

**作业名称（详解）**：针对于不同的数据运用pandas和sklearn处理的方式区别是什么？说明模型只能拟合什么样子的数据，为什么数据归一化和不归一化的结果会有差距？提交成绩的截图。

**作业提交形式：**截图，文字，打卡提交。

**打卡内容：**（可以只是文字提交，或图片提交，或组合都行）文字要求最少200字 图片要求最少1张

### 1.7 特征工程知识部分讲解

<video width=80%  controls >
	<source type="video/mp4" src="1-kaggle-house-prices-advanced-regression-techniques/1-7-1.mp4" />
</video>

<video width=80%  controls >
	<source type="video/mp4" src="1-kaggle-house-prices-advanced-regression-techniques/1-7-2.mp4" />
</video>

**任务名称**：特征工程知识点的讲解以及特征工程对成绩的提高

**任务简介**：运用特征工程知识对成绩提高到top80%

**详细说明**：由于特征工程对于后续成绩的提高有着奇特的效果，所以在两次课中会运用特征组合以及管道知识对数据进行处理和特征的组合，希望大家不要完全按照我的方法来进行特征组合，这个时候大家自己要尝试不同的组合，对特征的重要度也需要进行区分。

**代码下载：**

链接：https://pan.baidu.com/s/11hmFMnKqnA1j_5NnpoGzVQ 

提取码：wxr6

**作业名称（详解）：**截图排名top80%及以上，描述一下pipline对特征组合的方便之处，还有哪些方法可以对成绩有所提高？

**作业提交形式**：PPT截图或手写拍照,打卡提交。

**打卡内容：**（可以只是文字提交，或图片提交，或组合都行）文字要求最少200字 图片要求最少1张

### 1.8 特征工程对baseline的提高

<video width=80%  controls >
	<source type="video/mp4" src="1-kaggle-house-prices-advanced-regression-techniques/1-8.mp4" />
</video>

**任务名称**：特征工程知识点的讲解以及特征工程对成绩的提高

**任务简介**：运用特征工程知识对成绩提高到top80%

**详细说明**：由于特征工程对于后续成绩的提高有着奇特的效果，所以在两次课中会运用特征组合以及管道知识对数据进行处理和特征的组合，希望大家不要完全按照我的方法来进行特征组合，这个时候大家自己要尝试不同的组合，对特征的重要度也需要进行区分。

**代码下载：**

链接：https://pan.baidu.com/s/11hmFMnKqnA1j_5NnpoGzVQ 

提取码：wxr6

**数据处理之特征选择知识课件下载:**

链接：https://pan.baidu.com/s/1CqK9sCZ7soLgv5k8i9T9lQ 

提取码：qo9j 

**作业名称（详解）：**截图排名top80%及以上，描述一下pipline对特征组合的方便之处，还有哪些方法可以对成绩有所提高？

**作业提交形式**：PPT截图或手写拍照,打卡提交。

**打卡内容：**（可以只是文字提交，或图片提交，或组合都行）文字要求最少200字 图片要求最少1张

### 1.9 模型集成原理与实践

<video width=80%  controls >
	<source type="video/mp4" src="1-kaggle-house-prices-advanced-regression-techniques/1-9.mp4" />
</video>

**任务名称**：模型的集成对成绩的提高

**任务简介**：运用模型的集成知识对成绩提高到top20%

**详细说明**：模型是不过是尽可能的逼近数据的上限，单个模型对数据的你拟合的效果不会很好，因此在这里实行三个臭皮匠顶一个诸葛亮的思想，采用模型的集成和模型的堆叠来对数据进行拟合，以达到最好的效果。

**房产预测总代码下载：**

链接：https://pan.baidu.com/s/11hmFMnKqnA1j_5NnpoGzVQ 

提取码：wxr6

**作业名称（详解）：**截图排名top20%及以上，描述一下模型的叠加对结果的影响，还有哪些方法可以对成绩有所提高？，之前的模型叠加的成绩不是很好，然后自己运用所学到的知识把排名提高到top10%，参考老师提供的代码方案，截图提交。

**作业提交形式**：PPT截图或手写拍照,打卡提交.

**打卡内容：**（可以只是文字提交，或图片提交，或组合都行）文字要求最少200字 图片要求最少2张

### 1.10 直播答疑

<video width=80%  controls >
	<source type="video/mp4" src="1-kaggle-house-prices-advanced-regression-techniques/1-10-1.mp4" />
</video>

<video width=80%  controls >
	<source type="video/mp4" src="1-kaggle-house-prices-advanced-regression-techniques/1-10-2.mp4" />
</video>

**直播内容：**直播答疑（第二周【Kaggle：房价预测】学习内容相关的问题）

**注：**直播结束后重新进入直播间即可进行回看

**直播代码下载：**

链接：https://pan.baidu.com/s/1fXkUY4fiKTVe6lZ_SE5USA 

提取码：49ya 
