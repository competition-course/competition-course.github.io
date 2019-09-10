## 1.【Kaggle】House Prices: Advanced Regression Techniques（上）

[Predict sales prices and practice feature engineering, RFs, and gradient boosting](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

### 1.1 kaggle账号注册与竞赛入门

<video width=80%  controls >
	<source type="video/mp4" src="1-kaggle-house-prices-advanced-regression-techniques/1-1.mp4" />
</video>

**任务名称：**赛前介绍和准备

**任务简介：**视频的第一个课时，赛前介绍

**详细说明：**本节将会向大家介绍一下老师竞赛的经历、为什么深度之眼要做一个这样子的比赛、kaggle竞赛的类别、kaggle竞赛的类型、如何入门kaggle、竞赛的技巧、kaggle账号的注册。

在这次课里面老师向大家介绍了kaggle竞赛的类型包括数据挖掘类的、图像类的、自然语言处理类的，kaggle竞赛得级别，有入门级、中级、以及高级比赛。

**如何入门kaggle？**

1. 首先要有kaggle账号

2. 机器学习基础、数据分析基础

3. 了解kaggle的各个模块

4. 学会薅资本主义的羊毛

**Kaggle竞赛的技巧：**

1. 多看、多学、多实践

2. 多看kaggle模块的里面的kernel以及discussion

3. 切忌单打独斗、一定要抱团取暖

**Kaggle模块的注册：**

1. 要科学上网

2. Kaggle模块的介绍

**作业名称：**截图显示自己已经注册好kaggle账号，并且可以命一个中文名字。

**作业提交形式：**kaggle网页截图,打卡提交.

**打卡内容：**（提交1张图片）

### 1.2 账号注册以及本地化jupyter notebook

<video width=80%  controls >
	<source type="video/mp4" src="1-kaggle-house-prices-advanced-regression-techniques/1-2.mp4" />
</video>

**任务名称：**本地化jupyter notebook以及对这个赛题的解读、kaggle账号的注册

**任务简介：**注册好kaggle账号、本地化jupyter notebook

**详细说明：**注册kaggle账号需要科学上网，验证手机号的时候可能会出错，存在验证不了的，这个时候就要换一个手机号（最好还是用Google账号注册、邮箱和手机号验证），对于本地化notebook要说明一下，因为咱们后期是需要用到notebook的环境的所以的话就要安装Anaconda软件，下面提供了下载网址：https://www.anaconda.com/distribution/

安装这个软件的时候最好是安装在D:/Anaconda目录下，对于他的安装视频里面有详细的教程，你只要在选择安装目录的时候改一下就OK了，然后就是修改jupyter notebook的路径，参照PPT上的链接，大家安装好了之后就可以进入notebook环境了，记住点击notebook之后那个黑框框，不要关掉，因为那是解释器。

今天就是给大家讲了一下如何注册kaggle账号以及本地化notebook的知识点。

**作业名称（详解）**：掌握注册kaggle以及本地化jupyter notebook

**作业提交形式：**kaggle注册好的截图，以及本地化notebook打开浏览器界面的截图,打卡提交。

**打卡内容：**最少提交2张图片

### 1.3 赛题思路分析

<video width=80%  controls >
	<source type="video/mp4" src="1-kaggle-house-prices-advanced-regression-techniques/1-3.mp4" />
</video>

**任务名称：**赛题的思路

**任务简介：**了解赛题要解决的问题，数据的说明以及介绍，要运用的算法

**详细说明：**进入比赛界面看到的第一眼的就是赛题的overview，要解决的问题，评估方式采用的是RMSE，接下来要看的就是Data，看数据Data里面的File Descriptions，看一下文件有哪些。再看的就是Data Fields里面的数据有哪些特征，然后就是数据的下载，点Download All 下载全部数据下来。

看到这个数据的第一眼就是SalePrice是房价也是标签，那肯定是用来做回归的了，这个时候就要想一下有哪些算法是可以用来做回归的了，还有看到数据的内容的时候，哪些是这个数值型，非数值的，大概就要想到有哪些的东西要做处理了以及有哪些方法来做处理了

**代码下载：**

链接：https://pan.baidu.com/s/15CVlreLNaTdtJKZvryFH2Q 

提取码：l3tv 

**作业名称（详解）：**掌握kaggle里每一个比赛里面数据的查看以及下载，并且对下载下来的数据进行发表自己的看法

**作业提交形式：**数据下载到本地的截图，针对这个赛题的数据发表自己的评论,

**打卡内容：**图片至少1张、评论至少100字

### 1.4 数据清洗以及数据处理

<video width=80%  controls >
	<source type="video/mp4" src="1-kaggle-house-prices-advanced-regression-techniques/1-4.mp4" />
</video>

**任务名称：**房价的数据预处理之探索性数据分析

**任务简介：**数据的读取与显示、查看数据的缺失情况、查看数据的类型情况、查看特征之间的相关性。

**详细说明：**数据的读取与显示利用pandas来读取语法是train=pandas.read_csv(filepath)。数据的显示直接在cell中打印train就好了，如要显示前五行的话那么就要train.head(),或者是利用train.sample(5)随机显示五行数据，在这里呢，用一个叫做pandas_profiling的包来进行查看数据的缺失情况、数据的类型情况、特征之间的相关性，具体的用法的是

```python
import pandas_profiling as ppf

ppf.profileReport(train)
```

就行啦！后面的话会生成的一个报表包括上面的所有的信息。这里要提醒一下的是要用这个包的话，那么就要在Anaconda Prompt中运行pip install pandas_profiling，稍微等待一下就好了。

从这个报表里面可以看到的是数据缺失情况，以及每一个特征的分布情况，特征与特征之间的相关性等等信息。

**代码下载：**

链接：https://pan.baidu.com/s/1pAqxY3ZudZZxk_UK62J7Yw 

提取码：j7v3 

**作业名称（详解）：**读取train.csv文件并且显示后五行，自行查阅pandas中的两种数据结构分别是什么？并做出总结（文字）。Pandas的可视化函数有哪些？（文字），person系数为什么可以用来衡量数据之间的相关性？（图文并茂的提交），如何利用pandas来显示数据信息（代码运行截图）

**作业提交形式：**打卡提交，文字最少70字，图片最少2张

### 1.5 直播答疑

<video width=80%  controls >
	<source type="video/mp4" src="1-kaggle-house-prices-advanced-regression-techniques/1-5-1.mp4" />
</video>

<video width=80%  controls >
	<source type="video/mp4" src="1-kaggle-house-prices-advanced-regression-techniques/1-5-2.mp4" />
</video>

**直播内容：**直播答疑（第一周【Kaggle：房价预测】学习内容相关的问题）

**注：**直播结束后重新进入直播间即可进行回看

**直播代码下载：**

链接：https://pan.baidu.com/s/1qo8M3TZJfyJp3fwv02Ma_Q 

提取码：t085 
