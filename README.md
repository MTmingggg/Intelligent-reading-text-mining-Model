# Intelligent-reading-text-mining-Model
本模型是基于Python3编写的智能文本挖掘模型。主要步骤包含数据集的格式转换，Jieba分词，去停用词，高权重的关键词提取，利用向量化方法对文本信息进行特征提取等。
# 训练集格式转换
transform.py中的transform()函数拿出content和question两列文本，保存于lab.txt。
# Jieba中文分词
在transform.py中的seg_sentence()函数中实现，jieba.cut()的参数sentence为需要分词的字符串，参数cut_all控制不使用全模式分词。
# 去停用词
transform.py中的seg_sentence()函数加载停用词表的路径。
# 文本信息特征提取
train.py中的important()函数，line为待提取的文本，top8为返回8个TF/IDF权重最大的关键词，withWeight设置为true返回关键词权重值。
# 模型训练
train.py中使用Decision Trees、Gaussian Naive Bayes、Ensemble(Random Forests)、SVC四种方法对训练集进行训练。使用predict()分析经过同样处理的测试集文本信息，输出答案。
# 模型评估与优化
train.py中计算各F1指标及模型准确度。test1.0.py中采用集成学习方法，使用贝叶斯模型训练第 1、2、4组样本，使用决策树分类模型训练第3、5组样本。test2.0.py中输出机器的预测值，保存于submit.txt。
# IDE
PyCharm2018.2+Anaconda3
