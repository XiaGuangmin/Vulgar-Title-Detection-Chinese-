Vulgar Title Detection (Chinese)
===================


  一个中文低俗标题分类，数据集大概六万+的文章标题（8846低俗+54937非低俗）。

----------


Documents
-------------

#### <i class="icon-folder-open"></i> ./data  
disu.csv   为低俗的标题 原始数据
nomal.csv 为正常非低俗标题 原始数据
disu_seg.csv  为切词后的低俗标题
nomal_seg.csv 为切词后的非低俗标题
word_seg.con 是全部数据的切词结果

> **数据处理**

>  $ `python ./ChineseData_process/word_seg.py` 
> 对原始数据进行切词并保存结果，切词工具是结巴分词
>  `$ python ./ChineseData_process/Chinese_word2vec.py`
> 利用word_seg.con 训练word2vec，生成三个文件( vocabulary, word2vec_gensim, word2vec_org )，保存在./embedding_model中

#### <i class="icon-file"></i> 训练数据
#### <i class="icon-folder-open"></i> ./source

    $python ./preprocess_embeddings.py 
    利用自己的语料训练embedding

    $python ./Chinese_CNN_Title.py
    训练cnn网络分类器

    $python ./Chinese_LSTM_title.py
    训练LSTM分类器

    $python ./detect.py lstm 性感的小姐姐
    $python ./detect.py cnn 性感的小姐姐
    用训练好的网络进行文章标题分类

#### <i class="icon-folder-open"></i> ./source/models

    convnets.py
    LSTMNET.py
    定义的网络结构

ps：觉得有用请给我 **star!!**  如有疑问，欢迎指正。