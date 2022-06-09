##   《Simplified DOM Trees for Transferable Attribute Extraction from the Web》论文解读
### 整体认知
    [本论文](./3.Simplified DOM Trees for Transferable Attribute Extraction from the Web.pdf)主要是把网页信息抽取当成一个分类任务。即对一个网页来说，对网页中的文本打标签。比如对汽车领域，会对文本标记为model,price,fuel_ecommy等。本轮文可以解决对于单个垂直领域和跨领域信息抽取的问题，两种任务使用的分类器有所区别。本轮文的准确度较高，也有[pytorch代码](https://github.com/MurtuzaBohra/SimpDOM)实现。
    本文把attribue当成要预测的class，也有些论文中称为filed。
### 脉络梳理
    本论文主要分为以下几个部分:
    1. introduction主要介绍了网页抽取问题以及如何去构建特征等。
    2. problem formulation and approach，主要是介绍了将网页信息抽取定义为垂直领域和跨领域两个问题。
    3. node encoder and classifier,主要描述了解决问题的方法。其中，如图figure 3展示的那样，输入的包含两个部分，分别在3.1节和3.3节有描述。text_encoder是在3.2节有描述，主要是将3.1节中的输入转化为向量，再和3.3中得到的向量进行拼接，然后再3.4节中进行两个任务的预测。
    4. experiments,主要是将本方法的结果，数据集使用的是[SWDE](https://drive.google.com/file/d/1aMuHb8RT_GrKr6VoUvmDsObEwIqypkHy/view?usp=sharing).本方法要比FreeDOM-Full要高一些。做成尝试使用bert来代替glove向量，发现效果反而低了9.8,详见table3中的Contextualied Embedding。
    5. related work,主要是横向描述了网页信息抽取任务定义的几种方法,并介绍了相关的论文。比如:
        Attribute extraction（本文）主要是识别网页中有的文本，然后进行划分。
        Relation extraction 主要是识别两个实体之间的关系，这里面又包含了两种：
            closed relation extraction
            open relation extraction
        Composite extraction 抽取更复杂的概念比如：观点，评论或者是情感等。
        Application-driven extraction 包含了一个更广泛的应用场景，比如pdf信息提取，ocr技术，以及网络攻击检测等。


