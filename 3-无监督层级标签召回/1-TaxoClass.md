## 《TaxoClass: Hierarchical Multi-Label Text Classification Using Only Class Names》论文解读
### 总体思路
本文主要是针对层级多标签树文本分类问题(Hierarchical multi-label text classification,HMTC)。其主要思路主要包含四个部分:
1. 计算document(长文本)和class(短文本)之间的相似度，这个靓点在于不是直接比较相似度，而是模拟人思考的过程，构建句对。比如：
把class(tag)转换为this document is <class>.  最后构成的句对为(<document>,this document is <class>)
这一步可以得到文本(document)和class之间的相似度。这一部分主要在3.1部分中。
2. 3.2部分主要是对核心类的挖掘。包含两个子部分：
 3.2.1 核心类候选集的筛选，首先通过3.1的分数，对标签树进行剪枝（减少计算量),比如对于l层，我们筛选出其子类(l+1)层的个数为l+1个。即在l+1层，其子类个数为l+2.也就是在l层，其子类节点个数为l-1 + 2个。从figure3左半部分图可以看出，第一层节点点个数为2，第二层节点个数为3.然后再裁剪的基础上，每一层攻击挑选(l+1)^2个类。
 3.2.2 我们通过计算置信度，得到候选类
 3.3 主要是对文本(document)和类(class)分别进行编码，然后通过matching network进行预测得分和类别，训练。
 3.4 主要是进行多标签的预测，这一部分还没看懂。

