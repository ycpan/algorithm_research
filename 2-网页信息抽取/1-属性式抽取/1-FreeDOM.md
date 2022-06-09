##   《FreeDOM: A Transferable Neural Architecture for Structured Information Extraction on Web Documents》论文解读
### 整体认知
    本方法主要使用了两个任务来进行信息抽取。第一个节点预测，通过3.1中的特征(当前网页本身的特征，也叫local feature)，预测当前节点的类别；第二种主要是使用relation network来预测关联节点是不是某个类别(filed/class).使用的是全局特征(就是整个网站的数据特征，也叫global feature).第二个任务主要是对第一个任务中，预测为None类型的node进行一个召回（预测成正常的filed，这里也可以认为是进行了泛化）
### 重点说明
    主要说明论文的第4个部分RELATION INFERENCE MODULE。由于在第三部分预测出来的节点有很多都是None类型的，由于某些网页会存在多个相似文本，比如日期(详见figure2),这些文本大多是没有target filed（目标类别)的，其中有一个可能是目标类别，这样预测出来的可能都是None类型。
    通过观察发现，相关领域（related fileds)的node有一个相似的布局(比如xpath相似),如figure9所示，所以使用relate network进行关联预测。
    在4.1节中阐述了关联节点的构建方法：主要是将节点分类两个group，一个group是certain fileds(该filed至少包含一个预测成该filed的节点),另一个是uncertain fileds(该filed没有出现任何预测的节点).
    假设一共有K个fileds，其中T个certain fileds，K-T个uncertain fileds。则从T个领域中，分别抽取一个代表性的节点代表当前filed。所以可以组成(T*(T-1)个领域的组合。因为是每个领域选择一个节点，所以这也是节点个数的组合。从K-T个filed中抽取得分最高的前m个（uncertain节点),代表K-T个领域中的一个领域。所以K-T个领域本身范围内的*节点*组合为：(K-T)(K-T-1)*m^2.(其中(K-T)*(K-T-1)表示领域之间的自由组合，m^2代表可能的节点之间的组合).
    uncertain组中的node也可能会预测到T领域的filed，因此针对于K-T个领域的(K-T)*m 个节点来说，可能会和T个领域的T*1个节点进行排列组合，可能的组合节点为2*T*(K-T)*m,这里面考虑了前后顺序，所以乘以了2.
    经过上述组合，其实对pair已经包含了filed的信息，下面在预测的时候，只要预测当前组合中的node是不是filed的信息中的value即可。所以预测目标设计成如下集合:
    {none-none, none-value, value-none, value-value}.
    4.3说明了如何获取这些filed. 由于certain filed中的节点类型是明确的，所以只需要重现考虑uncertain filed中的node节点即可.假设uncertain filed中的节点参与了X个pairs，则会得到X个Labels，如果某个label(filed)下面的某个uncertain filed节点预测除了N个value，则认为该Lable就是当前节点label(filed）(成功将第一步节点预测为None的node预测成了某个tagert filed).(4.3中的方法可能会使得某个node预测多个lable，用投票法筛选最高的那个label即可)


