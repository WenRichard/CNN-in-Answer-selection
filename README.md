# WikiQA on QACNN
## 复现论文《APPLYING DEEP LEARNING TO ANSWER SELECTION: A STUDY AND AN OPEN TASK》  
本项目采取了论文中最好的模型进行实验，数据集采用WikiQA，后期会上传insuranceQA的实验结果  
模型图如下：  
![model]( https://github.com/WenRichard/QACNN/raw/master/photo/model.png)  
**实验结果**：    

|Model|CNN share|Dropout|Parameters|Margin|Epoch|MAP|MRR|  
|-|-|-|-|-|-|-|-|  
|QACNN|No|0.5|2115200|0.5|100|0.655|0.673|  
|QACNN|Yes|0.5|481664|0.5|100|0.684|0.697|  

**Loss**：
![Pairwise Loss]( https://github.com/WenRichard/QACNN/raw/master/photo/loss0.5.png)  

**有时间就会更新QA实验，有兴趣的同学可以follow一下，也欢迎Fork和Star！**  
**留言请在Issues或者email xiezhengwen2013@163.com**
