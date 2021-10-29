# LSTM模型实例

## 一、文件介绍

### 1.__pycache__文件夹

​        这里面包含由python自动生成的字节码缓存文件，即编译的python或 .pyc 文件。

### 2.models文件夹

​        这里面存放的是训练后生成的.ckpt模型文件。

### 3.penn_small文件夹

​        这里面存放的是测试、训练、验证用的数据，分别对应test.txt、train.txt、valid.txt文件。

### 4.give_valid_test.py

​        这个文件里写的是一些数据处理时用到的函数。

### 5.LSTMLM.py

​        这份代码能完整实现LSTM模型，包括数据处理、模型训练、模型验证等等。其中LSTM模型的构建使用了pytorch框架中的torch.nn.LSTM()函数，容易搭建。

### 6.LSTMLM_onelayer.py

​        这份代码能完整实现LSTM模型，包括数据处理、模型训练、模型验证等等。其中使用torch.nn.Linear()、torch.nn.Parameter()等基础函数搭建了一个单层LSTM模型，方便读者理解LTSM模型的内部结构。

### 7.LSTMLM_twolayer.py

​        这份代码能完整实现LSTM模型，包括数据处理、模型训练、模型验证等等。它在单层LSTM模型基础之上拓展实现了双层的LSTM模型，读者可以根据这个文件尝试实现多层LTSM模型。

## 二、参考资源

### 1.Pytorch官方文档中对torch.nn.LSTM()函数的介绍：

[LSTM — PyTorch 1.10.0 documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM)

### 2.东北大学自然语言处理实验室提供的学习资源：

[VuePress (nlplab.com)](https://www.nlplab.com/courses/nlp2021/)

## 三、联系博主

邮箱：1263785136@qq.com

最后，感谢[东北大学自然语言处理实验室 (nlplab.com)](https://www.nlplab.com/)的老师与同学们提供的帮助与指导！