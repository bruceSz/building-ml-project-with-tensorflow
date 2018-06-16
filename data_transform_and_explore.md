#### 一. 数据探索和转换

* tf是一个开源软件库，它使用数据流图来进行数值计算。图中的节点标书数学操作，边表示节点间传递的
多维矩阵(即tensors)。
利用tf提供的各种函数，你可以实现前沿的卷积神经网络cnn来处理图片，或者实现递归/循环神经网络rnn来处理文字（当然cnn/rnn的威力不止如此，这里不用较真）。使用tf作为开发框架，你可以
很容易的构建和使用自己的机器学习模型，而将计算的复杂性交由tf提供的数据流图处理。
tf支持在各种异构环境上运行，从CPU到移动设备到高度并行的GPU，以及其他可以叫上名字的其他混合架构，如下图：
pic1.

### tf的主要数据结构-Tensors
* tf使用tensor来管理它的数据。这里的Tensor是来自数学中的概念，是线性代数中向量和矩阵的泛化。
在tf中，tensor指有类型的，多维矩阵，支持多种操作，建模为tensor对象。

## tensor的属性： 秩，形状，类型
* 正如之前提到的，tf使用tensor来表达所有数据，一个tensor具有静态类型和动态维度，你可以实时修改tensor的内部组织结构。另外，只有tensor类型对象才能在数据流图的节点之间传递。下面就逐个看看tensor的三个主要属性（以后再提到tensor，即指tensor对象）。

# tensor 秩
tensor的秩描述了tensor的维度信息。和矩阵的秩不同的是，tensor的秩指tensor维度的数量,并不能准确衡量tensor的行/列或者其他空间的对应描述。
秩为1的tensor其实就是向量，秩为2的tensor是矩阵。2秩tensor可以使用类似t[i,j]的形式来访问某个元素，3秩tensor可以使用t[i,j,k]的形式来访问某个元素，以此类推。
下面代码中，创建了一个tensor，放并访问它的组件：
```
import tensorflow as tf
sess = tf.Session()
t1 = tf.constant([[[1,2],[2,3],[3,4],[5,6]]])
print(sess.run(t1))[0,3,0]
```
上面的tensor的秩为3，上述代码输出： 5

下表是一个一些tensor的例子：
pic2.

# tensor 形状
tf文档使用三个计数惯例来描述tensor的维度：秩，形状，维度数,举例如下表：
pic3

# tensor数据类型
除了维度tensor还有一个固定的数据类型，tensor可以拥有如下类型中的一个：
pic4.

## 创建tensor
可以制定参数自己创建tensor，也可以通过著名的numpy库来间接创建。下面代码中，首先创建一些
numpy的数组，只有利用这些数组创建tensor。
```
import tensorflow as tf
import numpy as np
x = tf.constant(np.random.rand(32).astype(np.float32))
y = tf.constant([1,2,3])

```

# numpy和tensor相互转换
tf和numpy是可以互换使用的，只需要执行eval函数调用就可以返回numpy对象，与其他数值库交互了。
note2.
下面代码创建了两个numpy 数组，并将它们转化为tensor。
```
import tensorflow as tf
import numpy as np
x = tf.constant(np.random.rand(32).astype(np.float32))
sess = tf.Session()
x_data = np.array([[1,2,3],[3,2,6]])
x = tf.convert_to_tensor(x_data)
print(x)

```
convert_to_tensor将python的各种类型对象转换为tensor对象。它可以以：tensor对象，numpy数组，pyothon列表，python标量作为输入参数。
上述输出： Tensor("Const_2:0", shape=(2, 3), dtype=int64)

tf支持在python交互命令行执行，也可以使用ipython的命令行，这样的输出和notebook风格兼容，类似jupyter.
在通过交互模式运行tf的Session时，最好使用InteractiveSession对象。与Session不同，InteractiveSession会将自己作为默认session，所以执行tensor的eval时候，或者run一个op的时候，就不用制定Session对象了。

##tf的数据流图
tf的数据流图是模型如何进行计算的符号表示，如下图：
pic5.

 数据流图表示了tf中的计算流程，该图中，节点表示操作，边表示节点之间的流动的数据。
 正常情况下，节点实现了某种数学操作，同时也表示了一种输入/变量和输出数据之间的连接。
 边描述了节点之间的输入输出关系，并且只能是tensor类型。
 节点可以被指派给某个设备，当所有输出数据边都被满足之后，节点将被异步并行执行。所有的操作
 都有名字，描述了某个抽象计算过程。

#构建计算图

用户创建模型需要的tensors和operation之后，计算图Graph对象就自动被创建出来，python tensor构造器会为图
上添加需要的东西，operation也一样。例如 语句： c= tf.matmul(a,b)会创建一个matmul操作。这个操作有两个输入，
a和b（后文操作一律使用op表示）
# operation对象方法

* tf.operation.type: 返回操作的类型
* tf.Operation.inputs: 返回操作输入列表
* tf.Graph.get_operations(): 返回图上的op列表
* tf.Graph.version: 返回图版本。

# 数据输送feeding
tf支持将tensor直接输送到某个操作的机制。 输送过程就是将tensor的的输入替换为某个tensor的具体值。
使用方法为在执行run方法的时候将该值作为参数传入。实际上feeding也只在执行run调用的时候才会用到。
通常的使用方式是首先指定一个特殊的op:tf.placeholder()，作为某个op的输入，在要执行run方法的时候
将真实值作为参数传进去。

# 变量 variables
在很多计算中，一个图可能会被执行多次，但是大部分的tensor的生存周期只是某次执行。变量是一种特殊的op
它是一个指向可变,持久化的tensor的句柄，并且可以在图多次执行中存活（生存周期不再是某次执行）。
对于机器学习应用，模型参数一般会保存在varibles里面，在训练图的过程中不断更新。下面的代码使用了一个1000
个0 的数组初始化一个变量：
```
b = tf.Variable(tf.zeros([1000]))
```

# 保存数据流图
数据流图使用google的protobuf存储，因而可以在保存之后被跨语言读取。

# protobufer - 图序列化语言
protobuffer是跨语言，跨平台，用来序列结构化数据的可扩展机制。用户首先定义数据结构，之后就可以使用proto buffer
生成的代码来访问该数据结构，注意生成的代码是支持多语言的。

常用方法
tf.Graph.as_graph_def(from_version,add_shapes) 返回一个GraphDef对象，该对象是数据流图的序列化形式
 from_version: 如果不为空，则GraphDef只会记录from_version指定版本之后加入的节点
 add_shapes: 如果为true，为每个节点添加shape属性。

# 构建图例子：
```
import tensorflow as tf
g = tf.Graph()
    with g.as_default():
        sess = tf.Session()
        w = tf.Variable(tf.zeros([10,5]))
        v = tf.placeholder(tf.float32,[None,10])
        result = tf.matmul(v,w)
    print(g.as_graph_def())
```

## 运行程序- Session
客户端通过Session对象和tf交互。Session代表了计算执行的整个环境。初始情况，Session对象是空的，
随着各种op和tensor的不断生成，它们会被自动加到session内，直到run方法被调用后开始运行。
run方法接收一系列需要计算的输出名作为参数，以及需要feed的tensor，将对应的输入替换掉。
调用run之后，session会找到需要计算的输出的所有依赖（递归的），并且挨个计算他们。
下面语句创建一个新的Session对象：
```
s = tf.Session()
```

# 基本tensor方法
下面探讨下tf中的一些基本方法，通过这些方法可以对数据进行数据探索，或者为了更好地进行并行计算
对数据进行预处理。

简单矩阵操作
tf支持大量常用的矩阵操作，例如转置，矩阵相乘，行列式，矩阵求逆等,如下：
```
import tensorflow as tf
    sess  = tf.InteractiveSession()
    x = tf.constant([[1,2],[3,4]])
    print(tf.transpose(x).eval())
    print()

```
规约
规约操作将操作应用到tensor的一个维度上，输出结果比输入的tensor少一个维度。
规约操作包括：product,minimum,maximum,mean,all,any,accumulate_n等。
```
import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.constant([[1,2,3],[3,2,1],[-1,-2,-3]])
b_tensor = tf.constant([[True,False,True],[False,False,True],[True,False,False]])
tf.reduce_prod(x)
Out[27]: <tf.Tensor 'Prod:0' shape=() dtype=int32>
print(tf.reduce_prod(x))
Tensor("Prod_1:0", shape=(), dtype=int32)
print(tf.reduce_prod(x).eval())
-216
print(tf.reduce_prod(x,reduction_indices=1).eval())

```
输出： [ 6  6 -6]


#tensor分段
分段操作同样在tensor的某个维度上操作，不同的是，输出tensor取决于某个索引行（index row）。如果该行某些值重复了，那么这些值对应tensor的维度共享这个值作为索引。分段操作应用到这些共享索引的维度`值`上. （译者：分段类似于group By)
如下图 ： pic6.
注意，索引数组大小和tensor的第0维对应数组大小相同。并且是间距为1 逐个递增的。

```
import tensorflow as tf
sess = tf.InteractiveSession()
seg_ids = tf.constant([0,1,1,2,2])
 tens1 = tf.constant([[2,5,3,-5],[0,3,-2,5],[4,3,5,3],[6,1,4,0],[6,1,4,0]])
tf.segment_sum(tens1,seg_ids).eval()


```
Out[51]: 
array([[ 2,  5,  3, -5],
       [ 4,  6,  3,  8],
       [12,  2,  8,  0]], dtype=int32)

#序列
序列操作包含类似argmin,argmax这样返回一个维度上的最小，最大值的方法，以及listdiff这样
返回list之间差集的方法。以及where这种返回tensor上真值的index。还有类似unique
：返回一个list上的唯一值得集合。
```
import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.constant([[2,5,3,-5],[0,3,-2,5],[4,3,5,3],[6,1,4,0]])
listx = tf.constant([1,2,3,4,5,5,6,7,8])
listy = tf.constant([4,5,6,7])
boolx = tf.constant([[True,False],[False,True]])
tf.argmin(x,1).eval()
Out[59]: array([3, 2, 1, 3])
tf.argmin(x,0).eval()
Out[60]: array([1, 3, 1, 0])
tf.where(boolx).eval()
Out[61]: 
array([[0, 0],
       [1, 1]])
tf.unique(listx)[0].eval()
Out[62]: array([1, 2, 3, 4, 5, 6, 7, 8], dtype=int32)
tf.unique(listx)[1].eval()
Out[63]: array([0, 1, 2, 3, 4, 4, 5, 6, 7], dtype=int32)

```

# tensor 形状变换
tensor的形状变换和矩阵形状相关，用来转换不匹配的数据结构，并可以迅速获取数据计量信息（measures of data）,在运行时决定处理策略的时候特别有用。下面的例子中，首先从一个秩为2
的tensor开始，打印一些信息后，修改矩阵的维度：通过squeeze或者expand_dims减少或者添加新维度。
```
import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.constant([[2,5,3,-5],[0,3,-2,5],[4,3,5,3],[6,1,4,0]])
tf.shape(x).eval()
Out[84]: array([4, 4], dtype=int32)
x.shape
Out[85]: TensorShape([Dimension(4), Dimension(4)])
tf.rank(x)
Out[86]: <tf.Tensor 'Rank:0' shape=() dtype=int32>
tf.rank(x).eval()
Out[87]: 2
tf.reshape(x,[8,2]).eval()
Out[88]: 
array([[ 2,  5],
       [ 3, -5],
       [ 0,  3],
       [-2,  5],
       [ 4,  3],
       [ 5,  3],
       [ 6,  1],
       [ 4,  0]], dtype=int32)
tf.squeeze(x).eval()
Out[89]: 
array([[ 2,  5,  3, -5],
       [ 0,  3, -2,  5],
       [ 4,  3,  5,  3],
       [ 6,  1,  4,  0]], dtype=int32)
x.eval()
Out[90]: 
array([[ 2,  5,  3, -5],
       [ 0,  3, -2,  5],
       [ 4,  3,  5,  3],
       [ 6,  1,  4,  0]], dtype=int32)
tf.expand_dims(x,1).eval()
Out[91]: 
array([[[ 2,  5,  3, -5]],
       [[ 0,  3, -2,  5]],
       [[ 4,  3,  5,  3]],
       [[ 6,  1,  4,  0]]], dtype=int32)
tf.expand_dims(x,0).eval()
Out[92]: 
array([[[ 2,  5,  3, -5],
        [ 0,  3, -2,  5],
        [ 4,  3,  5,  3],
        [ 6,  1,  4,  0]]], dtype=int32)
tf.expand_dims(x,2).eval()
Out[93]: 
array([[[ 2],
        [ 5],
        [ 3],
        [-5]],
       [[ 0],
        [ 3],
        [-2],
        [ 5]],
       [[ 4],
        [ 3],
        [ 5],
        [ 3]],
       [[ 6],
        [ 1],
        [ 4],
        [ 0]]], dtype=int32)


```
#tensor 切片和关联
为了从大数据集上抽取和合并信息，slicing和joining操作特别有用。你可只保留那些需要的列，省下不需要的列占的内存，












note1：矩阵的列秩：矩阵线性独立的列的最大数。
note2: tensor对象只是某个操作结果的符号表示，它的结构内并不包含结果值，因此，需要执行eval函数才能获取真正的值，这点和执行Session.run(tensor_to_eval)效果一样。


# 数据流和结果可视化工具- tensorBoard
对结果进行可视化显示是数据科学家需要掌握的一门重要技术。
tensorBoard是一个可以图形化显示数据流图的软件工具包，通过log工具，还可以对结果进行很好的解释。
如下图.
pic7
tensorFlow中的所有操作和和tensor都可以被写入到log中，tensorBoard分析这些信息，可以在session运行时向用户图形化显示计算图中的各种元素。
# 命令行使用
``` 
tensorboard -h
```

# tensorBoard 工作方式
tensorflow有一个实时日志机制，对每个创建的数据图，模型运行过程中的所有信息都可以被记录下来。
模型构建者必须自己选择那些信息需要记录，以作为日后分析用。
tensorflow api使用数据输出对象Summaries来存储所有指定的信息。在session运行过程同时，Summaries对象会被写入到制定的事件文件中。下面的例子里，我们会在一个已经生成了日志文件的目录下启动tensorBoard。
``` 
tensorboard --logdir=. --port=8080
```

# 添加summary nodes
tensorflow session中的所有summaries使用SummaryWriter 对象进行记录。调用方法为：
```
tf.train.SummaryWriter.__init__(logdir,graph_def=None)
```
以上代码会创建一个SummaryWriter对象，和一个event file（在参数提供的目录下）
调用下面方法以后：add_summary(),add_sesison_log(),add_event()，该event file会包含一个event 类型的proto buffer。
如果把graph_def 的proto buffer传入，同样event 文件就会包含整个graph_def。（这和add_graph一样）。
之后运行tensorboard的时候，tb会从event文件中读取相应的图定义，并且图形化展示，方便交互。
整个流程如下：
1 首先创建tensorflow 计算图，确定需要使用summary 进行注解的节点。
要注意的是，因为在运行run之前，tf中的操作并不会被执行，（操作间接依赖的op和依赖op输出的op也一样）.因此创建summary 节点之后，它们只是计算图的附属品，也没有op依赖它们。需要执行summary op才会触发生成summaries。可以使用tf.merge_all_summaries方法，该方法会将左右的summary_op合并成一个op，这样一次执行该方法返回的op即可生成所有的summaries。
运行merged summary op之后，会生成当前状态下序列化的summary protobuf对象，之后将该对象传递给summary_writer即可将summary写入到磁盘上。
summary_writer的logdir参数非常重要，所有的event 文件都会写入到对应目录。如果构造函数中graph_def 也提供了，那么tb在可视化的时候会将整个计算图可视化。

现在已经修改过计算图，并且有了一个summary_writer，可以开始运行网络了。如果需要可以在计算图运行的每步执行一次summary。如果生成的记录太多，可以考虑没n步执行一次summary node。

# 常见的summary操作
* tf.scalar_summary
* tf.image_summary
* tf.histogram_summary
	
# 特殊的summary 函数
* tf.merge_summary (合并一个结合的summary node)
* tf.merge_all_summaries （合并整个图的summary node）
为易读性，可视化部分使用不用的icon来表示常量和summary 节点，如下表：
table 1.
# 和tensorboard ui交互
略
## 从磁盘读取信息
tesorflow支持大部分的标准格式，例如知名的csv，图片文件格式：jpg，png等已经tensorflow自有格式。
# 表格文件 csv
tensorflow 有自己的读取csv的方法，和类似pandas的库相比，该方法稍微复杂一点。
读取过程：
* 1首先创建文件名队列对象，可以包含多个文件。
* 2 创建一个TextLineReader
* 3 decode csv内容，并存储为tensor。
* 4 如果要混合同类数据，使用pack函数。
# iris 数据集
iris花数据集/fisher的iris数据集是分类问题的标准数据集。是由1936年ronald fisher 在阐述线性判别分析时候给出的多元数据集。
该数据集包含50个样本，每个样本四个特征，花萼的长和宽，花瓣的长和宽（厘米为单位）。通过组合这四中特征，fisher设计了线性判别模型来对花进行分类 。
```

```
#读取图片数据
tensorflow支持导入图片格式，这对面向图片的模型尤为重要。支持的图片格式为jpg,和png,内部表示为uint8 的多个tensor。每个图片通道，使用一个rank为2的tensor表示。
# 读取和处理图片。下面例子会导入一个例子图片，做一些处理之后，保存为另外一个单独的图片。
```
```
# 使用标准tensorflow格式读取数据
另外一种读取数据的方法是，首相将所有拥有的数据转化为tf官方格式，然后使用官方格式进行读取。这样的好处是更容易混合各种异构数据集和网络结构不同的数据。
tf提供了tf.python_io.TFRecordWriter类，通过它，可以首先将你已经获得的数据集塞进一个ptotobuf里，之后序列化这个pb，然后存入TFRecord文件里。
可以使用tf.TFRecordReader 和tf.parse_single_example 解码器来读取TFRecord文件，tf.parse_single_example会将读到的pb解码为tensor。
## 总结
在这节，我们学习了一些基本的数据结构和数据结构上的简单操作，以及对计算图的简洁summary。
这些操作是以后介绍的技术的基础，通过它们，数据科学家可以首先对数据集进行整体特性的官产进而决定，使用简单的模型来建模数据-如果数据分类或者调整函数足够清晰-，或者相反使用其他更加高级的技术。
下面的章节我们将开始创建并运行计算图。并使用这章提到的技术和方法解决实际问题 。




