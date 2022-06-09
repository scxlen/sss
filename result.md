# 黑板手写数字识别与打分系统软件设计

*注：以上为标题，标题中xxx为具体的项目名称。本文档斜体字部分为注解。*

#### 完成人：张秋月 沈晨栖 尹董艳

## 1. 概述

### 1.1. 系统简介
该系统基于Python的深度学习方法，构建和训练神经网络后，将通过摄像头实时监测，对捕捉到的图片进行处理输入网络，最后输出预测的识别结果和分数。

### 1.2. 目标读者

项目评定人、尝试该方面开发的初学者、使用该系统进行教学的家长或老师。

### 1.3. 书写约定

*注：特殊的字体，比如粗体字、斜体字表示什么含义；缩写的含义等。*

## 2. 总体设计

*注：用软件层次图或者结构图描述软件的总体架构。陈述对系统进行如此分解的依据。*

本系统为黑板手写数字识别与打分系统，所以必然需要一个实时抓取画面的模块。
抓取到画面后就需要对其进行一些必要的处理，从而去掉多余的突出重要区域，此为预处理模块。
为了最后可以得到结果，我们选择构造一个卷积神经网络，经训练、测试后可实现将预处理后的图片输入进去可以自动得到结果，此为网络模块。。
从网络模块中得到的结果需要输出让用户看见，此需要一个输出显示的模块，即识别数字，同时也需要对最后的结果打分，于是独立出一个打分模块。

*注：然后对总体架构中的每一个模块分小节描述。对每一个模块，用文字描述它所承担的功能；所使用的服务；所提供的服务。如下所示：*

### 2.1.实时监测视频模块
通过cv2.videocapture函数调用电脑摄像头，实时抓取画面。

### 2.2. 图像预处理模块
从摄像头中读取画面图像，进行灰度化、二值化、膨胀等处理，得到网络可接收大小的ROI区域。

### 2.3. 网络模块
构造一个Resnt网络，采用FishionMint数据集中的手写图片，对其进行训练、测试，从而得到一个可信的测试结果权重文件。

### 2.4. 得出结果模块
将ROI区域图片输入网络，得到最终数字结果。

### 2.5. 打分模块
对比ROI区域中的数字与最终结果的数字，可能性越高，分数越高。

## 3. 详细设计

*注：对每一个接口和模块进行详细描述。对接口，列出所有的函数，包括函数名、参数和返回类型，并用简短的文字描述它的功能。对模块，详细描述它所使用的数据结构和算法，可用教材中介绍的过程设计工具，如：程序流程图、PAD图、伪码等表示。*

### 3.1.predict接口详细设计
resnet_block(in_channels, out_channels, num_residuals, first_block=False)

参数 in_channels 决定了该残差网络模块输入图片来源数，输入的in_channel为2，意味着将从两张图片来源进行卷积；

参数 out_channels 决定了该残差网络模块输出图片的通道数，若out_channel为2，即有两个卷积核,对应的输出图片为两个通道；

参数 num_residuals 决定了该残差网络模块由几个残差块组成；

参数 first_block 决定了该残差网络模块的第一层残差块是否要使用1*1的卷积层，如下图所示：

![BF1(6B3ZF(GF8G0CM42JL~6](https://user-images.githubusercontent.com/106146337/172203625-fe68dac4-98a5-4ac4-b472-63cb8db254cd.jpg)

class Residual(nn.Module):

def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):

参数 in_channels 决定了该残差网络模块输入图片来源数，输入的in_channel为2，意味着将从两张图片来源进行卷积；

参数 out_channels 决定了该残差网络模块输出图片的通道数，若out_channel为2，即有两个卷积核,对应的输出图片为两个通道；

参数 stride 决定了该残差网络模块卷积层处理后的尺寸

根据该残差代码的设计，在实例化该 class 时，依据传入参数（use_1x1conv）布尔值的不同，会实例化出两种模式：

(1) 当 use_1x1conv = False 时：输入数据被直接叠加到第二个标准化层的输出.
该模式下，经过第 L 层神经网络和残差叠加处理后，数据的尺寸不变，通道数也不变。

![image](https://user-images.githubusercontent.com/106146337/172204974-68464cb2-ec76-4efb-9345-aacbbd09ae99.png)

(2) 当 use_1x1conv = True 时：输入数据先被第三个卷积层（即1*1的卷积层）卷积，再叠加到第二个标准化层的输出

def forward(self, X):
### 3.2. treatment接口详细设计
def get_number(img):
img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY) 颜色空间转换函数
img:要进行处理的图片
cv.COLOR_RGB2GRAY：要进行的色彩转换方式

img_gray_resize = cv.resize(img_gray, (600, 600))  图像缩放函数，这里把图像变成600*600的

def get_roi(img_bw): 
img_bw :预处理好的二值图
img_bw_sg_bord = cv.copyMakeBorder(img_bw_sg, add_c, add_c, add_r, add_r, cv.BORDER_CONSTANT, value=[0, 0, 0])
cv2.copyMakeBorder()用来给图片添加边框
img_bw_sg：要处理的原图
add_c, add_c, add_r, add_r：上下左右要扩展的像素数
cv.BORDER_CONSTANT：固定值填充
### 3.3.图像预处理模块详细设计

img_resize = cv.resize(img,(600,600))  resize成600*
img_gray = cv.cvtColor(img_resize, cv.COLOR_RGB2GRAY) 灰度化
ret, img_bw = cv.threshold(img_gray, 200, 255,cv.THRESH_BINARY) 对灰度图像进行阈值操作得到二值图像
img_open = cv.dilate(img_bw,kernel,iterations=3)膨胀，针对某一像素点，以其为中心建立蒙版，蒙版中的最大值赋值给该像素点
num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(img_open, connectivity=8, ltype=None)剔除小连通域，img_open : 是要处理的图片，官方文档要求是8位单通道的图像。
connectivity : 可以选择是4连通还是8连通，连通指的是上下左右，8连通指的是上下左右+左上、右上、右下、左下。这里选择8连通。
num_labels : 返回值是连通区域的数量。
labels : labels是一个与image一样大小的矩形（labels.shape = image.shape），其中每一个连通区域会有一个唯一标识，标识从0开始。
stats ：stats会包含5个参数分别为x,y,h,w,s。分别对应每一个连通区域的外接矩形的起始坐标x,y；外接矩形的wide,height；s其实不是外接矩形的面积，实践证明是labels对应的连通区域的像素个数。
centroids : 返回的是连通区域的质心。

cv.rectangle(img_open, tuple(sta[0:2]), tuple(sta[0:2] + sta[2:4]), (0, 0, 255), thickness=-1)
img_open：图像
tuple(sta[0:2])：矩形的一个顶点
tuple(sta[0:2] + sta[2:4])：矩形对角线上的另一个顶点
thickness：组成矩形的线条的粗细程度
![6BE1}J~`WG7ZW54V(X1)5SF](https://user-images.githubusercontent.com/106146337/172880671-05595ffc-7a71-4b14-b71b-5a308ac8e0a6.png)



### 3.4. 网络模块详细设计

采用Resnet（残差网络），残差网络的优势在于：

更易捕捉模型细微波动

更快的收敛速度
构建网络，ResNet模型

def get_net():

    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))第一个残差网络模块，包含2个残差块
    
    net.add_module("resnet_block2", resnet_block(64, 128, 2))第二个残差网络模块，包含2个残差块 
    
    net.add_module("resnet_block3", resnet_block(128, 256, 2))第三个残差网络模块，包含2个残差块
    
## 4. 数据设计

*注：如果需要对数据永久存储，即需要存储到文件或者数据库中，则需要对文件格式或数据库格式进行描述。对数据库，可以用一张表格介绍每一个关系模式中字段的名称、含义和类型。*

## 5. 系统部署

*注：用UML部署图或者类似的框图来描述系统在硬件上的部署方案。*

## 6. 其它事项

*注：上述章节中未尽事宜可在此描述。*


