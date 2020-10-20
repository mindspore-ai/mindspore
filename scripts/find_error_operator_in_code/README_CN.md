# 算子报错定位对应脚本源码
## 文档功能与适用场景
  在MindSpore进行计算调试，遇到算子报错时，用户希望能够通过报错信息查找到对应的Python源码。现有的算子编译(PreCompileProcessFailed)和执行失败(run task error)只是将算子名字打印到终端，无法直接指出用户代码中算子调用处的行号。本文的主要目的为指导用户通过算子的报错信息，然后在ANF图文件里找到对应算子的源码。   
  此指导文档适合运行在 **Ascend硬件** 环境下的计算，且忽略算子融合场景和算子反向报错的情况。
## 解析流程与辅助工具使用
 1. 查找对应源码行号的主要解析流程为：  
   ① 获取报错算子的全称full_name(full_name由scope和算子名组成，scope名会以Default或Gradients开头，Default表示正向，Gradients表示反向，这里我们只关注Default的情况)。  
   ② 获取报错算子的输入输出属性等信息。  
   ③ 通过full_name和输入输出信息在[0-10]_py_pre_ad.dat(优先)或[0-10]_validate.dat文件中找到对应的算子和对应的代码行号。  

 2. 算子报错时使用脚本的3步操作：  
   ① 用户在训练脚本里设置context.set_context(mode=context.GRAPH_MODE, save_graphs=True)，进行图文件的保存。  
   ② 用户在执行代码时将日志信息定向到文件中，使用方法为：python xxx.py &> log_name &  
   ③ 通过辅助脚本解析报错算子全称、输入输出信息，同时尝试查找对应代码行号。  
   &nbsp; 脚本名： **find_error_operator_in_code.py**  
   &nbsp; 执行方式:
    ```
    python3 find_error_operator_in_code.py 
            --log_path [the path of log, default is the current path](option)
            --log_name [the file name of log](required, log_name)
    ```
 3. 解析效果
   解析文件时通常有3种情况：  
   ① 全匹配(算子名称、输入输出的shape与dtype都匹配时)：
    ```
    [INFO] Detect "task exception error".                                 【检测报错类型】
    [INFO] Find operation 1 times!                                        【在python源码中查找匹配到的次数】
    [INFO] In file test_mul.py(29)/ return self.mul(x, y)                 【查找到的文件名、行号与该行号的代码内容】
    [INFO] Exception operator is "Default/Mul".                           【出错的算子】
    [INFO] Have 2 input in operator:                                      【该出错算子有2个输入】
           input 1/1th: dtype is float16, shape is [526338, 21].   【第一个输入的dtype和shape，1/1th表示第一个输入的第1个值】
           input 1/2th: dtype is float16, shape is [526338, 1].    【第二个输入的dtype和shape】
    [INFO] Have 1 output in operator:                                     【该出错算子有1个输出】
           output 1/1th: dtype is float16, shape is [526338, 21].  【第一个输出的dtype和shape，1/1th表示第一个输出的第1个值】
    ```
    ② 单匹配(算子名称匹配、但只匹配输入或输出的shape与dtype都匹配时):
    
    ```
    [INFO] Detect "compile error".                                                              【检测报错类型】
    [INFO] Find operation 1 times!                                                              【在python源码中查找匹配到的次数】
    [INFO] In file split_ops.py(17)/ return self.net(input)                                     【查找到的文件名、行号与该行号的代码内容】
    [INFO] Exception operator is "Default/Split".                                               【出错的算子】
    [INFO] Have 1 input in operator:                                                            【该出错算子有1个输入】
           input 1/1th: dtype is float32, shape is [32, 192, 56, 56].                    【第一个输入的dtype和shape，1/1th表示第一个输入的第1个值】
    [WARNING] Cannot match output information! Please check whether the operator's output is:   【输出未完全匹配告警】
    [INFO] Have 1 output in operator:                                     　　　　　　　　　　　  【该出错算子有1个输出】
           output 1/1th: dtype is float32, shape is [32, 6, 56, 56]. 　　　　　　　　　　　【第一个输出的dtype和shape，1/1th表示第一个输出的第1个值】
           output 2/1th: dtype is float32, shape is [32, 6, 56, 56].　　　　　　　　　　　 【第一个输出的dtype和shape，2/1th表示第一个输出的第2个值】
           output 3/1th: dtype is float32, shape is [32, 6, 56, 56].  　　　　　　　　　　 【第一个输出的dtype和shape，3/1th表示第一个输出的第3个值】
    ```
    ③ 未匹配(未匹配到算子名称，或者匹配算子名称但输入输出未匹配时):
    
    ```
    [INFO] Detect "task exception error".                                                              【检测报错类型】
    [WARNING] Cannot find operation! Need to find in the script based on the following information:    【未在源码中匹配到算子告警】
    [INFO] Exception operator full name is "Default/test".                                             【出错的算子】
    [INFO] Have 2 input in operator:                                                                   【该出错算子有2个输入】
           input 1/1th: dtype is float16, shape is [526338, 21].                                【第一个输入的dtype和shape，1/1th表示第一个输入的第1个值】
           input 1/2th: dtype is float16, shape is [526338, 1].                                 【第二个输入的dtype和shape】
    [INFO] Have 1 output in operator:                                                           　　　　【该出错算子有1个输出】
           output 1/1th: dtype is float16, shape is [526338, 21].                               【第一个输出的dtype和shape，1/1th表示第一个输出的第1个值】
    [WARNING] Do you want to research in source code? set source code path to research or press enter to research in current path, input n/no to exit.
    Input:                                                                                             【未匹配到代码时会提示是否在脚本中进行搜索，n/no表示不搜索，传入代码路径在指定路径搜索，回车默认表示当前路径搜索】
    ```
    
 4. 手动代码查找  
   这里还会存在些特殊情况，例如用户在源码中调用框架提供的nn.cell网络层，会发现查找出来的为框架里的代码行号。此时用户需要利用工具查找出的full_name和输入输出信息回到源码中进行对应代码的查找。  
   举个例子说明如何手动在代码中查找指定full_name和shape的算子，例如full_name为: Default/network/network/aspp/aspp_pooling/ResizeNearestNeighbor，输入的shape为[8, 256, 1, 1]， dtype为float32。    
   可以观察到其scope为: Default/network/network/aspp/aspp_pooling，算子名为: ResizeNearestNeighbor。注意：scope中会存在Default、network自动填充，Default表示正向，network为网络名。  
   查看以下用户定义的代码，首先我们先分析scope: Default/network/network/aspp/aspp_pooling。由network/aspp可定位到算子的定义与调用处分别为26行与31行，继续由network/aspp/aspp_pooling，可以定位到定义与调用处分别为4行与8行，然后通过算子名ResizeNearestNeighbor可以定位至定义与调用处分别为16行与19行。最后若存在相同scope下存在相同的算子名时，需要通过输入的shape和dtype进行进一步判断。    
    ```
      1 class ASPP(nn.Cell):
      2     def __init__(self):
      3         super(ASPP, self).__init__()
      4         self.aspp_pooling = ASPPPooling()
      5         self.drop = nn.Dropout(0.3)
      6
      7     def construct(self, x):
      8         x5 = self.aspp_pooling(x)
      9         x = self.drop(x)
     10         return x
     11
     12 class ASPPPooling(nn.Cell):
     13     def __init__(self):
     14         super(ASPPPooling, self).__init__()
     15         self.shape = P.Shape()
     16         self.resizenearestneighbor = P.ResizeNearestNeighbor((size[2], size[3]), True)
     17     def construct(self, x):
     18         size = self.shape(x)
     19         out = self.resizenearestneighbor(out)
     20         return out
     21
     22 # 主结构
     23 class DeepLabV3(nn.Cell):
     24     def __init__(self, phase='train', num_classes=21, output_stride=16, freeze_bn=False):
     25         super(DeepLabV3, self).__init__()
     26         self.aspp = ASPP()
     27         self.shape = P.Shape()
     28
     29     def construct(self, x):
     30         size = self.shape(x)
     31         out = self.aspp(out)
     32         return out
    ```
