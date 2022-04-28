## 联邦学习端侧独立编译指导

1. 编译环境准备请参考<https://www.mindspore.cn/lite/docs/zh-CN/r1.6/use/build.html#linux>
2. 进入工程目录$mindspore/mindspore/lite/java/java/fl_client
3. 编译执行
     a. 执行`gradle build flReleaseJarAAR -x test`，生成供andriod使用的aar包
     b. 执行`gradle build flReleaseJarX86 -x test`，生成供X86 ST使用的jar包
     c. 执行`gradle build flUTJarX86 -x test`，生成供X86 UT使用的jar包
     注意：编译UT使用的jar包需要事先准备好模型包quick_start_flclient.jar，并将路径配置到settings.gradle，否则会导致单元测试包打包失败

## 联邦学习端侧X86 UT测试指导

1. 准备好端侧依赖的so， 请参考<https://www.mindspore.cn/federated/docs/zh-CN/r1.6/deploy_federated_client.html#id7>，并更新settings.gradle的test stask中的LD_LIBRARY_PATH
2. 准备号端侧依赖的模型jar包quick_start_flclient.jar，并更新settings.gradle的test stask中的MS_FL_UT_BASE_PATH
3. 可以直接在idea中执行test进行ut测试
4. 命令行ut测试需要先生成flUTJarX86包，具体的执行指令类似如下：

>LD_LIBRARY_PATH=${lite_x86_lib_path}   java   -javaagent:build/libs/jarX86UT/jmockit-1.49.jar  -cp build/libs/jarX86UT/mindspore-lite-java-flclient.jar org.junit.runner.JUnitCore com.mindspore.flclient.FLFrameUTRun
