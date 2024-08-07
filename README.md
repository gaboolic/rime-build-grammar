
执行cat merge_2_3.txt| ./build_grammar

但是这里注意build_grammar是macos下编译的，其他系统需要在各自系统下编译。编译方式参考https://github.com/gaboolic/librime/blob/master/.github/workflows/release-ci.yml

即可生成.gram文件

后续详细写说明

简要步骤：

1 收集语料

2 分词

3 生成.arpa文件

4 把arpa转成librime-octagram的tool用的格式

5 执行librime-octagram的build_grammar
