
执行`cat merge_2_3.txt| ./build_grammar` 即可生成.gram文件

简要步骤：

1 收集语料

可以参考[制作白霜词库的过程](https://moqiyinxing.chunqiujinjing.com/index/jin-jie-ji-shu-xi-jie/zhi-zuo-bai-shuang-ci-ku-de-guo-cheng)

2 分词

脚本见<https://github.com/gaboolic/rime-frost/blob/master/others/program/mnbvc/yuliao_fenci_to_txt.py>

3 生成.arpa文件

使用[kenlm](https://github.com/kpu/kenlm) 这个开源的语言模型库，编译后，进入build目录，把分词后的文本也放入build目录执行
`bin/lmplz -o 4 --verbose_header --text ./zhihu_deal_fenci_merge.txt --arpa MyModel/log.arpa  --prune 0 50 100`

即可生成.arpa文件

4 把arpa转成librime-octagram的tool用的格式 雨辰提供

先执行`arpa.py`

再执行`merge_ngram.py`合并arpa文件中的ngrams结果，获得merge_2_3.txt

5 执行librime-octagram的build_grammar

`cat merge_2_3.txt| ./build_grammar`

但是这里注意build_grammar是macos下编译的，其他系统需要在各自系统下编译。编译方式参考<https://github.com/gaboolic/librime/blob/master/.github/workflows/release-ci.yml>
