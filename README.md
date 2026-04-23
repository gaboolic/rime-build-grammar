
Windows 下执行`Get-Content .\merge_2_3.txt | .\build_grammar.windows.exe zh-moqi`即可生成`.gram`文件

WSL / Linux 下执行`cat merge_2_3.txt | ./build_grammar.linux zh-moqi`即可生成`.gram`文件

mac下  执行`cat merge_2_3.txt| ./build_grammar` 即可生成`.gram`文件

## 简要步骤：

###  1 收集语料

可以参考[制作白霜词库的过程](https://moqiyinxing.chunqiujinjing.com/index/jin-jie-ji-shu-xi-jie/zhi-zuo-bai-shuang-ci-ku-de-guo-cheng)

### 2 分词

脚本见<https://github.com/gaboolic/rime-frost/blob/master/others/program/mnbvc/yuliao_fenci_to_txt.py>

###  3 生成.arpa文件

使用[kenlm](https://github.com/kpu/kenlm) 这个开源的语言模型库，编译后，进入build目录，把分词后的文本也放入build目录执行
`bin/lmplz -o 4 --verbose_header --text ./zhihu_deal_fenci_merge.txt --arpa MyModel/log.arpa  --prune 0 50 100`

`--prune 0 50 100`是剪枝，电脑配置好就不需要

即可生成.arpa文件

### 4 把arpa转成librime-octagram的tool用的格式 雨辰提供

先执行`python arpa.py log.arpa --output-dir output`

再执行`python merge_ngram.py --input-dir output --orders 2 3`合并arpa文件中的ngrams结果，获得`merge_2_3.txt`

### 5 执行librime-octagram的build_grammar

`cat merge_2_3.txt| ./build_grammar`

但是这里注意build_grammar是macos下编译的，其他系统需要在各自系统下编译。编译方式参考<https://github.com/gaboolic/librime/blob/master/.github/workflows/release-ci.yml>

项目里额外放了两个可执行文件：

- `build_grammar.windows.exe`：Windows 版本
- `build_grammar.linux`：WSL / Linux 版本

Windows 下可以这样生成 `.gram`：

`Get-Content .\merge_2_3.txt | .\build_grammar.windows.exe zh-moqi`

这样会在当前目录生成`zh-moqi.gram`。

WSL 下可以这样生成 `.gram`：

`cat merge_2_3.txt | ./build_grammar.linux zh-moqi`

这样也会在当前目录生成`zh-moqi.gram`。

如果是从 `.arpa` 开始，完整示例可以写成：

`python arpa.py .\lm_sc.arpa --output-dir .\lm_sc_output`

`python merge_ngram.py --input-dir .\lm_sc_output --orders 2 3`

然后在 Windows 下执行：

`Get-Content .\lm_sc_output\merge_2_3.txt | .\build_grammar.windows.exe lm_sc`

或者在 WSL 下执行：

`cd /mnt/d/vscode/moqi-input-method-projs/rime-build-grammar/lm_sc_output && cat merge_2_3.txt | ../build_grammar.linux lm_sc`
