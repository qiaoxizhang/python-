# python-
本项目是《python程序设计语言》课程期末实验项目及其说明文档
项目名称：基于 Python 语言的深度跨模态哈希检索程序的设计与实现
项目内容：使用 Python 语言设计并实现基于深度学习的跨模态哈希检索程序算法，
在三个广泛使用的基准数据集上进行相关实验，并且与一些先进的跨模态哈希算法进行
比较分析。

运行下列命令即可安装本项目所需的三方库：
pip install -r requirements.txt
数据集下载：
下载数据集文件和预训练模型。我们使用与 SCAN 相同的预提取的特征和分割, 
下载地址：(https://kuanghuei.github.io/SCANProject/)

数据预处理：
数据分割来源于网络，原始图像可以从以下来源下载：(http://cs.stanford.edu/people/karpathy/deepimagesent/)
(http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html)
(http://shannon.cs.illinois.edu/DenotationGraph/)
在 Preprocessing 下:
data_split_1.py: 划分训练集、测试集、验证集
resize_data_2.py: 长宽比例不变，将短边拉伸为 256
count_vocab_3.py: 统计每个单词的词频
convert_annotations_4.py: 将.txt 格式的标注文件转换为.json
build_dictionary_5.py: 构建单词编号，即查询字典

模型训练：
在数据预处理完成后，在 config.py 中配置各文件的路径以及训练的参数。
1. 训练/验证数据
train-data: 训练数据；val-data: 验证数据
2. 训练参数
image-model: 指定图像
text-model: 指定文本
context-length: 文本输入序列长度
batch-size: 训练时单卡 batch-size。（保证训练样本总数 > batch-size * GPU 数，至少
满足 1 个训练 batch）
max-steps: 训练步数，也可通过 max-epochs 指定训练轮数。
3. 输出

图文特征提取: 支持使用 GPU 单卡进行图文特征提取
cd Chinese-CLIP/
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:‘pwd‘/src
python -u src/eval/extract_features.py
–extract-image-feats
–extract-text-feats
–image-data=”$DATAPATH/datasets/$dataset_name/lmdb/$split/imgs”
–text-data=”$DATAPATH/datasets/$dataset_name/$split_texts.jsonl”
–img-batch-size=32
–text-batch-size=32
–context-length=24
–resume=$resume
–vision-model=ViT-B-16
–text-model=RoBERTa-wwm-ext-base-chinese

KNN 检索: 计算文本到图片，图片到文本检索的 topN 召回结果。文本到图片：
cd Chinese-CLIP/
python -u src/eval/make_topk_predictions.py
–image-feats=”$DATAPATH/datasets/$dataset_name/$split_imgs.img_feat.jsonl”
–text-feats=”$DATAPATH/datasets/$dataset_name/$split_texts.txt_feat.jsonl”
–topN=10
–eval-batch-size=32768
–output=”$DATAPATH/datasets/$dataset_name/$split_predictions.jsonl”

生成的结果保存在指定的 jsonl 文件中，每行表示一个文本召回的 top-k 图片 id。
Recall 计算: 类似的，此处只给出文本到图片检索命令：
split=valid 指定计算 valid 或 test 集特征 python src/eval/evaluation.py
$DATAPATH/datasets/$dataset_name/$split_texts.jsonl
$DATAPATH/datasets/$dataset_name/$split_predictions.jsonl
