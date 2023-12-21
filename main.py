# Times: 2023-12-21 Deadline:2023-12-22
# Author: 秋天红叶(Wechat)
# Python Environment: Python 3.11
# Editor: Pycharm

# 导入包
import re
import requests
import xml.etree.ElementTree as ET
import csv
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import textblob
import jieba
from gensim.models import Word2Vec

'''
文本数据挖掘作业：

1、对互联网中的用户评论进行情感分析，可以是网购的评论，电影或者书或音乐的网站评论，或者弹幕评论，
建议是和购物相关的评论。
要求：
①评论不少于一千条                                            √
②挖掘出不少于4张的可视化图片（至少包含一张形状词云图）             √
③使用 python 程序实现。                                      √
2、文本内容不限，可以用于做主题分析（csv格式）
要求：
①自主分析关键词，展示向量化文本分析的结果，包含星空图。             √
②实现LDA主题模型：给出困惑度曲线和每个主题对应词语（截图）。         √

问题回容格式：①介绍数据来源（文本情况概述）②介绍所用挖掘技术③可视化挖掘结果。
'''

def extract_danmus(xml_file, csv_file):
    # 解析XML文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 打开CSV文件准备写入
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入列标题
        writer.writerow(['内容', '时间戳'])

        # 遍历XML文件中的所有弹幕节点
        for danmu in root.findall('d'):
            # 获取弹幕内容和时间戳
            content = danmu.text
            timestamp = danmu.attrib['p'].split(',')[0]
            # 写入CSV文件
            writer.writerow([content, timestamp])

# 情感分析
def analyze_sentiment(text):
    analysis = textblob.TextBlob(text)
    return analysis.sentiment.polarity

# 执行函数
if __name__ == '__main__':
    try:
        # Bilibili视频或番剧页面的URL
        url = "https://www.bilibili.com/festival/ymmxt2?bvid=BV19e411o7rY&spm_id_from=333.999.list.card_archive.click"

        # 设置用户代理
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

        # 发送HTTP请求获取页面内容
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            # 使用正则表达式从页面内容中提取cid
            cid = re.findall(r'"cid":(.*?),', res.text)[-1]

            # 构建弹幕文件的URL
            url = f'https://comment.bilibili.com/{cid}.xml'

            # 获取弹幕文件内容
            res = requests.get(url)
            if res.status_code == 200:
                # 将弹幕内容写入到本地XML文件
                with open(f'{cid}.xml', 'wb') as f:
                    f.write(res.content)
                print("弹幕文件下载成功")
            else:
                print("获取弹幕文件失败")
        else:
            print("获取页面内容失败")
    except Exception as e:
        print(f"出现错误: {e}")

    # 调用函数
    extract_danmus(f'{cid}.xml', 'danmus.csv')

    '''
    对弹幕数据进行挖掘
    '''

    # 读取弹幕文本
    # 确保使用正确的编码读取CSV文件
    df = pd.read_csv('danmus.csv', encoding='utf-8')
    text = ' '.join(df['内容'])  # 替换 '弹幕内容列名' 为CSV中对应的列名

    # 加载形状图片
    # mask = np.array(Image.open('shape.png'))

    # 生成词云
    wordcloud = WordCloud(
        background_color="white",
        # mask=mask,
        contour_width=3,
        contour_color='steelblue',
        font_path='C:\Windows\Fonts\STZHONGS.TTF'
    ).generate(text)  # 替换 '字体路径' 为适用于中文的字体路径，如果您的系统中有适用于中文的默认字体，可以省略font_path参数

    # 显示词云图
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis('off')
    # plt.show()

    wordcloud.to_file("wordcloud.jpg")

    # 进行词频统计
    word_counts = df['内容'].value_counts().head(20)

    # 绘制直方图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    word_counts.plot(kind='bar')
    plt.title('Top 20 Most Frequent Words in Danmus')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.savefig("wordcounts.jpg")
    plt.show()

    #时间序列图

    # 将时间戳那一列转换为时间数据类型
    df["时间戳"] = pd.to_datetime(df["时间戳"], unit="s")

    # 设置图表样式
    plt.style.use('ggplot')

    # 按时间戳分组,统计弹幕数量
    contents = df.groupby(df["时间戳"]).size()

    # 绘制时间序列折线图
    contents.plot()
    plt.xlabel("时间")
    plt.ylabel("弹幕数量")
    plt.title("弹幕时间序列图")
    plt.tight_layout()
    plt.savefig("danmu_time_series.png")
    plt.show()

    #情感分析图
    df["sentiment"] = df["内容"].apply(analyze_sentiment)
    sentiment_count = df.groupby([pd.cut(df.sentiment, 5)])["内容"].count()
    sentiment_count.plot(kind="bar")
    plt.title("情感分析图")
    plt.savefig("sentimental_analysis.jpg")
    plt.show()

    '''
    主题分析
    '''

    # 读取并连接所有弹幕内容,作为文本corpus
    df = pd.read_csv("danmus.csv")
    corpus = " ".join(df["内容"])

    # 使用jieba中文分词,切分文本
    txt = list(jieba.cut(corpus))

    # 基于切分后的文本生成词向量模型
    model = Word2Vec(txt)
    # keywords = extract_keywords(model, txt)
    from sklearn.manifold import TSNE

    words = list(model.wv.key_to_index)  # 替代 model.wv.vocab
    vectors = [model.wv[w] for w in words]  # 替代 model[w]

    tsne = TSNE(n_components=2)
    embed = tsne.fit_transform(model.wv.vectors)

    x = [p[0] for p in embed]
    y = [p[1] for p in embed]

    plt.figure(figsize=(14, 14))
    plt.scatter(x, y)

    for i in range(len(x)):
        plt.annotate(words[i], (x[i], y[i]))
    plt.savefig("Star.jpg")
    plt.show()

    # LDA模型
    '''
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    import pyLDAvis
    import pyLDAvis.sklearn

    df = pd.read_csv("danmus.csv")
    contents = df["内容"]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(contents)

    # 训练模型,评估不同主题数
    num_topics = [2, 5, 7, 10, 15]
    perplexities = []
    for n in num_topics:
        lda = LatentDirichletAllocation(n_components=n)
        perplexities.append(lda.fit(X).perplexity(X))

    # 绘制困惑度曲线
    plt.plot(num_topics, perplexities);
    plt.xlabel("# Topics");
    plt.ylabel("Perplexity");

    # 输出最佳主题数下的主题词
    n = 5  # 最佳主题数
    lda_model = LatentDirichletAllocation(n_components=n)
    lda_model.fit(X)

    for topic_id in range(n):
        print(f"Topic {topic_id}:")
        print(vectorizer.inverse_transform(lda_model.components_[topic_id]).sum(axis=0))

    # 可视化
    vis_data = pyLDAvis.sklearn.prepare(lda_model, X, vectorizer)
    pyLDAvis.save_html(vis_data, 'lda_danmu.html')
    '''
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    import pyLDAvis.sklearn
    import matplotlib.pyplot as plt

    # 读取数据
    df = pd.read_csv("danmus.csv")
    contents = df["内容"]

    # 初始化 CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(contents)

    # 训练模型，评估不同主题数
    num_topics = [2, 5, 7, 10, 15]
    perplexities = []
    for n in num_topics:
        lda = LatentDirichletAllocation(n_components=n)
        lda.fit(X)
        perplexities.append(lda.perplexity(X))

    # 绘制困惑度曲线
    plt.plot(num_topics, perplexities)
    plt.xlabel("# Topics")
    plt.ylabel("Perplexity")
    plt.show()

    # 选择最佳主题数并输出主题词
    n = 5  # 假设5是最佳主题数
    lda_model = LatentDirichletAllocation(n_components=n)
    lda_model.fit(X)

    for topic_id, topic in enumerate(lda_model.components_):
        print(f"Topic {topic_id}:")
        top_feature_indices = topic.argsort()[-10:][::-1]
        top_features = [vectorizer.get_feature_names_out()[i] for i in top_feature_indices]
        print(" ".join(top_features))

    # 可视化
    vis_data = pyLDAvis.sklearn.prepare(lda_model, X, vectorizer)
    pyLDAvis.save_html(vis_data, 'lda_danmu.html')





