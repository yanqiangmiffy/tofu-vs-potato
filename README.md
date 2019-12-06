# tofu-vs-potato
美食识别挑战（1）：豆腐VS土豆

## CV分类学习
### 尝试1 

使用VGG16/ResNet50/InceptionV3预训练模型，输入做了一些数据增强，效果不好，过拟合
线上得分：75.4537

### 尝试2

使用简单的CNN网络，效果不佳

线上得分：52

大佬建议1：
```text
@致Great 2层CNN可能不够哦
两层最多也就提取一些曲线特征
```

大佬建议2：
```text
参照vgg16多搞几层卷积

加入数据增强防止过拟合
```

### 尝试3
[阿水baseline](https://github.com/datawhalechina/competition-baseline/tree/master/competition/AI%E7%A0%94%E4%B9%A0%E7%A4%BE-%E7%BE%8E%E9%A3%9F%E8%AF%86%E5%88%AB%E6%8C%91%E6%88%98%EF%BC%881%EF%BC%89%EF%BC%9A%E8%B1%86%E8%85%90VS%E5%9C%9F%E8%B1%86 )
线上得分：98.8539

## 其他问题
- 使用os.listdir()存在文件名是按照字典序排序的，需要与label对应的文件专为一致
https://www.cnblogs.com/jins-note/p/9550388.html