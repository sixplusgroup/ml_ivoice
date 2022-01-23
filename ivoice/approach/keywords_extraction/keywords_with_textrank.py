from jieba import analyse


def extract_keywords(content, count=3):
    return analyse.textrank(content, count)

# print('-'*40)
# print(' TF-IDF')
# print('-'*40)
#
# s = u"此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。"
#
# s = u"你好！你好！今天的天气怎么样？北京市今天的天气相当好。设定明天早上八点的闹钟。闹钟已经设定到明天早上9点。百度查询马云。以下是我找到的资料。"
#
# for x, w in analyse.extract_tags(s, withWeight=True):
#     print('%s %s' % (x, w))
#
# print('-'*40)
# print(' TextRank')
# print('-'*40)
#
# a = analyse.textrank(s, 3)
# # print('a ', [x for x, _ in a[:3]])
# print('a ', a)
#
# print('='*40)
