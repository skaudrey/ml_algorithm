# encoding: utf-8
#!/usr/bin/env python
'''
@Author: Mia
@Contact: fengmiao@meituan.com
@Software: PyCharm
@Site    : 
@Time    : 2018/7/26 下午6:53
@File    : PaintUtil.py
@Theme   : Painting utility module
'''

from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def sample_class_show(y,savepath='res.png'):
    '''
    pie chart, y-axis denotes labels list
    '''
    target_stats=Counter(y)
    labels=list(target_stats.keys())
    sizes=list(target_stats.values())
    explode=tuple([0.1] * len(target_stats))
    fig, ax=plt.subplots()
    ax.pie(sizes, explode=explode,labels=labels, shadow=True,autopct='%1.1f%%')
    ax.axis('equal')
    plt.show()
    return fig

def savefig(fig,figName, savepath='/fig'):
    fig.savefig("%s/%s" % (savepath,figName))


from collections import OrderedDict

def tick_category(df,rng_date,frequency, step):
    '''
    根据频率和频数获取刻度的时间轴
    frequency 可取值: 'year','month','day','hour','minute','second'
    :arg:rng_date : 时间轴索引
    '''

    if frequency=='year':
        df['frequency'] = [tmp.strftime('%Y-%m-%d') for tmp in rng_date]
    elif frequency=='month':
        df['frequency'] = [tmp.strftime('%Y-%m') for tmp in rng_date]
    elif frequency=='day':
        df['frequency'] = [tmp.strftime('%Y-%m-%d') for tmp in rng_date]
    elif frequency=='hour':
        df['frequency'] = [tmp.strftime('%Y-%m-%d %H') for tmp in rng_date]
    elif frequency=='minute':
        df['frequency'] = [tmp.strftime('%Y-%m-%d %H:%M') for tmp in rng_date]
    elif frequency=='second':
        df['frequency'] = [tmp.strftime('%Y-%m-%d %H:%M:%S') for tmp in rng_date]
    else:
        df['frequency'] = rng_date.strftime('%Y')

    num_date = OrderedDict()
    for item in df.groupby('frequency'):
        num_date[item[1].index[-1]] = item[0]

    nums = list(num_date.keys())[::step]
    dates = list(num_date.values())[::step]
    return num_date,nums,dates

def show_plot(df,frequency, step, columns, title,tick_count=10):
    '''
    >>> show_plot(df,'year',1, ['open', 'high','low','close'], tick_count=10)
    The 3rd parameter is the column names' list
    :param df:
    :param frequency:
    :param step:
    :param columns:
    :param tick_count:
    :return:
    '''
    num_date,nums,dates = tick_category(df,df['currenttime'],frequency, step)
    end_pos = nums[tick_count] if tick_count<len(nums) else nums[-1]
    data = df.loc[:,columns][:end_pos]
    axes = data.plot()
    axes.set_xticks(nums[:end_pos])
    axes.set_xticklabels(dates[:end_pos],rotation=45)
    plt.show()
    plt.title('userid--%s'%title)