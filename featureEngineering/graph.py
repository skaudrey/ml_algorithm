#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/13 14:26
# @Author  : MiaFeng
# @Contact : skaudrey@163.com
# @Site    :
# @File    : graph.py
# @Software: PyCharm
__author__ = 'MiaFeng'

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris



class CBindMobileChargeMobileGraph:
    def __init__(self):
        pass

    def createGraph(self,matrixItm,V):
        # 创建图
        G = nx.Graph()  # 建立一个空的无向图
        # v = range(matrix.shape[0])  # 一维行向量，从1到8递增
        G.add_nodes_from(V)  # 从v中添加结点，相当于顶点编号为1到8
        # line = file_color.read()  # 读取颜色向量
        # colors = (line.split(' '))  # 颜色向量
        # for i in range(len(colors)):
        #     colors[i] = int(colors[i])  # 将字符转为数字


        for edge in zip(matrixItm.index,matrixItm['cnt']):
            nodes = edge[0][:].split('_')
            node1 = nodes[0]
            node2 = nodes[1]

            G.add_edge(node1,node2,weight=edge[1])

        return G

    def weightAdjust(self):
        pass

    def drawGraph(self,G,bigWeightThreshold,saveFileName):
        # 按权重划分为重权值得边和轻权值的边
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > bigWeightThreshold]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= bigWeightThreshold]
        # 节点位置
        pos = nx.spring_layout(G)  # positions for all nodes
        # 首先画出节点位置
        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=400)
        # 根据权重，实线为权值大的边，虚线为权值小的边
        # edges
        nx.draw_networkx_edges(G, pos, edgelist=elarge,
                               width=6)
        nx.draw_networkx_edges(G, pos, edgelist=esmall,
                               width=6, alpha=0.5, edge_color='b', style='dashed')

        # labels标签定义
        nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

        plt.axis('off')

        plt.show()  # display

        plt.savefig("weighted_graph.png",dpi=500)  # save as png

    def find_community(self,G, k):
        return list(nx.k_clique_communities(G, k))


if __name__=='__main__':
    V = pd.read_csv('mobileMap.csv',sep='\t')   # Read Vertices

    matrixItm = pd.read_csv('bankChargeMobileCocurrence.csv', sep='\t',index_col=0)
    # Read edges, which is weighted by the co-current matrix


    # matrixItm.set_index()

    graphHandler = CBindMobileChargeMobileGraph()

    G = graphHandler.createGraph(matrixItm,V)

    # graphHandler.drawGraph(G,2,'%sbindBankMobileCocurrenceGraph.png' % getPath(category='fig'))

    result = graphHandler.find_community(G,3)

    print result[0]

    X,y = load_iris()

    df_neg_charge_visit = pd.DataFrame(data = [False]*len(X['chargemoblie'].drop_duplicates().values),
                                       index = X['chargemoblie'].drop_duplicates().values,columns='charge')

    df_neg_charge_mobile = X['chargemoblie']

    # print df_pos_charge_visit.loc[['18702480951'],:]

    hitCnt = 0

    mobileList = set()

    for idx in xrange(len(result)):
        for i in result[idx]:
            mobileList.add(i)

    for i in mobileList:
        print df_neg_charge_visit.get_value(i, ['charge'])
        if (i in df_neg_charge_mobile) and (df_neg_charge_visit.get_value(i, ['charge']) != None):
            if df_neg_charge_visit.loc[[i], 0][0] == False:
                hitCnt = hitCnt + 1
                df_neg_charge_visit[i] = True


    print hitCnt


