# encoding: utf-8
#!/usr/bin/env python
'''
@Author: Mia
@Contact: fengmiao@meituan.com
@Software: PyCharm
@Site    : 
@Time    : 2018/8/10 上午7:03
@File    : communityDetect.py
@Theme   :
'''


import networkx as nx

import sys
import networkx as nx
import time


def find_community(graph, k):
    return list(nx.k_clique_communities(graph, k))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Usage: %s <InputEdgeListFile>" % sys.argv[0]
        sys.exit(1)

    # 创建一个无向、无权图
    edge_list_file = sys.argv[1]
    wbNetwork = nx.read_edgelist(edge_list_file, delimiter='\t')
    print "图的节点数：%d" % wbNetwork.number_of_nodes()
    print "图的边数：%d" % wbNetwork.number_of_edges()

    # 调用kclique社区算法
    for k in xrange(3, 6):
        print "############# k值: %d ################" % k
        start_time = time.clock()
        rst_com = find_community(wbNetwork, k)
        end_time = time.clock()
        print "计算耗时(秒)：%.3f" % (end_time - start_time)
        print "生成的社区数：%d" % len(rst_com)
