import networkx as nx
import numpy as np
import os, sys
import json
import time
import operator
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from scipy.stats import entropy
import pickle
from pqdict import PQDict
import copy

from v_opt import VOpt

class Rout():
    def __init__(self, fpath, T, fpath_desty, fvedge_desty, fedge_desty, graph_store_name, degree_file, subpath, axes_file, pairs_name, speed_file, true_path):
        self.fpath = fpath
        self.hU = {}
        self.T = T
        self.sigma = 30
        self.vopt = VOpt()
        self.B = 3
        self.fpath_desty = fpath_desty
        self.fvedge_desty = fvedge_desty
        self.fedge_desty = fedge_desty
        self.graph_store_name  = graph_store_name
        self.degree_file =  degree_file
        self.subpath = subpath
        self.axes_file = axes_file
        self.pairs_name = pairs_name
        self.pairs_num = 90
        self.speed_file = speed_file
        self.true_path = true_path
        self.speed = 50
    
    def dplot(self, plot_data, name1, name2):
        #plt.hist(plot_data, bins = 4)
        label_x = ['0-5km', '5km-10km', '10km-50km', '50+km']
        #label_y = []
        plt.bar(x=range(len(label_x)),height=plot_data, width=0.4, alpha=0.8, color='red', )
        plt.xticks(range(len(label_x)), label_x)
        plt.savefig('./figure/'+name2+'_'+name1+'.png')
        plt.close()

    def get_tpath(self, points):
        with open(self.subpath+self.true_path) as js_file:
            true_path = json.load(js_file)
        TA, TB, TC, TD = [], [], [], []
        for tkey in true_path:
            a = tkey.find('-')
            a1 = tkey[:a]
            b = tkey.rfind('-')
            b1 = tkey[b+1:]
            tkey_ = a1+'-'+b1
            dist = self.get_distance(points, (a1, b1))
            valu = [float(l)/100 for l in true_path[tkey].values()]
            if dist < 5: TA.append((tkey, valu, a1, b1))
            elif dist < 10: TB.append((tkey, valu, a1, b1))
            elif dist < 25: TC.append((tkey, valu, a1, b1))
            else: TD.append((tkey, valu, a1, b1))
        print(len(TA))
        print(len(TB))
        print(len(TC))
        print(len(TD))
        A = sorted(np.random.randint(0, len(TA), self.pairs_num))
        B = sorted(np.random.randint(0, len(TB), self.pairs_num))
        C = sorted(np.random.randint(0, len(TC), self.pairs_num))
        D = sorted(np.random.randint(0, len(TD), self.pairs_num))
        ta = [TA[l] for l in A]
        tb = [TB[l] for l in B]
        tc = [TC[l] for l in C]
        td = [TD[l] for l in D]
        return [ta, tb, tc, td]


    def get_pairs(self, ):
        lines = [1706875, 1811547, 2160483, 41122]

        r_pairs = [[], [], [], []]
        for l, pname in enumerate(self.pairs_name):
            A = sorted(np.random.randint(0, lines[l], self.pairs_num))
            with open(pname) as pn:
                i, j = 0, 0
                for line in pn:
                    line = line.strip().split(',')
                    if i == A[j]:
                        r_pairs[l].append((line[1], line[0]))
                        j += 1
                        if j == self.pairs_num: break
                    i += 1
        return r_pairs

    def get_axes(self, ):
        fo = open(self.axes_file)
        points = {}
        for line in fo:
            line = line.split('\t')
            points[line[0]] = (float(line[2]), float(line[1]))
        return points

    def get_distance(self, points, point):
        (la1, lo1) = points[point[0]]
        (la2, lo2) = points[point[1]]
        return geodesic((lo1, la1), (lo2, la2)).kilometers

    def get_U(self, u_name):
        if u_name in self.hU:
            return self.hU[u_name]
        fn = open(self.fpath+u_name)
        U = {}
        for line in fn:
            line = line.strip().split(';')
            U[line[0]] = np.array([float(l) for l in line[1:]])
        self.hU[u_name] = U
        return U

    def get_dict(self, ):
        with open(self.subpath+self.fpath_desty) as js_file:
            path_desty = json.load(js_file)
        #with open(self.subpath+self.fvedge_desty) as js_file:
        #    vedge_desty = json.load(js_file)
        #    vedge_desty = dict(sorted(vedge_desty.items(), key=operator.itemgetter(0)))
        with open(self.subpath+self.fedge_desty) as js_file:
            edge_desty = json.load(js_file)
            edge_desty = dict(sorted(edge_desty.items(), key=operator.itemgetter(0)))
        vedge_desty = {}
        for p_k in path_desty:
            vedge_desty[p_k] = path_desty[p_k][0][1]
        return vedge_desty, edge_desty, path_desty

    def get_speed(self, ):
        speed_dict = {}
        with open(self.speed_file) as fn:
            for line in fn:
                line = line.strip().split('\t')
                speed_dict[line[0]] = {3600*float(line[1])/float(line[2]): 1.0}
        return speed_dict

    def get_graph(self, edge_desty, vedge_desty):
        speed_dict = self.get_speed()
        all_nodes, all_edges = set(), set()
        for key in speed_dict:
            line = key.split('-')
            all_nodes.add(line[0])
            all_nodes.add(line[1])
            all_edges.add(key)
        all_nodes, all_edges = list(all_nodes), list(all_edges)
        All_edges = []
        for edge in all_edges:
            edge_ = edge.split('-')
            if edge in edge_desty:
                cost1 = edge_desty[edge].keys()
                cost = min(float(l) for l in cost1)
                All_edges.append((edge_[0], edge_[1], cost))
            elif edge in speed_dict:
                cost1 = speed_dict[edge].keys()
                cost = min(float(l) for l in cost1)
                All_edges.append((edge_[0], edge_[1], cost))
        G2 = nx.DiGraph()
        G2.add_nodes_from(all_nodes)
        G2.add_weighted_edges_from(All_edges)


        fn = open(self.subpath+self.graph_store_name)
        edges, nodes = [], set()
        edges_ = set()
        l, l2 = 0, 0
        for line in fn:
            line = line.strip()
            cost = 0
            flag = '1'
            lline = ''
            if line not in edge_desty: #and line not in vedge_desty:
                if line in speed_dict:
                    cost1 = speed_dict[line]
                    cost = cost1
                    edge_desty[line] = cost
                    flag = 'a1'
                    lline = line
                else:
                    l += 1
                    continue
                #l += 1
            if line in edge_desty:
                cost1 = edge_desty[line].keys()
                cost = min(float(l) for l in cost1)
                flag = 'a2'
                lline = line
            #if line in vedge_desty:
            #    cost1 = vedge_desty[line]
            #    cost = min(abs(float(l)) for l in cost1)
            #    flag = 'a3'
            #    lline = line
            if cost < 0: 
                print(cost1)
                print(flag)
                print(lline)
            l2 += 1
            line = line.split('-')
            edges.append((line[0], line[1], cost))
            edges_.add(line[0] + '-' + line[1])
            nodes.add(line[0])
            nodes.add(line[1])
        fn.close()
        print('%d %d'%(l, l2))
 
        for node in all_nodes:
            if node not in nodes:
                nodes.add(node)
        for edge in speed_dict:
            if edge not in edges_:
                edg = edge.split('-')
                edges.append((edg[0], edg[1], speed_dict[edge].keys()[0]))
 
        nodes = list(nodes)
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(edges)
        self.speed_dict = speed_dict
        return edges, nodes, G2, speed_dict
 
    def conv(self, A, B):
        D = {}
        for a_k in A:
            for b_k in B:
                n_k = float(a_k) + float(b_k)
                n_v = float(A[a_k]) * float(B[b_k])
                D[n_k] = n_v
        if len(D) <= 3:
            return D
        D = dict(sorted(D.items(), key=operator.itemgetter(0)))
        new_w = self.vopt.v_opt(list(D.keys()), list(D.values()), self.B) 
        new_w = dict(sorted(new_w.items(), key=operator.itemgetter(0)))
        return new_w


    def get_min(self, U, node):
        return np.argwhere(U[node] > 0)[0][0]

    def cosin_distance(self, vector1, vector2):
        dot_product = 0.0
        normA = 0.0
        normB = 0.0
        for a, b in zip(vector1, vector2):
            dot_product += a * b
            normA += a ** 2
            normB += b ** 2
        if normA == 0.0 or normB == 0.0:
            return None
        else:
            return dot_product / ((normA * normB) ** 0.5)

    def lcs(self, A, B):
        print('A')
        print(A)
        print('B')
        print(B)
        def seq(K):
            K = K.strip().split(';')
            D = [k.split('-')[0] for k in K[:-1]]
            D_ = K[-1].split('-')
            D.append(D_[0])
            D.append(D_[1])
            return D
        A_, B_ = seq(A), seq(B)
        print('A_')
        print(A_)
        print('B_')
        print(B_)
        m, n = len(A_), len(B_)
        L = [[0] *(n) for i in range(m)]
        max = 0
        for i in range(m):
            for j in range(n):
                if A_[i] == B_[j]:
                    if i == 0 or j == 0:
                        L[i][j] = 1
                    else:
                        L[i][j] = L[i-1][j-1] + 1
                    if max < L[i][j]:
                        max = L[i][j]

        return max, len(A_), len(B_)

    def lcs2(self, A, B):
        print('A')
        print(A)
        print('B')
        print(B)
        def seq(K):
            K = K.strip().split(';')
            D = [k.split('-')[0] for k in K[:-1]]
            D_ = K[-1].split('-')
            D.append(D_[0])
            D.append(D_[1])
            return D
        A_, B_ = set(seq(A)), set(seq(B))
        print('A_')
        print(A_)
        print('B_')
        print(B_)
        max_ = len(A_ - (A_ - B_ ))
        return max_, len(A_), len(B_)

    '''
    def get_dijkstra(self, G, source, target):
        nodes = G.nodes
        dist = {} 
        Que = []
        has_visit = set()
        neigh = list(G.successors(start))
        prev = {}
        for node in neigh:
            if node in has_visit: continue
            else: has_visit.add(node)
            Que.append((node, G[start][node]['weight']))
            prev[node] = start
        while len(Que) != 0:
            (node, cost) = Que.pop(0)
            neigh = list(G.successors(node))
            for u in neigh:
                if u in has_visit: continue
                else: has_visit.add(u)
                Que.append((u, G[node][u]['weight']))
                prev[u] = node
                if u == target: break
        path = []
        
        for node in nodes:
            dist[node] = float("inf")
            prev = None
            Q.append(node, dist[node])
        dist[source] = 0
        while len(queue) > 0:
            min_dist = float("inf")
            min_node = None
            for v in Q:

            for n in queue:
                if dist[n][0] < min_dist and n not in seen:
                    min_dist = dist[n][0]
                    min_node = n
        return path'''

    def get_dijkstra3(self, G,  target):
        Gk = G.reverse()
        inf = float('inf')
        D = {target:0}
        Que = PQDict(D)
        P = {}
        nodes = Gk.nodes
        U = set(nodes)
        while U:
            #print('len U %d'%len(U))
            #print('len Q %d'%len(Que))
            if len(Que) == 0: break
            (v, d) = Que.popitem()
            D[v] = d
            U.remove(v)
            #if v == target: break
            neigh = list(Gk.successors(v))
            for u in neigh:
                if u in U:
                    d = D[v] + Gk[v][u]['weight']
                    if d < Que.get(u, inf):
                        Que[u] = d
                        P[u] = v
        
        '''
        dist, pred = D, P
        v = target
        path = [v]
        while v != source:
            v = pred[v]
            path.append(v)
        path.reverse()
        return path
        '''
        return P

    def get_dijkstra2(self, G, source, target):
        nodes = G.nodes
        distances = {node: 10e8 for node in nodes}
        previous = {node: None for node in nodes}
        distances[source] = 0
        Que = list(nodes)
        while len(Que) > 0:
            current = min(Que, key = lambda node: distances[node])
            Que.remove(current)
            if distances[current] == 10e8: break
            neighs = G.successors(current)
            for u in neighs:
                alters = distances[current] +G[current][u]['weight'] 
                if alters < distances[u]:
                    distances[u] = alters
                    previous[u] = current
        path, current = [], target
        while previous[current] is not None:
            path.append(current)
            current = previous[current]
        if path:
            path.append(current)
        return list(reversed(path))

    def rout(self, start, desti, edge_desty, vedge_desty, nodes_order, U, G, points, pred):
        path = []
        Q, P_hat = [], []
        neigh = list(G.successors(start))
        print('neigh %d'%len(neigh))
        start_ax = points[start]
        desti_ax = points[desti]
        s_d_arr = [desti_ax[0] - start_ax[0], desti_ax[1] - start_ax[1]]
        all_expire2 = 0.0
        #start1 = time.time()
        #pred = self.get_dijkstra3(G, desti)
        #all_expire2 += time.time()-start1 
        def get_weight(p_hat):
            w_p_hat = {}
            if p_hat in edge_desty:
                w_p_hat = edge_desty[p_hat]
            elif p_hat in vedge_desty:
                w_p_hat = vedge_desty[p_hat]
            return w_p_hat

        def get_maxP3(w_p_, vv, vi_getmin):
            p_max = 0.0 
            for pk in w_p_:
                w_pk = w_p_[pk]
                pk_ = int(int(pk) / self.sigma)
                if int(pk) % self.sigma == 0: pk_ += 1
                if (1.0*self.T - float(pk)) > vi_getmin:
                    p_max += float(w_pk)
            return p_max

        def get_vigetmin(vv):
            start1 = time.time()
            #path_ = self.get_dijkstra3(G, vv, desti)
            v = vv 
            path_ = [v]
            while v != desti:
                v = pred[v]
                path_.append(v)
            #path_.reverse()
            #print(path_)
            #sys.exit()
            spath = path_[0]
            vi_getmin = 0.0
            for epath in path_[1:]:
                key = spath+'-'+epath
                #if key in edge_desty:
                vi_getmin += min(abs(float(l)) for l in edge_desty[key].keys())
                #else:
                #    vi_getmin += min(abs(float(l)) for l in vedge_desty[key].keys())
                spath = epath
            expire_time = time.time()-start1
            return vi_getmin, expire_time

        has_visit = set()
        has_visit.add(start)
        QQue = PQDict.maxpq()
        Q = {}
        for vi in neigh:
            if vi in has_visit: continue
            else: has_visit.add(vi)
            p_hat = start +'-'+ vi
            w_p_hat = get_weight(p_hat)
            w_min = min([float(l) for l in w_p_hat.keys()])
            p_order = nodes_order[vi] #p_hat]
            vi_getmin, ex_p = get_vigetmin(vi)
            all_expire2 += ex_p
            cost_time = w_min + vi_getmin
            print('vi get min: %f'%vi_getmin)
            print('cost_time: %f'%cost_time)

            if cost_time <= self.T:
                #p_max = get_maxP3(w_p_hat, vi, vi_getmin)   
                p_max = max(list(w_p_hat.values()))
                #Q.append((p_hat, p_max, w_p_hat, cost_time))
                QQue[p_hat] = p_max
                Q[p_hat] = (p_max, w_p_hat, cost_time)
        print('len Q %d'%len(Q))
        #print(Q)
        #QQ, PP, KK = [], [], []
        QQ = {}
        p_best_p, flag = 'none', False
        p_max_m, p_best_cost, p_w_p = -1, -1, -1
        if len(Q) == 0: return 'none1', -1, -1, -1, all_expire2, -1
        all_rounds = 0
        while len(Q) != 0:
            (p_hat, pqv) = QQue.popitem()
            all_rounds += 1
            (p_max, w_p_hat, cost_time) = Q[p_hat]
            del Q[p_hat]
            #(p_hat, p_max, w_p_hat, cost_time) = Q.pop(0)
            #print(p_hat)
            a = p_hat.rfind('-')
            v_l = p_hat[a+1:]
            if v_l == desti:
                p_best_p = p_hat
                p_max_m = p_max
                p_best_cost = cost_time
                p_w_p = w_p_hat
                flag = True
                break
            neigh = list(G.successors(v_l))
            cost_sv = min([float(l) for l in w_p_hat.keys()])
            vd_d_arr = [points[desti][0]-points[v_l][0], points[desti][1]-points[v_l][1]]
            for u in neigh:
                if u == desti:
                    vu = v_l + '-' + u
                    w_vu = get_weight(vu)
                    if len(w_vu) == 0: cost_vu = 0
                    else: cost_vu = min([float(l) for l in w_vu.keys()])
                    vi_getmin, ex_p = get_vigetmin(u)
                    all_expire2 += ex_p
                    p_best_p = p_hat + ';' + vu
                    p_w_p = self.conv(w_p_hat, w_vu)
                    #p_max_m = get_maxP3(w_p_hat, u, vi_getmin)
                    p_max_m = max(list(p_w_p.values()))
                    p_best_cost = cost_sv + cost_vu + vi_getmin#inx_min*self.sigma
                    flag = True
                    break
                if u in has_visit: 
                    #print('u1 %s'%u)
                    continue
                else: has_visit.add(u)
                if u in p_hat: 
                    #print('u2 %s'%u)
                    continue
                vu = v_l + '-' + u
                w_vu = get_weight(vu)
                if len(w_vu) == 0:
                    #print('vu %s'%vu)
                    continue
                cost_vu = min([float(l) for l in w_vu.keys()])
                p_order = nodes_order[u] #p_hat]
                #vi_getmin = self.get_distance(points, (u, desti)) / self.speed * 3.6
                vi_getmin, ex_p = get_vigetmin(u)
                all_expire2 += ex_p 
                cost_time = cost_sv + cost_vu + vi_getmin#inx_min*self.sigma
                if cost_time <= self.T:
                    p_hat_p = p_hat + ';' + vu
                    w_p_hat_p = self.conv(w_p_hat, w_vu)
                    #p_hat_max = get_maxP3(w_p_hat, u, vi_getmin)
                    p_hat_max = max(list(w_p_hat_p.values()))
                    #QQ.append((p_hat_p, p_hat_max, w_p_hat_p, cost_time))
                    QQ[p_hat_p] = (p_hat_max, w_p_hat_p, cost_time)

                #else:
                #    print('large cost time')
                #    print(cost_time)
                #    print(vu)
            if flag: break
            if len(Q) == 0:
                #print('len QQ %d'%(len(QQ)))
                #Q = QQ[:]
                #QQ = []
                Q = copy.deepcopy(QQ)
                for qqk in QQ:
                    QQue[qqk] = QQ[qqk][0]
                QQ = {}
        return p_best_p, p_max_m, p_best_cost, p_w_p, all_expire2, all_rounds

    def get_T(self, start, desti, G, points):
        shortest_path = nx.shortest_path(G, start, desti)
        #print(shortest_path)
        l_sp = shortest_path[0]
        sums_d, sums_d2 = 0.0, 0.0
        for sp in shortest_path[1:]:
            spd = list(self.speed_dict[l_sp +'-'+ sp].keys())
            #print(spd)
            sums_d += self.get_distance(points, (l_sp, sp)) / self.speed * 3600
            #sums_d2 += spd[0]
            l_sp = sp
        #print(sums_d)
        #print(sums_d2)
        #sys.exit()
        return sums_d

    def get_T2(self, sspath):
        sspath = sspath.split(';')
        sums_d = 0
        for sp in sspath:
            spd = list(self.speed_dict[sp].keys())
            sums_d += spd[0]
        return sums_d 



    def main(self, ):
        #path_desty, vedge_desty, edge_desty = self.get_dict()
        vedge_desty, edge_desty, path_desty = self.get_dict()
        edges, nodes, G, speed_dict = self.get_graph(edge_desty, vedge_desty)
        #source = '585408'
        #target = '156809'
        #path = self.get_dijkstra3(G, source, target)
        #print(path)
        #sys.exit()
        points = self.get_axes()
        #r_pairs = self.get_tpath(points)
        #df2 = open('temp/r_pairs2.txt', 'rb')
        #df2 = open('temp/b3_pairs.txt', 'rb')
        df2 = open('test/new_temp4_.txt', 'rb')
        r_pairs = pickle.load(df2)
        df2.close()
        df3 = open('temp/odgt_30_b3pair.txt', 'rb')
        gt_data = pickle.load(df3)
        df3.close()

        #sys.exit()
        #r_pairs = self.get_pairs()
        nodes_order, i = {}, 0
        for node in nodes:
            nodes_order[node] = i
            i += 1

        #r_pairs = [[('289377', '21851')]]
        plot_data1, plot_data2 = [0]*5, [0]*5
        one_plot1, one_plot2 = [], []
        sums = [0] * 5 
        All_rounds = np.zeros(20).reshape(4, 5)
        One_Plot = np.zeros(20).reshape(4, 5)
        One_Plot2 = np.zeros(20).reshape(4, 5)
        One_Sums = np.zeros(20).reshape(4, 5)
        one_dis = -1
        stores = {}
        for pairs in r_pairs:
            one_dis += 1
            print('one dis: %d'%one_dis)
            #for pair_ in pairs:
            #    _ = self.get_U(pair_[-1])
            #for pairs in r_pairs:
            tstart = time.time()
            print('len pairs %d'%len(pairs))
            all_expires = 0.0
            sums2 = 0
            cost_t, cost_t2 = 0, 0
            for pair_ in pairs:
            #examp = [['77774' , '77773'], ['393129', '393121'], ['233966', '115217'], ['188605', '298874'], ['336662', '71213'], ['302115', '485873'], ['610692', '60727'], ['337153', '261429'], ['475782', '225428']]
            #examp = [['188605', '298874']]
            #for pair_ in examp: 
                print(pair_)
                start, desti = pair_[-2], pair_[-1]
                pred = self.get_dijkstra3(G, desti)
                path_, st1 = [start], start
                distan2 = 0
                while st1 != desti:
                    st2 = st1
                    st1 = pred[st1]
                    path_.append(st1)
                    distan2 += self.get_distance(points, (st2, st1))
                distan = self.get_distance(points, (start, desti))
                st_key = start + '+' + desti + ':'+str(distan)+':'+str(distan2)
                stores[st_key] = {}
                st1, time_budget = start, 0.0
                for st2 in path_[1:]:
                    sedge = st1+'-'+st2
                    if sedge in edge_desty:
                        speed_key = list([float(l) for l in edge_desty[sedge].keys()])
                        time_budget += max(speed_key)
                    elif sedge in speed_dict:
                        speed_key = list([float(l) for l in speed_dict[sedge].keys()])
                        time_budget += max(speed_key)
                    #elif sedge in vedge_desty:
                    #    speed_key = list([float(l) for l in vedge_desty[sedge].keys()])
                    #    time_budget += float(speed_key[0])
                    else: 
                        print(' edge: %s not in speed_dict, exit'%sedge)
                        sys.exit()
                    st1 = st2 
                print('time budget: %f'%time_budget)
                #self.T = self.get_T(start, desti, G, points)
                #self.T = self.get_T2(pair_[0])
                #self.T = self.T * 0.5

                for t_b_, t_b in enumerate([0.5, 0.75, 1.0, 1.25, 1.5]):
                    tstart = time.time()
                    self.T = time_budget * t_b
                    U = ''#self.get_U(desti)
                    best_p, max_m, best_c, best_pw, all_expire, all_rounds = self.rout(start, desti, edge_desty, vedge_desty, nodes_order, U, G, points, pred)
                    #stores[st_key][str(t_b)] = [time.time()-tstart, all_rounds]
                    stores[st_key][str(t_b)] = [time.time()-tstart-all_expire, all_rounds]
                    #print('distance %f km'%(self.get_distance(points, (start, desti))))
                    #print('best path %s'%best_p)
                    #print(len(best_p))
                    all_expires += all_expire
                    if best_p == 'none1': continue
                    if not isinstance(best_pw, dict): continue
                    tend = time.time()
                    plot_data1[t_b_] += tend - tstart
                    plot_data2[t_b_] += tend - tstart - all_expire
                    sums[t_b_] += 1

                    One_Plot[one_dis][t_b_] += tend - tstart 
                    One_Plot2[one_dis][t_b_] += tend - tstart  - all_expire
                    One_Sums[one_dis][t_b_] += 1
                    All_rounds[one_dis][t_b_] += all_rounds
                    print('cost time : %f'%(tend-tstart))
                    print('cost time2 : %f'%(tend-tstart-all_expire))
                    if t_b_ == 2:
                        sums2 += 1
                        cost_t += tend - tstart
                        cost_t2 += tend - tstart - all_expire
            #sys.exit()

            print('a cost time %f'%cost_t)
            print('a cost time2 %f'%cost_t2)
            print('sums2: %d'%sums2)
            
            one_plot1.append(round(cost_t/sums2, 4))
            one_plot2.append(round(cost_t2/sums2, 4))
        for i in range(5):
            if sums[i] == 0:
                print('zero %d'%i)
                continue
            plot_data1[i] /= sums[i]
            plot_data2[i] /= sums[i]
        print(plot_data1)
        print(plot_data2)
        print(sums)
        print('one plot, routing cost time for distance')
        print(one_plot1)
        print(one_plot2)
        print(One_Sums)
        One_Plot = One_Plot / One_Sums 
        One_Plot2 = One_Plot2 / One_Sums 
        print('One Plot')
        print(One_Plot)
        print(One_Plot.mean(0))
        print(One_Plot.mean(1))
        print('One Plot2')
        print(One_Plot2)
        print(One_Plot2.mean(0))
        print(One_Plot2.mean(1))
        print('All_rounds')
        print(All_rounds)
        print(All_rounds / One_Sums)
        All_rounds = All_rounds / One_Sums
        print(All_rounds.mean(0))
        print(All_rounds.mean(1))

        fname = 'mrev_1.json'
        with open(self.subpath + fname, 'w') as fw:
            json.dump(stores, fw, indent=4)


if __name__ == '__main__':

    pairs_name = ['./test/t16A', './test/t16B', './test/t16C', './test/t16D']
    threads_num = 15
    sigma = 30
    subpath = './res3/'
    fpath = './res3/umatrix2/'
    fpath = './res3/u_mul_matrix3/'
    fpath = './res3/u_mul_matrix_sig%d/'%sigma
    true_path = 'new_path_desty2.json'
    fpath_desty = 'KKdesty_num_%d.json'%threads_num #'new_path_desty1.json'
    #fvedge_desty = 'M_vedge_desty_num_%d.json'%threads_num
    fvedge_desty = 'M_vedge_desty2.json'
    fedge_desty = 'M_edge_desty.json'
    graph_store_name = 'KKgraph_%d.txt'%threads_num
    graph_store_name = 'Mgraph_10.txt'
    degree_file = 'KKdegree2_%d.json'%threads_num
    axes_file = '../../data/vertices.txt'
    speed_file = '../../data/AAL_NGR'
    time_budget = 5000
    rout = Rout(fpath, time_budget, fpath_desty, fvedge_desty, fedge_desty, graph_store_name, degree_file, subpath, axes_file, pairs_name, speed_file, true_path)
    rout.main()

