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
from math import sin, cos, sqrt, atan2, radians

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
        lat1, lon1, lat2, lon2 = radians(lo1), radians(la1), radians(lo2), radians(la2)
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        R = 6373.0
        distance = R * c
        #return distance
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
            if line not in edge_desty :#and line not in vedge_desty:
                if line in speed_dict:
                    edge_desty[line] = speed_dict[line]
                    cost = speed_dict[line]
                else:
                    l += 1
                    continue
                #l += 1
            if line in edge_desty:
                cost = min(float(l) for l in edge_desty[line].keys())
            l2 += 1
            line = line.split('-')
            edges.append((line[0], line[1], cost))
            edges_.add(line[0] + '-' + line[1])
            nodes.add(line[0])
            nodes.add(line[1])
        fn.close()
        print('%d %d'%(l, l2))
        '''
        for node in all_nodes:
            if node not in nodes:
                nodes.add(node)
        for edge in speed_dict:
            if edge not in edges_:
                edg = edge.split('-')
                edges.append((edg[0], edg[1], speed_dict[edge].keys()[0]))
        '''     
        nodes = list(nodes)
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(edges)
        self.speed_dict = speed_dict
        return edges, nodes, G2, speed_dict

    #def get_graph2(self, edge_desty, vedge_desty):
    def get_graph2(self, edge_desty, vedge_desty):
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

        return all_edges, all_nodes, G2, speed_dict

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
        return P


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
 
    #def rout(self, start, desti, edge_desty, vedge_desty, nodes_order, U, G, points):
    #    #def get_dijkstra(self, source, target):
    #    path = nx.dijkstra_path(G, start, desti)
    #    return path

    def rout(self, start, desti, edge_desty, vedge_desty, speed_dict, nodes_order, U, G, points):
        path = []
        Q, P_hat = [], []
        neigh = list(G.successors(start))
        print('neigh %d'%len(neigh))
        start_ax = points[start]
        desti_ax = points[desti]
        s_d_arr = [desti_ax[0] - start_ax[0], desti_ax[1] - start_ax[1]]
        all_expire = 0.0
        start1 = time.time() 
        def get_weight(p_hat):
            w_p_hat = {}
            if p_hat in edge_desty:
                w_p_hat = edge_desty[p_hat]
            elif p_hat in speed_dict:
                w_p_hat = speed_dict[p_hat]
            elif p_hat in vedge_desty:
                w_p_hat = vedge_desty[p_hat]
            return w_p_hat

        def get_maxP(w_p_, vv):
            p_max = 0.0
            for pk in w_p_:
                w_pk = w_p_[pk]
                pk_ = int(int(pk) / self.sigma)
                if int(pk) % self.sigma == 0: pk_ += 1
                p_max += float(w_pk ) * U[vv][pk_]
            return p_max

        def get_maxP2(w_p_, vv):
            p_max = 0.0 
            start1 = time.time() 
            vi_getmin = self.get_distance(points, (vv, desti)) / self.speed * 3600
            for pk in w_p_:
                w_pk = w_p_[pk]
                pk_ = int(int(pk) / self.sigma)
                if int(pk) % self.sigma == 0: pk_ += 1
                if (1.0*self.T - float(pk)) > vi_getmin:
                    p_max += float(w_pk)
            all_expire += time.time() - start1
            return p_max

        has_visit = set()
        has_visit.add(start)
        Que = PQDict.maxpq()
        Q = {}
        print('T: %f'%self.T)
        for vi in neigh:
            if vi in has_visit: continue
            else: has_visit.add(vi)
            p_hat = start +'-'+ vi
            w_p_hat = get_weight(p_hat)
            w_min = min([float(l) for l in w_p_hat.keys()])
            p_order = nodes_order[vi] #p_hat]
            start1 = time.time() 
            vi_getmin = self.get_distance(points, (vi, desti)) / self.speed * 3600
            all_expire += time.time() - start1
            #print('vi_getmin3: %f'%vi_getmin)
            #print('dis :%f'%self.get_distance(points, (vi, desti)))
            #inx_min = np.argwhere(U[vi] > 0)
            #if len(inx_min) == 0: continue
            #inx_min = inx_min[0][0]
            cost_time = w_min + vi_getmin
            #print('T: %d, cost time: %d'%(self.T, cost_time))
            if cost_time <= self.T:
                #p_max = get_maxP2(w_p_hat, vi)   
                p_max = max(list(w_p_hat.values()))
                #p.maxin = *U[nodes_order[p_hat]]
                #Q.append((p_hat, p_max, w_p_hat, cost_time))
                Que[p_hat] = p_max
                Q[p_hat] = (p_max, w_p_hat, cost_time)
        print('len Q %d'%len(Q))
        #print(Q)
        #QQ, PP, KK = [], [], []
        QQ = {}
        p_best_p, flag, p_max_m, p_best_cost, p_w_p = 'none', False, -1, -1, -1
        if len(Q) == 0: return 'none1', -1, -1, -1, all_expire, -1
        all_counts = 0
        while len(Q) != 0:
            (p_hat, pqv) = Que.popitem()
            all_counts += 1
            (p_max, w_p_hat, cost_time) = Q[p_hat]
            del Q[p_hat]
            #print(len(Q))
            #print(len(Que))
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
                    start1 = time.time() 
                    vi_getmin = self.get_distance(points, (u, desti)) / self.speed * 3600
                    all_expire += time.time() - start1
                    #print('vi_getmin1: %f'%vi_getmin)
                    #print('dis :%f'%self.get_distance(points, (vi, desti)))
                    #inx_min = np.argwhere(U[u] > 0)
                    #inx_min = inx_min[0][0]
                    p_best_p = p_hat + ';' + vu
                    p_w_p = self.conv(w_p_hat, w_vu)
                    #p_max_m = get_maxP2(w_p_hat, u)
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
                start1 = time.time() 
                vi_getmin = self.get_distance(points, (u, desti)) / self.speed * 3600
                all_expire += time.time() - start1
                #print('vi_getmin2: %f'%vi_getmin)
                #print('dis :%f'%self.get_distance(points, (vi, desti)))
                cost_time = cost_sv + cost_vu + vi_getmin#inx_min*self.sigma
                #print('qq %d'%(cost_time))
                if cost_time <= self.T:
                    p_hat_p = p_hat + ';' + vu
                    w_p_hat_p = self.conv(w_p_hat, w_vu)
                    #p_hat_max = get_maxP2(w_p_hat, u)
                    p_hat_max = max(list(w_p_hat_p.values()))
                    #if self.check_domin3(w_p_hat_p, QQ):
                    #QQ.append((p_hat_p, p_hat_max, w_p_hat_p, cost_time))
                    QQ[p_hat_p] = (p_hat_max, w_p_hat_p, cost_time)
                    #Que[p_hat_p] = p_hat_max
                    #has_visit.add(p_hat_p)
                    #else:
                    #    print('p hat p %s'%p_hat_p)
                #else:
                #    print('T vu %s'%vu)
            if flag: break
            if len(Q) == 0:
                #print('len QQ %d'%(len(QQ)))
                #print(QQ)
                #Q = QQ[:]
                #QQ = []
                Q = copy.deepcopy(QQ)
                for qqk in QQ:
                    Que[qqk] = QQ[qqk][0]
                QQ = {}
        #if p_best_p == 'none': return 'none2', -1, -1, -1
        return p_best_p, p_max_m, p_best_cost, p_w_p, all_expire, all_counts

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
        print(sums_d)
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
        print('len of edge_desty: %d'%len(edge_desty))
        edges, nodes, G, speed_dict = self.get_graph2(edge_desty, vedge_desty)
        print('len of edge_desty: %d'%len(edge_desty))
        #sys.exit()
        points = self.get_axes()
        #r_pairs = self.get_tpath(points)
        #df2 = open('temp/r_pairs2.txt', 'rb')
        #df2 = open('temp/b3_pairs.txt', 'rb')
        df2 = open('test/new_temp4_.txt', 'rb')
        r_pairs = pickle.load(df2)
        df2.close()
        #df3 = open('temp/gt_30_b3pair_.txt', 'rb')
        df3 = open('temp/odgt_30_b3pair.txt', 'rb')
        gt_data = pickle.load(df3)
        df3.close()
        #sys.exit()
        #r_pairs = self.get_pairs()
        nodes_order, i = {}, 0
        for node in nodes:
            nodes_order[node] = i
            i += 1

        plot_data1 = [0] * 5
        sums = [0] * 5
        one_plot = []
        All_rounds = np.zeros(20).reshape(4, 5)
        One_Plot = np.zeros(20).reshape(4, 5)
        One_Plot2 = np.zeros(20).reshape(4, 5)
        One_Sums = np.zeros(20).reshape(4, 5)
        one_dis = -1
        stores = {}
        for pairs in r_pairs[:]:
            one_dis += 1
            print('one_dis : %d'%one_dis)

            #for pair_ in pairs:
            #    _ = self.get_U(pair_[-1])
            #for pairs in r_pairs:
            tstart = time.time()
            print('len pairs %d'%len(pairs))
            sums2 = 0
            cost_t2 = 0
            for pair_ in pairs[:]:
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
                print(path_)
                bpair = pair_[0].split(';')
                for st2 in path_[1:]:
                #for st2 in bpair:
                    sedge = st1+'-'+st2
                    #sedge = st2
                    print('sedge: %s'%sedge)
                    if sedge in edge_desty:
                        speed_key = list([float(l) for l in edge_desty[sedge].keys()])
                        time_budget += float(np.max(speed_key))
                        #print('speed2')
                        #print(speed_key)

                    elif sedge in speed_dict:
                        speed_key = list(speed_dict[sedge].keys())
                        time_budget += float(np.max(speed_key))
                        #print('speed1')
                        #print(speed_key)
                    else: 
                        print(' edge: %s not in speed_dict, exit'%sedge)
                        sys.exit()
                    st1 = st2 
                print('time budget: %f'%time_budget)
                #budget_ = self.get_T(start, desti, G, points)
                #self.T = self.get_T2(pair_[0])
                for t_b_, t_b in enumerate([0.5, 0.75, 1.0, 1.25, 1.5]):
                    tstart = time.time()
                    self.T = time_budget * t_b * 2
                    U = ''#self.get_U(desti)
                    best_p, max_m, best_c, best_pw, all_expire, all_rounds = self.rout(start, desti, edge_desty, vedge_desty, speed_dict, nodes_order, U, G, points)
                    #stores[st_key][str(t_b)] = [time.time()-tstart, all_rounds]
                    stores[st_key][str(t_b)]=[time.time()-tstart-all_expire,all_rounds]
                    #print('distance %f km'%distan)
                    #print('best path %s'%best_p)
                    #print(len(best_p))
                    if best_p == 'none1': continue
                    if not isinstance(best_pw, dict): continue
                    #sums += 1
                    tend = time.time()
                    plot_data1[t_b_] += tend - tstart
                    sums[t_b_] += 1

                    One_Plot[one_dis][t_b_] += tend - tstart 
                    One_Plot2[one_dis][t_b_] += tend - tstart  - all_expire
                    One_Sums[one_dis][t_b_] += 1
                    All_rounds[one_dis][t_b_] += all_rounds
                    print('cost time : %f'%(tend-tstart))
                    print('cost time 2 : %f'%(tend-tstart-all_expire))

                    if t_b_ == 2:
                        sums2 += 1
                        cost_t2 += tend - tstart
                
                #sys.exit()
            print('sums2: %d'%sums2)
            print('cost_t2: %f'%cost_t2)
            one_plot.append(round(cost_t2/sums2, 4))
        for i in range(5):
            if sums[i] == 0:
                print('zero %d'%i)
                continue
            plot_data1[i] /= sums[i]
        print(plot_data1)
        print(sums)
        print('one plot, routing cost time for distance')
        print(one_plot)
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
        fname = 'meclud_1.json'
        with open(self.subpath + fname, 'w') as fw:
            json.dump(stores, fw, indent=4)

if __name__ == '__main__':

    pairs_name = ['./test/t16A', './test/t16B', './test/t16C', './test/t16D']
    threads_num = 15
    sigma = 90
    subpath = './res3/'
    #fpath = './res3/umatrix2/'
    #fpath = './res3/u_mul_matrix3/'
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
