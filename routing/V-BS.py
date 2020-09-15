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
    def __init__(self, fpath, T, fpath_desty, fvedge_desty, fedge_desty, graph_store_name, degree_file, subpath, axes_file, pairs_name, speed_file, true_path, sigma, eta):
        self.fpath = fpath
        self.hU = {}
        self.T = T
        self.sigma = sigma
        self.eta = eta
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

    def get_vtpath(self, edge_desty, path_desty, vpath):
        vpath = vpath.strip().split(';')
        new_vpath = []
        for vp in vpath:
            if vp in path_desty:
                new_vpath.append(path_desty[vp][0][0])
            else:
                new_vpath.append(vp)
        
        return ';'.join(l for l in new_vpath)

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
        if not os.path.isfile(self.fpath+u_name):
            return {}
        fn = open(self.fpath+u_name)
        U = {}
        for line in fn:
            line = line.strip().split(';')
            U[line[0]] = np.array([float(l) for l in line[1:]])
        self.hU[u_name] = U
        return U

    def get_U2(self, u_name):
        if u_name in self.hU:
            return self.hU[u_name]
        if not os.path.isfile(self.fpath+u_name):
            #print('no this U : %s'%(self.fpath+u_name))
            print(u_name)
            return {}
        fn = open(self.fpath + u_name)
        U = {}
        for line in fn:
            line = line.strip().split(';')
            U[line[0]] = np.zeros(self.eta)
            if line[1] == '-1' and line[2] == '-1':
                pass
            elif line[1] == '-1' and line[2] == '0':
                U[line[0]] = np.ones(self.eta)
            else:
                for i in range(int(line[1])+1, int(line[2])):
                    t = 3 + i - int(line[1]) - 1
                    U[line[0]][i] = float(line[t])
                for i in range(int(line[2]), self.eta):
                    U[line[0]][i] = 1.0
        fn.close()
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
                #speed_dict[line[0]] = {int(float(line[1])/float(line[2])*3600): 1.0}
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
        for edge in edge_desty:
            #edge_ = edge.split('-')
            if edge not in speed_dict:
                print(edge)
        G2 = nx.DiGraph()
        G2.add_nodes_from(all_nodes)
        G2.add_weighted_edges_from(All_edges)

        fn = open(self.subpath+self.graph_store_name)
        for line in fn:
            edge = line.strip()
            if edge not in edge_desty and edge not in vedge_desty:
                if edge in speed_dict:
                    edge_desty[edge] = speed_dict[edge]
                else:
                    continue
                cost = min(float(l) for l in edge_desty[edge].keys())
                edge_ = edge.split('-')
                All_edges.append((edge_[0], edge_[1], cost))
                all_edges.add(edge)
        fn.close()
        
        for edge in vedge_desty:
            if edge not in edge_desty and edge not in speed_dict:
                edge_ = edge.split('-')
                cost1 = vedge_desty[edge].keys()
                cost = min(float(l) for l in cost1)
                All_edges.append((edge_[0], edge_[1], cost))
                all_edges.add(edge)
        
        all_nodes, all_edges = list(all_nodes), list(all_edges)
        G = nx.DiGraph()
        G.add_nodes_from(all_nodes)
        G.add_weighted_edges_from(All_edges)
        return all_edges, all_nodes, G, G2, speed_dict

    def get_graph2(self, edge_desty, vedge_desty, r_pairs, points):
        speed_dict = self.get_speed()
        all_nodes, all_edges = set(), set()
        for key in speed_dict:
            line = key.split('-')
            all_nodes.add(line[0])
            all_nodes.add(line[1])
            all_edges.add(key)
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
        for edge in edge_desty:
            #edge_ = edge.split('-')
            if edge not in speed_dict:
                print(edge)
        G2 = nx.DiGraph()
        G2.add_nodes_from(all_nodes)
        G2.add_weighted_edges_from(All_edges)

        temp_edges, temp_nodes = set(), set()
        for pairs in r_pairs:
            for pair in pairs:
                stedge = pair[-2] + '-' + pair[-1]
                #print(stedge)
                temp_edges.add(stedge)
                temp_nodes.add(pair[-2])
                temp_nodes.add(pair[-1])
        '''
        fn = open(self.subpath+self.graph_store_name)
        l5 = 0
        for line in fn:
            edge = line.strip()
            if edge in temp_edges: 
                l5 += 1
                continue 
            if edge not in edge_desty and edge not in vedge_desty:
                if edge in speed_dict:
                    edge_desty[edge] = speed_dict[edge]
                else:
                    continue
                cost = min(float(l) for l in edge_desty[edge].keys())
                edge_ = edge.split('-')
                distan = self.get_distance(points, (edge_[0], edge_[1]))
                #if distan < 5: continue 
                All_edges.append((edge_[0], edge_[1], cost))
                all_edges.add(edge)
        fn.close()
        print('l5 :%d'%l5)
        '''
        A = [0] * 4
        for edge in vedge_desty:
            if edge == '663487-609899':
                print('shot ')
            if edge not in edge_desty and edge not in speed_dict:
                if edge == '663487-609899':
                    print('shot ')
                if edge in temp_edges: 
                    print('del %s '%edge)
                    continue
                edge_ = edge.split('-')
                #if edge_[0] in temp_nodes or edge_[1] in temp_nodes: continue
                if np.random.rand() > 0.5: continue
                cost1 = vedge_desty[edge].keys()
                cost = min(float(l) for l in cost1)
                distan = self.get_distance(points, (edge_[0], edge_[1]))
                if distan < 5: A[0] += 1
                elif distan < 10: A[1] += 1
                elif distan < 25: A[2] += 1
                else: A[3] += 1
                if distan > 17: continue 
                All_edges.append((edge_[0], edge_[1], cost))
                all_edges.add(edge)
        print('A')
        print(A)
        #sys.exit()
        for edge in temp_edges:
            if edge in vedge_desty:
                del vedge_desty[edge]
        print('len of vedge_desty 2: %d'%len(vedge_desty))
        all_nodes, all_edges = list(all_nodes), list(all_edges)
        G = nx.DiGraph()
        G.add_nodes_from(all_nodes)
        G.add_weighted_edges_from(All_edges)
        return all_edges, all_nodes, G, G2, speed_dict
 
    def get_P(self, ):
        pass

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

    def check_domin(self, w_p_hat_p, QQ):
        #if len(QQ) == 0: return True
        w_len = len(w_p_hat_p)
        A1, A2 = list(w_p_hat_p.keys()), list(w_p_hat_p.values())
        dom = 0
        AA = []
        for (_, _, Q3, _) in QQ:
            q_len = len(Q3)
            M = min(w_len, q_len)
            B1, B2 = list(Q3.keys()), list(Q3.values())
            a, b = 0, 0
            for m in range(M):
                if A1[m] > B1[m]:
                    a += 1
                elif A1[m] < B1[m]:
                    b += 1
                else:
                    if A2[m] > B2[m]:
                        a += 1
                    else:
                        b += 1
            #if a < b:
            if b == M:
                AA.insert(0, dom)
                #return True
            dom += 1
        if len(AA) == 0: return False, QQ
        if len(AA) == len(QQ): return True, []
        for aa in AA: del QQ[aa]
        return False, QQ

    def check_domin3(self, w_p_hat_p, QQ):
        if len(QQ) == 0: return True
        w_len = len(w_p_hat_p)
        A1, A2 = list(w_p_hat_p.keys()), list(w_p_hat_p.values())
        dom = 0
        #print('w p hat p')
        #print(w_p_hat_p)
        for (_, _, Q3, _) in QQ:
            #print('Q3')
            #print(Q3)
            q_len = len(Q3)
            M = min(w_len, q_len)
            B1, B2 = list(Q3.keys()), list(Q3.values())
            a, b = 0, 0
            for m in range(M):
                if float(A1[m]) > float(B1[m]):
                    a += 1
                elif float(A1[m]) < float(B1[m]):
                    b += 1
                else:
                    if float(A2[m]) < float(B2[m]):
                        a += 1
                    else:
                        b += 1
            if a > b: return False
        return True

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
 
    def rout(self, start, desti, edge_desty, vedge_desty, speed_dict, nodes_order, U, G, points):
        path = []
        Q, P_hat = [], []
        neigh = list(G.successors(start))
        print('neigh %d'%len(neigh))
        start_ax = points[start]
        desti_ax = points[desti]
        s_d_arr = [desti_ax[0] - start_ax[0], desti_ax[1] - start_ax[1]]
        all_expire = 0.0 
        def get_weight(p_hat):
            w_p_hat = {}
            if p_hat in edge_desty:
                w_p_hat = edge_desty[p_hat]
            elif p_hat in vedge_desty:
                w_p_hat = vedge_desty[p_hat]
            elif p_hat in speed_dict:
                w_p_hat = speed_dict[p_hat]
            return w_p_hat

        def get_maxP(w_p_, vv):
            p_max = 0.0
            for pk in w_p_:
                w_pk = w_p_[pk]
                pk_ = int(int(pk) / self.sigma)
                if int(pk) % self.sigma == 0: pk_ += 1
                if len(U[vv]) <= pk_: pk_ = len(U[vv]) - 1
                p_max += float(w_pk ) * U[vv][pk_]
            return p_max
        has_visit = set()
        has_visit.add(start)
        Que = PQDict.maxpq()
        Q = {}
        for vi in neigh:
            if vi in has_visit: continue
            else: has_visit.add(vi)
            p_hat = start +'-'+ vi
            w_p_hat = get_weight(p_hat)
            w_min = min([float(l) for l in w_p_hat.keys()])
            p_order = nodes_order[vi] #p_hat]
            tcost1 = time.time()
            inx_min = np.argwhere(U[vi] > 0)
            if len(inx_min) == 0: continue
            inx_min = inx_min[0][0]
            all_expire += time.time() - tcost1
            cost_time = w_min + inx_min*self.sigma
            #print('T %d'%(cost_time))
            if cost_time <= self.T:
                p_max = get_maxP(w_p_hat, vi)   
                Que[p_hat] = p_max
                Q[p_hat] = (p_max, w_p_hat, cost_time)
        print('len Q %d'%len(Q))
        QQ = {}
        p_best_p, flag = 'none', False
        p_max_m, p_best_cost, p_w_p = -1, -1, -1
        all_rounds = 0
        if len(Q) == 0: return 'none1', -1, -1, -1, all_expire, all_rounds 
        while len(Q) != 0:
            (p_hat, pqv) = Que.popitem()
            all_rounds += 1
            (p_max, w_p_hat, cost_time) = Q[p_hat]
            del Q[p_hat]
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
            #print('len neight_2 %d'%len(neigh))
            cost_sv = min([float(l) for l in w_p_hat.keys()])
            vd_d_arr = [points[desti][0]-points[v_l][0], points[desti][1]-points[v_l][1]]
            for u in neigh:
                if u == desti:
                    vu = v_l + '-' + u
                    w_vu = get_weight(vu)
                    if len(w_vu) == 0: cost_vu = 0
                    else: cost_vu = min([float(l) for l in w_vu.keys()])
                    tcost1 = time.time()
                    inx_min = np.argwhere(U[u] > 0)
                    inx_min = inx_min[0][0]
                    p_best_p = p_hat + ';' + vu
                    p_w_p = self.conv(w_p_hat, w_vu)
                    p_max_m = get_maxP(w_p_hat, u)
                    all_expire += time.time() - tcost1
                    p_best_cost = cost_sv + cost_vu + inx_min*self.sigma
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
                tcost1 = time.time()
                inx_min = np.argwhere(U[u] > 0)
                if len(inx_min) == 0 : 
                    #print('inx vu %s'%vu)
                    continue
                inx_min = inx_min[0][0]
                all_expire += time.time() - tcost1
                cost_time = cost_sv + cost_vu + inx_min*self.sigma
                if cost_time <= self.T:
                    p_hat_p = p_hat + ';' + vu
                    w_p_hat_p = self.conv(w_p_hat, w_vu)
                    p_hat_max = get_maxP(w_p_hat, u)
                    QQ[p_hat_p] = (p_hat_max, w_p_hat_p, cost_time)
            if flag: break
            if len(Q) == 0:
                #print('len QQ %d'%(len(QQ)))
                #tcost1 = time.time()
                Q = copy.deepcopy(QQ)
                for qqk in QQ:
                    Que[qqk] = QQ[qqk][0]
                QQ = {}
                #all_expire += time.time() - tcost1

        return p_best_p, p_max_m, p_best_cost, p_w_p, all_expire, all_rounds
        
    def main(self, ):
        #path_desty, vedge_desty, edge_desty = self.get_dict()
        vedge_desty, edge_desty, path_desty = self.get_dict()
        print('len of vedge_desty: %d'%len(vedge_desty))
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
        df4 = open('test/new_temp.txt', 'rb')
        r_pairs2 = pickle.load(df4)
        df4.close()
        #sys.exit()
        #r_pairs = self.get_pairs()
        #edges, nodes, G, G2, speed_dict = self.get_graph(edge_desty,vedge_desty)
        edges, nodes, G, G2, speed_dict = self.get_graph2(edge_desty, vedge_desty,r_pairs, points)
        print('num of edges: %d, num of nodes: %d'%(len(edges), len(nodes)))
        print('len of vedge_desty 2: %d'%len(vedge_desty))

        nodes_order, i = {}, 0
        for node in nodes:
            nodes_order[node] = i
            i += 1
        PT = 5
        #r_pairs = [[('289377', '21851')]]
        plot_data1, plot_kl, plot_lcs, plot_gt_kl= [0.0]*PT, [0.0]*PT, [0.0]*PT, [0.0]*PT
        sums = [0] * PT
        all_kl = []
        all_iters, flag = 0, False
        all_mps = [0.0]*PT
        all_mps2 = [0.0]*PT
        ls = 0
        one_plot = []
        All_rounds = np.zeros(20).reshape(4, 5)
        One_Plot = np.zeros(20).reshape(4, 5)
        One_Plot2 = np.zeros(20).reshape(4, 5)
        One_Sums = np.zeros(20).reshape(4, 5)
        one_dis = -1
        stores = {}
        for pairs in r_pairs:
            one_dis += 1
            #if one_dis != 2: continue
            print('one dis : %d'%one_dis)
            #examp = [('152522','552130'), ('220007', '404561'), ('554534', '529747'), ('165242', '197585'), ('180296', '435501'), ('188521', '350195'), ('321692', '261958'), ('605316', '343667'), ('271336', '547890'), ('304501', '64596'), ('175653', '343667'), ('390147', '569359'), ('421275', '413361'), ('420893', '159085'), ('108596', '232137'), ('78037', '83321'), ('263265', '173706'), ('5920', '18137'), ('72665', '113028'), ('176111', '354903'), ('309021', '354903'), ('322559', '78036')]
            for pair_ in pairs:
            #for pair_ in examp:
                _ = self.get_U2(pair_[-1])
            dij_time = 0.0
            #continue
            #for pairs in r_pairs:
            print('len pairs %d'%len(pairs))
            kl_, lcs_ = 0.0, 0.0
            sums2 = 0
            if flag: break
            cost_t2 = 0
            for pair_ in pairs:
            #for pair_ in examp:
                all_iters += 1
                #if all_iters > 20: 
                #    flag = True
                #    break
                tstart = time.time()
                print(pair_)
                start, desti = pair_[-2], pair_[-1]
                #if (start, desti) in examp: continue
                #pred = self.get_dijkstra3(G2, desti)
                pred = self.get_dijkstra3(G, desti)
                print('len of pred : %d'%len(pred))
                path_, st1 = [start], start
                distan2 = 0
                flg = False
                while st1 != desti:
                    st2 = st1
                    if st1 not in pred:
                        flg = True
                        break
                    st1 = pred[st1]
                    path_.append(st1)
                    distan2 += self.get_distance(points, (st2, st1))
                if flg : 
                    print('isolated node')
                    print(pair_)
                    continue
                distan = self.get_distance(points, (start, desti))
                st_key = start + '+' + desti + ':'+str(distan)+':'+str(distan2)
                stores[st_key] = {}
                st1, time_budget = start, 0.0
                for st2 in path_[1:]:
                    sedge = st1+'-'+st2
                    if sedge in edge_desty:
                        speed_key = list([abs(float(l)) for l in edge_desty[sedge].keys()])
                        #time_budget += max(speed_key)
                        time_budget += (min(speed_key)+ max(speed_key))/2
                    elif sedge in vedge_desty:
                        speed_key = list([abs(float(l)) for l in vedge_desty[sedge].keys()])
                        #time_budget += max(speed_key)
                        time_budget += (min(speed_key) + max(speed_key))/2
                    elif sedge in speed_dict:
                        speed_key = list([abs(float(l)) for l in speed_dict[sedge].keys()])
                        #time_budget += max(speed_key)
                        time_budget += (min(speed_key) + max(speed_key)) / 2
                    else: 
                        print(' edge: %s not in speed_dict, exit'%sedge)
                        sys.exit()
                    st1 = st2 
                print('time budget: %f'%time_budget)
                #self.T = time_budget 
                U = self.get_U2(desti)
                if len(U) == 0: continue
                tend = time.time()
                ls += 1
                all_kl_ = []
                for t_b_, t_b in enumerate([0.5, 0.75, 1.0, 1.25, 1.5]):
                    #if t_b_ != 1: continue
                    tstart = time.time()
                    self.T = time_budget * t_b
                    best_p, max_m, best_c, best_pw, all_expires, all_rounds = self.rout(start, desti, edge_desty, vedge_desty, speed_dict, nodes_order, U, G, points)
                    stores[st_key][str(t_b)] = [time.time()-tstart-all_expires, all_rounds]
                    if best_p == 'none1' or best_p == 'none2' or best_p == 'none':
                        print('fail routing')
                        all_rounds = 0
                        continue
                    tend = time.time()
                    print('ttt %f'%(tend - tstart))
                    plot_data1[t_b_] += tend - tstart
                    sums[t_b_] += 1

                    One_Plot[one_dis][t_b_] += tend - tstart 
                    One_Plot2[one_dis][t_b_] += tend - tstart - all_expires
                    One_Sums[one_dis][t_b_] += 1
                    All_rounds[one_dis][t_b_] += all_rounds

                    if t_b_ == 2:
                        sums2 += 1
                        cost_t2 += tend - tstart 
            print('cost t2 : %f'%cost_t2)
            print('sums2 : %d'%sums2)
            if sums2 < 1: continue
            one_plot.append(round(cost_t2 / sums2, 4))
            print('time cost')
            #sys.exit()
        for i in range(PT):
            if sums[i] == 0:
                print('zero %d'%i)
                continue
            plot_data1[i] /= sums[i]
        print(plot_data1)
        print(sums)
        print('one_plot, cost time for distance')
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
        #sys.exit()
        fname = 'mcrout2_%d_1.json'%self.sigma
        with open(self.subpath + fname, 'w') as fw:
            json.dump(stores, fw, indent=4)


if __name__ == '__main__':

    pairs_name = ['./test/t16A', './test/t16B', './test/t16C', './test/t16D']
    threads_num = 15
    dinx = 30
    #sigma, eta = 10, 800
    sigma, eta = 30, 333
    #sigma, eta = 60, 170
    #sigma, eta = 90, 111
    print('eta: %d, sigma: %d'%(eta, sigma))

    subpath = './res3/'
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
    fname = '337153'
    time_budget = 5000
    rout = Rout(fpath, time_budget, fpath_desty, fvedge_desty, fedge_desty, graph_store_name, degree_file, subpath, axes_file, pairs_name, speed_file, true_path, sigma, eta)
    rout.main()

