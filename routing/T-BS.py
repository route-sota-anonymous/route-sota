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
        self.sigma = sigma
        self.eta = eta
    
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
            print('no this U : %s'%(self.fpath+u_name))
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
            edges_.add(line[0]+'-'+line[1])
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
        return edges, nodes, G, G2, speed_dict
 
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
 
    def rout(self, start, desti, edge_desty, vedge_desty, nodes_order, U, G, G2, points):
        path = []
        Q, P_hat = [], []
        neigh = list(G2.successors(start))
        print('neigh %d'%len(neigh))
        start_ax = points[start]
        desti_ax = points[desti]
        s_d_arr = [desti_ax[0] - start_ax[0], desti_ax[1] - start_ax[1]]
        all_expire = 0.0 
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
        has_visit = set()
        has_visit.add(start)
        Que = PQDict.maxpq()
        Q = {}
        for vi in neigh:
            if vi in has_visit: continue
            else: has_visit.add(vi)
            #vi_d_arr = [points[vi][0]-start_ax[0], points[vi][1]-start_ax[1]]
            #if self.cosin_distance(s_d_arr, vi_d_arr) < 0: continue
            p_hat = start +'-'+ vi
            w_p_hat = get_weight(p_hat)
            w_min = min([float(l) for l in w_p_hat.keys()])
            p_order = nodes_order[vi] #p_hat]
            tcost1 = time.time()
            #inx_min = np.argwhere(U[p_order] > 0)[0][0]
            inx_min = np.argwhere(U[vi] > 0)
            #inx_min = np.argwhere(U[vi] > 0.1)
            if len(inx_min) == 0: 
                print('u 0 vd: %s %s'%(vi, desti))
                continue
            inx_min = inx_min[0][0]
            all_expire += time.time() - tcost1
            cost_time = w_min + inx_min*self.sigma
            #print('w min : %f'%w_min)
            #print('inx sigma : %f'%(inx_min*self.sigma))
            #print('cost time %d'%(cost_time))
            #print('T %d'%(self.T))
            if cost_time <= self.T:
                tcost1 = time.time()
                p_max = get_maxP(w_p_hat, vi)
                all_expire += time.time() - tcost1
                #p.maxin = *U[nodes_order[p_hat]]
                #Q.append((p_hat, p_max, w_p_hat, cost_time))
                Que[p_hat] = p_max
                Q[p_hat] = (p_max, w_p_hat, cost_time)
        print('len Q %d'%len(Q))
        #print(Q)
        #QQ, PP, KK = [], [], []
        QQ = {}
        p_best_p, flag = 'none', False
        p_max_m, p_best_cost, p_w_p = -1, -1, -1
        if len(Q) == 0: return 'none1', -1, -1, -1, all_expire, -1
        all_rounds = 0
        while len(Q) != 0:
            (p_hat, pqv) = Que.popitem()
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
                #PP.append((p_hat, p_max, w_p_hat, cost_time))
                #return p_hat, p_max, w_p_hat, cost_time
                #KK.append((p_best_p, p_max_m, p_best_cost, p_w_p))
                break
            #print('v_l : %s'%v_l)
            neigh = list(G2.successors(v_l))
            #print('len neight_2 %d'%len(neigh))
            #print(neigh)
            #w_sv = get_weight(p_hat)
            #w_sv = w_p_hat
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
                    #inx_min = np.argwhere(U[u] > 0.1)
                    inx_min = inx_min[0][0]
                    p_best_p = p_hat + ';' + vu
                    p_w_p = self.conv(w_p_hat, w_vu)
                    p_max_m = get_maxP(w_p_hat, u)
                    all_expire += time.time() - tcost1
                    p_best_cost = cost_sv + cost_vu + inx_min*self.sigma
                    flag = True
                    break
                    #KK.append((p_best_p, p_max_m, p_best_cost, p_w_p))
                    #continue
                #print('u : %s'%u)
                if u in has_visit: 
                    print('u1 %s, vd %s'%(u, desti))
                    continue
                else: has_visit.add(u)
                #vi_d_arr = [points[u][0]-points[v_l][0], points[u][1]-points[v_l][1]]
                #if self.cosin_distance(vd_d_arr, vi_d_arr) < 0: continue
                if u in p_hat: 
                    print('u2 %s, vd %s'%(u, desti))
                    continue
                vu = v_l + '-' + u
                w_vu = get_weight(vu)
                if len(w_vu) == 0:
                    print('vu %s, vd %s'%(vu, desti))
                    continue
                cost_vu = min([float(l) for l in w_vu.keys()])
                p_order = nodes_order[u] #p_hat]
                tcost1 = time.time()
                #inx_min = np.argwhere(U[p_order] > 0)[0][0]
                inx_min = np.argwhere(U[u] > 0)
                #inx_min = np.argwhere(U[u] > 0.1)
                if len(inx_min) == 0 : 
                    print('inx vu %s, vd %s'%(vu, desti))
                    continue
                inx_min = inx_min[0][0]
                all_expire += time.time() - tcost1
                cost_time = cost_sv + cost_vu + inx_min*self.sigma
                #print('vi min : %f' % (inx_min*self.sigma))
                #print('cost time 3 : %f'%(cost_time))
                if cost_time <= self.T:
                    p_hat_p = p_hat + ';' + vu
                    w_p_hat_p = self.conv(w_p_hat, w_vu)
                    tcost1 = time.time()
                    p_hat_max = get_maxP(w_p_hat, u)
                    all_expire += time.time() - tcost1
                    #if self.check_domin3(w_p_hat_p, QQ):
                    #QQ.append((p_hat_p, p_hat_max, w_p_hat_p, cost_time))
                    QQ[p_hat_p] = (p_hat_max, w_p_hat_p, cost_time)
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
        '''
        if len(PP) == 0: return KK#'none3', -1, -1
        PPP = self.check_domin2(PP)
        for l, ppp in enumerate(PP):
            if PPP[l] == 0: KK.append(ppp)
        return KK #p_best_p, p_max_m, p_best_cost
        '''
        return p_best_p, p_max_m, p_best_cost, p_w_p, all_expire, all_rounds
        #return KK
        
    def main(self, ):
        #path_desty, vedge_desty, edge_desty = self.get_dict()
        vedge_desty, edge_desty, path_desty = self.get_dict()
        edges, nodes, G, G2, speed_dict = self.get_graph(edge_desty, vedge_desty)
        #sys.exit()
        points = self.get_axes()
        #r_pairs = self.get_tpath(points)
        #df2 = open('temp/r_pairs2.txt', 'rb')
        #df2 = open('temp/b3_pairs.txt', 'rb')
        df2 = open('test/new_temp4_.txt', 'rb')
        r_pairs = pickle.load(df2)
        df2.close()
        df3 = open('temp/gt_30.txt', 'rb')
        gt_kl = pickle.load(df3)
        df3.close()
        #sys.exit()
        #r_pairs = self.get_pairs()
        nodes_order, i = {}, 0
        for node in nodes:
            nodes_order[node] = i
            i += 1
        PT = 5
        #r_pairs = [[('289377', '21851')]]
        plot_data1, plot_data2, plot_kl, plot_lcs = [0.0]*PT, [0.0]*PT, [0.0]*PT, [0.0]*PT
        sums = [0] * PT
        all_kl = []
        all_iters, flag = 0, False
        one_plot, one_plot2 = [], []
        All_rounds = np.zeros(20).reshape(4, 5)
        One_Plot = np.zeros(20).reshape(4, 5)
        One_Plot2 = np.zeros(20).reshape(4, 5)
        One_Sums = np.zeros(20).reshape(4, 5)
        one_dis = -1
        stores = {}
        for pairs in r_pairs:
            one_dis += 1
            print('one_dis : %d'%one_dis)
            examp = [['247680', '332558']]
            examp = [['393127', '166298']]
            examp = [['512700', '451997']]
            for pair_ in pairs[:]:
            #for pair_ in examp:
                _ = self.get_U2(pair_[-1])
            dij_time = 0.0
            #for pairs in r_pairs:
            print('len pairs %d'%len(pairs))
            kl_, lcs_ = 0.0, 0.0
            sums2, atimes = 0, 0
            if flag: break
            cost_t2, cost_t3 = 0, 0
            for pair_ in pairs[:]:
            #for pair_ in examp:
                all_iters += 1
                #if all_iters > 20: 
                #    flag = True
                #    break
                tstart = time.time()
                print(pair_)
                start, desti = pair_[-2], pair_[-1]
                pred = self.get_dijkstra3(G2, desti)
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
                    #    speed_key = list(vedge_desty[sedge].keys#())
                    #    time_budget += float(speed_key[0])
                    else: 
                        print(' edge: %s not in speed_dict, exit'%sedge)
                        sys.exit()
                    st1 = st2 
                print('time budget: %f'%time_budget)
                #self.T = time_budget 
                #U = self.get_U(desti)
                U = self.get_U2(desti)
                if len(U) == 0 : continue
                #t_b_t, t_b_kl, t_b_lcs = [], [], []
                tend = time.time()
                #plot_data1 = [tend-tstart+l for l in plot_data1]
                #sums += 1
                all_kl_ = []
                for t_b_, t_b in enumerate([0.5, 0.75, 1.0, 1.25, 1.5]):
                    #if t_b_ != 2: continue
                    tstart = time.time()
                    self.T = time_budget * t_b
                    best_p, max_m, best_c, best_pw, all_expires, all_rounds = self.rout(start, desti, edge_desty, vedge_desty, nodes_order, U, G, G2, points)
                    stores[st_key][str(t_b)] = [time.time()-tstart, all_rounds]
                    if best_p == 'none1' or best_p == 'none2' or best_p == 'none':
                        print('fail routing')
                        continue
 
                    tend = time.time()
                    plot_data1[t_b_] += tend - tstart
                    plot_data2[t_b_] += tend - tstart - all_expires
                    sums[t_b_] += 1

                    One_Plot[one_dis][t_b_] += tend - tstart 
                    One_Plot2[one_dis][t_b_] += tend - tstart - all_expires
                    One_Sums[one_dis][t_b_] += 1
                    All_rounds[one_dis][t_b_] += all_rounds
                    #print('cost time : %f'%(tend - tstart))
                    #print('cost time 2 : %f'%(tend - tstart - all_expires))
                    #print('all rounds : %d'%all_rounds)
                    if t_b_ == 2:
                        sums2 += 1
                        cost_t2 += tend - tstart
                        cost_t3 += tend - tstart - all_expires
            #sys.exit()
            print('time cost 2 : %f'%cost_t2)
            print('time cost 3 : %f'%cost_t3)
            print('sums2: %d'%sums2)
            one_plot.append(round(cost_t2 / sums2, 4))
            one_plot2.append(round(cost_t3 / sums2, 4))
            
        for i in range(PT):
            if sums[i] == 0:
                print('zero %d'%i)
                continue
            plot_data1[i] /= sums[i]
            plot_data2[i] /= sums[i]
        print(plot_data1)
        print(plot_data2)
        print(sums)
        print('one plot, routing cost time for distance')
        print(one_plot)
        print(one_plot2)
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
        fname = 'ncrout_%d_8.json'%self.sigma
        with open(self.subpath + fname, 'w') as fw:
            json.dump(stores, fw, indent=4)

if __name__ == '__main__':

    pairs_name = ['./test/t16A', './test/t16B', './test/t16C', './test/t16D']
    threads_num = 15
    dinx = 30
    #sigma, eta = 10, 800
    #sigma, eta = 30, 333
    #sigma, eta = 60, 170
    sigma, eta = 90, 111
    print('eta: %d, sigma: %d'%(eta, sigma))
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
    fname = '337153'
    time_budget = 5000
    rout = Rout(fpath, time_budget, fpath_desty, fvedge_desty, fedge_desty, graph_store_name, degree_file, subpath, axes_file, pairs_name, speed_file, true_path, sigma, eta)
    rout.main()

