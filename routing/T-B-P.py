import networkx as nx
import numpy as np
import os, sys
import json
import time
import operator
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from scipy.stats import entropy
from pqdict import PQDict
import pickle
import copy

from v_opt import VOpt

class Modified():

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
    
    def add_tpath(self,):
        with open(self.subpath + self.true_path) as fn:
            gt_path_ = json.load(fn)
        gt_path = {}
        for gt in gt_path_:
            a = gt.find('-')
            a1 = gt[:a]
            b = gt.rfind('-')
            b1 = gt[b+1:]
            gt_key = a1+'-'+b1
            nw_key = list(gt_path_[gt].keys())
            nw_value = list(gt_path_[gt].values())
            lens1 = len(gt.split(';'))
            if gt_key not in gt_path:
                gt_path[gt_key] = [str(gt), gt_path_[gt], lens1, nw_key[0], a1, b1]
            else:
                ex_path = gt_path[gt_key]
                #lens2 = len(ex_path[0].split(';'))
                mins  = min(len(ex_path[1]), len(gt_path_[gt]))
                ex_key = list(ex_path.keys())
                ex_value = list(ex_path.values())
                a, b = 0, 0
                for i in range(mins):
                    if float(ex_key[i]) < float(nw_key[i]):
                        a += 1
                    elif  float(ex_key[i]) > float(nw_key[i]): 
                        b += 1
                    else:
                        if float(ex_value[i]) > float(nw_value[i]):
                            a += 1
                        else:
                            b += 1
                if lens1 < gt_path[gt_key][2] : b += 1
                elif lens1 > gt_path[gt_key][2] : a += 1
                if a < b: gt_path[gt_key] = [str(gt), gt_path_[gt], lens1, nw_key[0], a1, b1]

        return gt_path,gt_path_


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

    def get_graph(self, edge_desty, gt_path, vedge_desty):
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

        for gt_ in gt_path:
            #edges.append((gt_path[gt_][-2], gt_path[gt_][-1], abs(float(gt_path[gt_][3]))))
            All_edges.append((gt_path[gt_][-2], gt_path[gt_][-1], abs(float(gt_path[gt_][3]))))
            temp_edge = gt_path[gt_][-2] + '-' + gt_path[gt_][-1]
            if temp_edge not in edge_desty:
                edge_desty[temp_edge] = {abs(float(gt_path[gt_][3])):1.0}
        G = nx.DiGraph()
        G.add_nodes_from(all_nodes)
        G.add_weighted_edges_from(All_edges)
        for edge in speed_dict:
            if edge not in edge_desty:
                edge_desty[edge] = speed_dict[edge]
        return all_edges, all_nodes, G, G2, speed_dict
 
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
       

    def get_modified_one_to_all3(self, G,  target):
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


    def get_modified_one_to_all(self, G, gt_path, edge_desty, target):
        # c1 < uc1 and c2 > uc2 is the ideal state
        def check_dominance(c1, c2, uc1, uc2):
            if c1 < uc1 and c2 < uc2 or c1 > uc1 and c2 > uc2: #non-dominant
                return 0
            elif c1 < uc1 and c2 >= uc2 or c1 <= uc1 and c2 > uc2: #positive dominant
                return 1
            elif c1 > uc1 and c2 <= uc2 or c1 >= uc1 and c2 < uc2: #negative dominant
                return -1

        def check_dominance2(c1, c2, uc1, uc2):
            if c1 < uc1 and c2 < uc2 : #non-dominant
                return 0
            elif c1 < uc1 and c2 >= uc2 or c1 <= uc1 and c2 > uc2: #positive dominant
                return 1
            #elif c1 > uc1 and c2 <= uc2 or c1 >= uc1 and c2 < uc2: #negative dominant
            else:
                return -1

        Qk = [target]
        attr = {target: [0, 0, -1]} # c1, c2, parent
        def trace_path(v1, v2, v_m = None):
            k = 0
            if v1 == v2: return v1
            paths = []
            if v_m is None:
                v_k = v1
                paths = [v1]
            else:
                v_k = v_m
                paths = [v1, v_k]
            while attr[v_k][2] != -1:
                v_k = attr[v_k][2]
                paths.append(v_k)
                k += 1
                #if k > 100:
                #    print(paths)
                #    print('paths')
                #    break
            return '-'.join(v_ for v_ in paths)
        edges = {}
        l = 0
        print(len(list(G.nodes)))

        while len(Qk) != 0:
            #print('len Qk %d'%len(Qk))
            v = Qk.pop()
            #T.append(v)
            #print('v %s' %v)
            #print('len Qk %d'%len(Qk))
            #if v == source: break
            #print('l %d' % l)
            neigh_ = list(G.predecessors(v))#G.successors(v))
            l += 1
            #print('len neigh %d'%len(neigh_))
            #print(neigh_)
            for u in neigh_:
                #print(u)
                if u == target: continue
                if u not in attr:
                    attr[u] = [np.inf, 0, -1]
                #elif attr[u][2] != -1: continue
                key_ = u + '-'+ v 
                if key_ not in edges:
                    edges[key_] = 0
                else: 
                    #print(key_)
                    edges[key_] += 1
                    if edges[key_] > 3:
                        continue
                w = 0.0
                if key_ in edge_desty:
                    w = float(list(edge_desty[key_].keys())[0])
                elif key_ in gt_path:
                    w = float(gt_path[key_][3])
                else:
                    print('hehe')
                c1 = attr[v][0] + w #edge_desty[]
                if key_ in gt_path:
                    a = gt_path[key_][2]
                else:
                    a = 0
                c2 = attr[u][1] + a
                res = check_dominance2(c1, c2, attr[u][0], attr[u][1])
                if res == 0:
                    #print('res 0')
                    p_old = trace_path(u, target, v_m = v)
                    p_new = u+'-'+trace_path(v, target)
                    if p_old != p_new and c1 < attr[u][0]:
                        attr[u][0], attr[u][1] = c1, c2
                        attr[u][2] = v
                        Qk.append(u)
                    if p_old == p_new and c2 > attr[u][1]:
                    #if p_old == p_new and c1 < attr[u][0]:
                        attr[u][0], attr[u][1] = c1, c2
                        attr[u][2] = v
                        Qk.append(u)
                elif res == 1:
                    #print('res 1')
                    attr[u][0], attr[u][1] = c1, c2
                    attr[u][2] = v
                    Qk.append(u)
                else:
                    #print('res 2')
                    pass
        #return T, attr
        #if v != source: print("error happend")
        #print(attr[v][0])
        #return attr[source][0]
        return attr

    def get_gtedge(self, rpath, edge_desty, gt_path):
        rpath = rpath.split(';')
        all_ = []
        for edge in rpath:
            if edge in gt_path:
                tedge = gt_path[edge][0]
                all_.append(tedge)
            else:
                all_.append(edge)
        return ';'.join(l for l in all_)

    def rout(self, start, desti, edge_desty, vedge_desty, speed_dict, nodes_order, U, G, G2, points, gt_path, gt_path_, pred):
        Q, P_hat = [], []
        #neigh = list(G.successors(start))
        neigh = list(G2.successors(start))
        print('neigh %d'%len(neigh))
        start_ax = points[start]
        desti_ax = points[desti]
        s_d_arr = [desti_ax[0] - start_ax[0], desti_ax[1] - start_ax[1]]
        all_expire = 0.0
        #start1 = time.time() 
        #pred = self.get_modified_one_to_all3(G, desti)
        #pred = self.get_modified_one_to_all(G, desti)
        #pred = self.get_modified_one_to_all(G, gt_path, edge_desty, desti)
        #all_expire += time.time() - start1
        def get_weight(p_hat):
            w_p_hat = {}
            if p_hat in edge_desty:
                w_p_hat = edge_desty[p_hat]
            elif p_hat in gt_path:
                w_p_hat = gt_path[p_hat][1]
            elif p_hat in vedge_desty:
                w_p_hat = vedge_desty[p_hat]
                #print('vedge %s' %p_hat)
            elif p_hat in speed_dict:
                w_p_hat = speed_dict[p_hat]
            #else:
            #    #print('other %s' %p_hat)
            if len(w_p_hat) == 0:
                print('zero %s' %p_hat)
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
        def get_vigetmin2(vv):
            #path_ = self.get_dijkstra3(G, vv, desti)
            start1 = time.time()
            v = vv 
            path_ = [v]
            while v != desti:
                v = pred[v]#[2]
                path_.append(v)
            #path_.reverse()
            paths_ = []
            spath = path_[0]
            vi_getmin = 0.0
            expire_time = time.time()-start1
            for epath in path_[1:]:
                key = spath+'-'+epath
                if key in edge_desty:
                    vi_getmin += min(abs(float(l)) for l in edge_desty[key].keys())
                elif key in speed_dict:
                    vi_getmin += min(abs(float(l)) for l in speed_dict[key].keys())
                elif key in gt_path:
                    vi_getmin += min(abs(float(l)) for l in gt_path[p_hat][1].keys())
                else:
                    print('edge not in here')
                spath = epath
            return vi_getmin, expire_time

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
            vi_getmin, ex_p = get_vigetmin2(vi)
            all_expire += ex_p
            cost_time = w_min + vi_getmin
            if start == '148517' or start == '433490':
                print(cost_time)
                print(p_order)
                print(w_p_hat)
                print(p_hat)
                print(vi_getmin)
            #print('vi get min: %f'%vi_getmin)
            #print('cost_time: %f'%cost_time)
            if cost_time <= self.T:
                #p_max = get_maxP3(w_p_hat, vi, vi_getmin)
                p_max = max(list(w_p_hat.values()))
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
                break
            #print('v l %s'%v_l)
            #neigh = list(G.successors(v_l))
            neigh = list(G2.successors(v_l))
            if len(w_p_hat.keys()) == 0:
                print(w_p_hat)
                print(p_hat)
            cost_sv = min([float(l) for l in w_p_hat.keys()])
            vd_d_arr = [points[desti][0]-points[v_l][0], points[desti][1]-points[v_l][1]]
            for u in neigh:
                if u == desti:
                    vu = v_l + '-' + u
                    w_vu = get_weight(vu)
                    if len(w_vu) == 0: cost_vu = 0
                    else: cost_vu = min([float(l) for l in w_vu.keys()])
                    vi_getmin, ex_p = get_vigetmin2(u)
                    all_expire += ex_p
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
                vi_getmin, ex_p = get_vigetmin2(u)#self.get_distance(points, (u, desti)) / self.speed * 3.6
                all_expire += ex_p
                cost_time = cost_sv + cost_vu + vi_getmin#inx_min*self.sigma
                #print('vi get min: %f'%vi_getmin)
                #print('cost time : %f'%cost_time)
                if cost_time <= self.T:
                    p_hat_p = p_hat + ';' + vu
                    w_p_hat_p = self.conv(w_p_hat, w_vu)
                    #p_hat_max = get_maxP3(w_p_hat, u, vi_getmin)
                    p_hat_max = max(list(w_p_hat_p.values()))
                    #QQ.append((p_hat_p, p_hat_max, w_p_hat_p, cost_time))
                    QQ[p_hat_p] = (p_hat_max, w_p_hat_p, cost_time)
            if flag: break
            if len(Q) == 0:
                stemp = time.time()
                #print('len QQ %d'%(len(QQ)))
                #Q = QQ[:]
                #QQ = []
                Q = copy.deepcopy(QQ)
                for qqk in QQ:
                    Que[qqk] = QQ[qqk][0]
                QQ = {}
                #all_expire += time.time() - stemp
        return p_best_p, p_max_m, p_best_cost, p_w_p, all_expire, all_rounds

    def get_T(self, start, desti, G, points):
        shortest_path = nx.shortest_path(G, start, desti)
        print(shortest_path)
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
        gt_path, gt_path_ = self.add_tpath()
        print('len of edge_desty: %d'%len(edge_desty))
        edges, nodes, G, G2, speed_dict = self.get_graph(edge_desty, gt_path, vedge_desty)
        print('len of edge_desty2: %d'%len(edge_desty))
        #source = '585408'
        #target = '156809'
        #path = self.get_dijkstra(G, source, target)
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
        print(len(r_pairs))
        for r_pair in r_pairs:
            print(len(r_pair))
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
            print('one_dis : %d'%one_dis)

            #for pair_ in pairs:
            #    _ = self.get_U(pair_[-1])
            #for pairs in r_pairs:
            tstart = time.time()
            print('len pairs %d'%len(pairs))
            kl_, lcs_ = 0.0, 0.0
            gt_kl_, gt_lcs_ = 0.0, 0.0
            sums2  = 0
            all_expires = 0.0
            mps, mps2 = 0.0, 0.0
            cost_t1, cost_t2 = 0, 0
            for pair_ in pairs:
            #examp = [['77774' , '77773'], ['393129', '393121'], ['233966', '115217'], ['188605', '298874'], ['336662', '71213'], ['302115', '485873'], ['610692', '60727'], ['337153', '261429'], ['475782', '225428']]
            #examp = [['475782', '225428']]
            #examp = [['188605', '298874']]
            #examp = [['247680', '332558']]
            #examp = [['393127', '166298']]
            #examp = [['512700', '451997']]
            #for pair_ in examp:
                print(pair_)
                start, desti = pair_[-2], pair_[-1]
                pred2 = self.get_modified_one_to_all(G, gt_path, edge_desty, desti)
                #pred2 = self.get_modified_one_to_all3(G, gt_path, edge_desty, desti)
                path_2 , st1 = [start], start
                while st1 != desti:
                    st2 = st1 
                    st1 = pred2[st1][2]
                    path_2.append(st1)
                at = 0
                for st2 in path_2[1:]:
                    sedge = st1 + '-' + st2
                    if sedge in gt_path:
                        at += 1
                
                #pred = self.get_dijkstra3(G2, desti)
                pred = self.get_dijkstra3(G, desti)
                path_, st1 = [start], start
                distan2 = 0
                while st1 != desti:
                    st2 = st1
                    st1 = pred[st1]
                    path_.append(st1)
                    distan2 += self.get_distance(points, (st2, st1))
                distan = self.get_distance(points, (start, desti))
                st_key = start + '+' + desti + ':'+str(distan)+':'+str(distan2)+':'+str(len(pred2))+':'+str(at)
                stores[st_key] = {}
                st1, time_budget = start, 0.0
                print(path_)
                for st2 in path_[1:]:
                    sedge = st1+'-'+st2
                    if sedge in edge_desty:
                        speed_key = list([float(l) for l in edge_desty[sedge].keys()])
                        time_budget += max(speed_key)
                    elif sedge in speed_dict:
                        #speed_key = list(speed_dict[sedge].keys())
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
                #self.T = self.T * 1.5

                for t_b_, t_b in enumerate([0.5, 0.75, 1.0, 1.25, 1.5]):
                    self.T = time_budget * t_b
                    tstart = time.time()
                    U = ''#self.get_U(desti)
                    best_p, max_m, best_c, best_pw, all_expire, all_rounds = self.rout(start, desti, edge_desty, vedge_desty, speed_dict, nodes_order,U,G, G2, points, gt_path, gt_path_, pred2)
                    #stores[st_key][str(t_b)] = [time.time()-tstart, all_rounds]
                    stores[st_key][str(t_b)] = [time.time()-tstart-all_expire, all_rounds]
                    all_expires += all_expire
                    #print('distance %f km'%(self.get_distance(points, (start, desti))))
                    #print('best path %s'%best_p)
                    #print(len(best_p))
                    #print(pair_[1])
                    if best_p == 'none1': continue
                    if not isinstance(best_pw, dict): continue
                    tend = time.time()
                    sums[t_b_] += 1
                    plot_data1[t_b_] += tend - tstart
                    plot_data2[t_b_] += tend - tstart - all_expire

                    One_Plot[one_dis][t_b_] += tend - tstart 
                    One_Plot2[one_dis][t_b_] += tend - tstart  - all_expire
                    One_Sums[one_dis][t_b_] += 1
                    All_rounds[one_dis][t_b_] += all_rounds
                    print('cost time : %f'%(tend - tstart))
                    print('cost time 2 : %f'%(tend - tstart - all_expire))
                    print('all rounds: %d'%all_rounds)
                    if t_b_ == 2:
                        sums2 += 1
                        cost_t1 += tend - tstart
                        cost_t2 += tend - tstart - all_expire
            #sys.exit()
            print('cost t1: %f, cost t2: %f'%(cost_t1, cost_t2))
            print('sums2: %d'%sums2)
            one_plot1.append(round(cost_t1/sums2, 4))
            one_plot2.append(round(cost_t2/sums2, 4))
        for i in range(5):
            if sums[i] == 0:
                print('zero %d'%i)
                continue
            plot_data1[i] /= sums[i]
            plot_data2[i] /= sums[i]
        print(plot_data1)
        print(plot_data2)
        print(sums2)
        print('one plot, routing cost time for distance')
        print(one_plot1)
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
        fname = 'mmod_2.json'
        with open(self.subpath + fname, 'w') as fw:
            json.dump(stores, fw, indent=4)


if __name__ == '__main__':

    pairs_name = ['./test/t16A', './test/t16B', './test/t16C', './test/t16D']
    threads_num = 15
    sigma = 30
    subpath = './res3/'
    fpath = './res3/u_mul_matrix3/'
    fpath = './res3/u_mul_matrix_sig%d/'%sigma
    true_path = 'new_path_desty2.json'
    #true_path_dest = 'path_desty2.json'
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
    rout = Modified(fpath, time_budget, fpath_desty, fvedge_desty, fedge_desty, graph_store_name, degree_file, subpath, axes_file, pairs_name, speed_file, true_path)
    rout.main()

