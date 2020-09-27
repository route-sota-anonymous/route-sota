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

    def __init__(self, policy_path, true_path, virtual_path, edge_desty, subpath, axes_file, speed_file, query_name):
        self.sigma = 30
        self.vopt = VOpt()
        self.B = 3
        self.policy_path = policy_path
        self.true_path = true_path
        self.virtual_path = virtual_path
        self.edge_desty = edge_desty
        self.subpath = subpath
        self.axes_file = axes_file
        self.speed_file = speed_file
        self.speed = 50
        self.query_name = query_name
 
   
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


    def get_dict(self, ):
        path_desty = {}
        for l in range(1, 5):
            with open(self.subpath+self.virtual_path+'_%d.json'%l) as js_file:
                path_desty_ = json.load(js_file)
            path_desty.update(path_desty_)

        with open(self.subpath+self.edge_desty) as js_file:
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
            All_edges.append((gt_path[gt_][-1], gt_path[gt_][-2], abs(float(gt_path[gt_][3]))))
            temp_edge = gt_path[gt_][-2] + '-' + gt_path[gt_][-1]
            temp_edge2 = gt_path[gt_][-1] + '-' + gt_path[gt_][-2]
            if temp_edge not in edge_desty:
                edge_desty[temp_edge] = {abs(float(gt_path[gt_][3])):1.0}
            if temp_edge2 not in edge_desty:
                edge_desty[temp_edge2] = {abs(float(gt_path[gt_][3])):1.0}

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
        
        return P


    def rout(self, start, desti, edge_desty, vedge_desty, speed_dict, nodes_order, U, G, G2, points, gt_path, gt_path_, pred):
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
            elif p_hat in gt_path:
                w_p_hat = gt_path[p_hat][1]
            elif p_hat in vedge_desty:
                w_p_hat = vedge_desty[p_hat]
                #print('vedge %s' %p_hat)
            elif p_hat in speed_dict:
                w_p_hat = speed_dict[p_hat]
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
            start1 = time.time()
            v = vv 
            path_ = [v]
            while v != desti:
                v = pred[v]#[2]
                path_.append(v)
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
            if cost_time <= self.T:
                p_max = max(list(w_p_hat.values()))
                Que[p_hat] = p_max
                Q[p_hat] = (p_max, w_p_hat, cost_time)
        print('len Q %d'%len(Q))
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
            a = p_hat.rfind('-')
            v_l = p_hat[a+1:]
            if v_l == desti:
                p_best_p = p_hat
                p_max_m = p_max
                p_best_cost = cost_time
                p_w_p = w_p_hat
                flag = True
                break
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
                vi_getmin, ex_p = get_vigetmin2(u)
                all_expire += ex_p
                cost_time = cost_sv + cost_vu + vi_getmin#inx_min*self.sigma
                if cost_time <= self.T:
                    p_hat_p = p_hat + ';' + vu
                    w_p_hat_p = self.conv(w_p_hat, w_vu)
                    p_hat_max = max(list(w_p_hat_p.values()))
                    QQ[p_hat_p] = (p_hat_max, w_p_hat_p, cost_time)
            if flag: break
            if len(Q) == 0:
                Q = copy.deepcopy(QQ)
                for qqk in QQ:
                    Que[qqk] = QQ[qqk][0]
                QQ = {}
        return p_best_p, p_max_m, p_best_cost, p_w_p, all_expire, all_rounds


    def main(self, ):
        vedge_desty, edge_desty, path_desty = self.get_dict()
        gt_path, gt_path_ = self.add_tpath()
        #print('len of edge_desty: %d'%len(edge_desty))
        edges, nodes, G, G2, speed_dict = self.get_graph(edge_desty, gt_path, vedge_desty)
        #print('len of edge_desty2: %d'%len(edge_desty))
        points = self.get_axes()
        df2 = open(self.query_name, 'rb')
        r_pairs = pickle.load(df2)
        df2.close()
        nodes_order, i = {}, 0
        for node in nodes:
            nodes_order[node] = i
            i += 1
        One_Plot2 = np.zeros(20).reshape(4, 5)
        One_Sums = np.zeros(20).reshape(4, 5)
        one_dis = -1
        cate = ['0-5km', '5-10km', '10-25km', '25-35km']
        for pairs in r_pairs:
            one_dis += 1
            #print('one_dis : %d'%one_dis)
            print('distance category %s'%cate[one_dis])

            tstart = time.time()
            #print('len pairs %d'%len(pairs))
            all_expires = 0.0
            for pair_ in pairs:
                print('o-d pair: %s'%pair_[0]+'-'+pair_[1])
                start, desti = pair_[-2], pair_[-1]
                pred2 = self.get_modified_one_to_all3(G, desti)
                path_2 , st1 = [start], start
                while st1 != desti:
                    st2 = st1 
                    st1 = pred2[st1]#[2]
                    path_2.append(st1)
                at = 0
                for st2 in path_2[1:]:
                    sedge = st1 + '-' + st2
                    if sedge in gt_path:
                        at += 1
                
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
                st1, time_budget = start, 0.0
                print(path_)
                for st2 in path_[1:]:
                    sedge = st1+'-'+st2
                    if sedge in edge_desty:
                        speed_key = list([float(l) for l in edge_desty[sedge].keys()])
                        time_budget += max(speed_key)
                    elif sedge in speed_dict:
                        speed_key = list([float(l) for l in speed_dict[sedge].keys()])
                        time_budget += max(speed_key)
                    else: 
                        print(' edge: %s not in speed_dict, exit'%sedge)
                        sys.exit()
                    st1 = st2 
                print('time budget: %f'%time_budget)

                for t_b_, t_b in enumerate([0.5, 0.75, 1.0, 1.25, 1.5]):
                    self.T = time_budget * t_b
                    tstart = time.time()
                    best_p, max_m, best_c, best_pw, all_expire, all_rounds = self.rout(start, desti, edge_desty, vedge_desty, speed_dict, nodes_order, '', G, G2, points, gt_path, gt_path_, pred2)
                    all_expires += all_expire
                    if best_p == 'none1': continue
                    if not isinstance(best_pw, dict): continue
                    tend = time.time()

                    One_Plot2[one_dis][t_b_] += tend - tstart  - all_expire
                    One_Sums[one_dis][t_b_] += 1

        One_Plot2 = One_Plot2 / One_Sums 
        One_Plot2 = np.nan_to_num(One_Plot2)

        print('The success account')
        print(One_Sums)
        print('The time cost for routing')
        print(One_Plot2)
        print('Time cost for budget: 50%, 75%, 100%, 125%, 150%')
        print(One_Plot2.mean(0))
        print('Time cost for distance: 0-5km, 5-10km, 10-25km, 25-35km')
        print(One_Plot2.mean(1))

if __name__ == '__main__':

    threads_num = 15
    sigma = 30
    subpath = '../data/'
    policy_path = subpath + 'matrix_u/%d/'%sigma
    true_path = 'T-path_desty.json'
    virtual_path = 'V-path_desty'
    edge_desty = 'edge_desty.json'
    axes_file =  '../data/vertices.txt'
    speed_file = '../data/map_ngr'
    query_name = subpath + 'odpairs.txt'
    rout = Modified(policy_path, true_path, virtual_path, edge_desty, subpath, axes_file, speed_file, query_name)
    rout.main()

