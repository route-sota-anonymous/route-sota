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
import argparse

from v_opt import VOpt

class Rout():
    def __init__(self, policy_path, true_path, virtual_path, edge_desty, subpath, axes_file, speed_file, query_name, sigma, eta):
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
        self.sigma = sigma
        self.eta = eta
        self.hU = {}

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


    def get_U2(self, u_name):
        if u_name in self.hU:
            return self.hU[u_name]
        if not os.path.isfile(self.policy_path+u_name):
            print('no this U : %s'%(self.fpath+u_name))
            return {}
        fn = open(self.policy_path + u_name)
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
        return all_edges, all_nodes, G2, speed_dict
 
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

 
    def rout(self, start, desti, edge_desty, vedge_desty, nodes_order, U,  G2, points, speed_dict):
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
            p_hat = start +'-'+ vi
            w_p_hat = get_weight(p_hat)
            w_min = min([float(l) for l in w_p_hat.keys()])
            p_order = nodes_order[vi] #p_hat]
            tcost1 = time.time()
            inx_min = np.argwhere(U[vi] > 0)
            if len(inx_min) == 0: 
                #print('u 0 vd: %s %s'%(vi, desti))
                continue
            inx_min = inx_min[0][0]
            all_expire += time.time() - tcost1
            cost_time = w_min + inx_min*self.sigma
            if cost_time <= self.T:
                tcost1 = time.time()
                p_max = get_maxP(w_p_hat, vi)
                all_expire += time.time() - tcost1
                Que[p_hat] = p_max
                Q[p_hat] = (p_max, w_p_hat, cost_time)
        #print('len Q %d'%len(Q))
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
                    #print('u1 %s, vd %s'%(u, desti))
                    continue
                else: has_visit.add(u)
                if u in p_hat: 
                    #print('u2 %s, vd %s'%(u, desti))
                    continue
                vu = v_l + '-' + u
                w_vu = get_weight(vu)
                if len(w_vu) == 0:
                    #print('vu %s, vd %s'%(vu, desti))
                    continue
                cost_vu = min([float(l) for l in w_vu.keys()])
                p_order = nodes_order[u] #p_hat]
                tcost1 = time.time()
                inx_min = np.argwhere(U[u] > 0)
                if len(inx_min) == 0 : 
                    #print('inx vu %s, vd %s'%(vu, desti))
                    continue
                inx_min = inx_min[0][0]
                all_expire += time.time() - tcost1
                cost_time = cost_sv + cost_vu + inx_min*self.sigma
                if cost_time <= self.T:
                    p_hat_p = p_hat + ';' + vu
                    w_p_hat_p = self.conv(w_p_hat, w_vu)
                    tcost1 = time.time()
                    p_hat_max = get_maxP(w_p_hat, u)
                    all_expire += time.time() - tcost1
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
        edges, nodes,  G2, speed_dict = self.get_graph(edge_desty, vedge_desty)
        points = self.get_axes()
        df2 = open(self.query_name,  'rb')
        r_pairs = pickle.load(df2)
        df2.close()
        nodes_order, i = {}, 0
        for node in nodes:
            nodes_order[node] = i
            i += 1
        PT = 5
        all_iters, flag = 0, False
        One_Plot = np.zeros(20).reshape(4, 5)
        One_Plot2 = np.zeros(20).reshape(4, 5)
        One_Sums = np.zeros(20).reshape(4, 5)
        one_dis = -1
        cate = ['0-5km', '5-10km', '10-25km', '25-35km']
        for pairs in r_pairs:
            one_dis += 1
            for pair_ in pairs[:]:
                _ = self.get_U2(pair_[-1])
            dij_time = 0.0
            print('distance category %s'%cate[one_dis])
            for pair_ in pairs[:]:
                all_iters += 1
                tstart = time.time()
                print('o-d pair: %s'%pair_[0]+'-'+pair_[1])
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
                st1, time_budget = start, 0.0
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
                #print('time budget: %f'%time_budget)
                U = self.get_U2(desti)
                if len(U) == 0 : continue
                tend = time.time()
                for t_b_, t_b in enumerate([0.5, 0.75, 1.0, 1.25, 1.5]):
                    tstart = time.time()
                    self.T = time_budget * t_b
                    best_p, max_m, best_c, best_pw, all_expires, all_rounds = self.rout(start, desti, edge_desty, vedge_desty, nodes_order, U, G2, points, speed_dict)
                    if best_p == 'none1' or best_p == 'none2' or best_p == 'none':
                        continue
                    tend = time.time()
                    One_Plot2[one_dis][t_b_] += tend - tstart - all_expires
                    One_Sums[one_dis][t_b_] += 1

        One_Plot2 = One_Plot2 / One_Sums 
        One_Plot2 = np.nan_to_num(One_Plot2)

        print('The time cost for routing')
        print(One_Plot2)
        print('Time cost for budget: 50%, 75%, 100%, 125%, 150%')
        print(One_Plot2.mean(0))
        print('Time cost for distance: 0-5km, 5-10km, 10-25km, 25-35km')
        print(One_Plot2.mean(1))

if __name__ == '__main__':

    threads_num = 15
    dinx = 30
    parser = argparse.ArgumentParser(description='T-BS')
    parser.add_argument('--sig', default=0, type=int)
    args = parser.parse_args()
    if args.sig == 0:
        sigma, eta = 10, 800
    elif args.sig == 1:
        sigma, eta = 30, 333
    elif args.sig == 2:
        sigma, eta = 60, 170
    elif args.sig == 3:
        sigma, eta = 90, 111
    else:
        print('wrong sig , exit')
        sys.exit()
    print('eta: %d, sigma: %d'%(eta, sigma))
    subpath = '../data/'
    policy_path = subpath + 'matrix_u/%d/'%sigma
    true_path = 'T-path_desty.json'
    virtual_path = 'V-path_desty'
    edge_desty = 'edge_desty.json'
    axes_file =  '../data/vertices.txt'
    speed_file = '../data/map_ngr'
    query_name = subpath + 'odpairs.txt'
    rout = Rout(policy_path, true_path, virtual_path, edge_desty, subpath, axes_file, speed_file, query_name, sigma, eta)
    rout.main()


