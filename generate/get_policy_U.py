import time
import numpy as np
import queue 
import json 
import sys, os
import multiprocessing
from multiprocessing import Process

import networkx as nx
from tpath import TPath

class Rout:
    #def __init__(self, G, maxsize, node_size, eta, node_list, graph_store_name, time_budget ):
    def __init__(self, subpath, graph_store_name, time_budget, filename, fpath_desty, fvedge_desty, maxsize, degree_file, fedge_desty, umatrix_path, upath, process_num, speed_file, eta=12, sigma=3):
        self.maxsize = maxsize
        self.eta = eta
        self.sigma = sigma
        #self.Q = queue.Queue(maxsize=maxsize)
        self.graph_store_name = graph_store_name
        self.time_budget = time_budget 
        self.filename = filename
        self.fpath_desty = fpath_desty
        self.fvedge_desty = fvedge_desty
        self.subpath = subpath
        self.degree_file = degree_file
        self.fedge_desty = fedge_desty
        self.umatrix_path = umatrix_path
        self.upath = upath
        self.process_num = process_num
        self.speed_file = speed_file

    def get_speed(self, ):
        speed_dict = {}
        with open(self.speed_file) as fn:
            for line in fn:
                line = line.strip().split('\t')
                speed_dict[line[0]] = {3600*float(line[1])/float(line[2]): 1.0}
        return speed_dict

    def get_graph(self, edge_desty):
        speed_dict = self.get_speed()
        fn = open(self.subpath+self.graph_store_name)
        edges, edges_p, nodes, nodes_p = [], [], set(), set()
        l, l2 = 0, 0
        for line in fn:
            line = line.strip()
            if line not in edge_desty :
                if line in speed_dict:
                    edge_desty[line] = speed_dict[line]
                else:
                    l += 1
                continue
            l2 += 1
            line = line.split('-')
            #if line[0] == '473061' or line[1] == '473061':
            #    print(line)
            #edges_p.append((line[0], line[1]))
            #nodes_p.add(line[0])
            #nodes_p.add(line[1])
            edges.append((line[0], line[1]))
            nodes.add(line[0])
            nodes.add(line[1])
        fn.close()
        print('%d %d'%(l, l2))
        for key in speed_dict:
            kk = key.strip().split('-')
            nodes.add(kk[0])
            nodes.add(kk[1])
            edges.append((kk[0], kk[1]))
            if key not in edge_desty:
                edge_desty[key] = speed_dict[key]
        #nodes_other = nodes - nodes_p
        nodes = list(nodes)
        nodes_p = list(nodes_p)
        print('len nodes %d'%len(nodes))
        print('len nodes_p %d'%len(nodes_p))
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        return edges, nodes, edge_desty, G
    
    def get_dict(self, ):
        #with open(self.subpath+self.fpath_desty) as js_file:
        #    path_desty = json.load(js_file)
        #with open(self.subpath+self.fvedge_desty) as js_file:
        #    vedge_desty = json.load(js_file)
        
        with open(self.subpath+self.fedge_desty) as js_file:
            edge_desty = json.load(js_file)

        return edge_desty
    def get_nodes(self, ):
        gnodes = os.listdir('./res3/u_mul_matrix3/')
        return gnodes

    #def get_dijkstra(self, source, target):
    #    path = nx.dijkstra_path(G, source, target)
    #    return path

    def computeU(self, v_i, v_order, edge_desty, nodes_order, U, Q, G, iset):
        T_s, T_e = np.inf, self.eta #*self.sigma
        T_e_ = T_e
        success = list(G.successors(v_i))
        real_success = []
        for suc in success:
            if suc in iset:
                real_success.append(suc)
        if len(success) < 1:
            print('len success: %d'%len(success))
        if len(real_success) < 1:
            print('len real_success: %d'%len(real_success))
        UZ = np.zeros(len(real_success) * (self.eta+1)).reshape(len(real_success), self.eta+1)
        if len(real_success) > 0: iset.add(v_i)
        for l, z in enumerate(real_success):
            z_order = nodes_order[z]
            #if U[z_order][0] == -1: self.computeU(z, z_order, edge_desty, nodes_order, U, Q, G, iset)
            if U[z_order][0] == -1: print('u z order -1')
            vkey = v_i + '-' + z
            vkey_ = z + '-' +v_i  
            if vkey in edge_desty:
                z_low = min([float(l_) for l_ in edge_desty[vkey].keys()])
                pdf = edge_desty[vkey]
            else:
                print('no pdf. ..') 

            pdf = dict(sorted(pdf.items(), key=lambda t: t[0]))
            u_min = -1
            for i in range(self.eta+1):
                if U[z_order][i] > 0:
                    u_min = i*self.sigma 
                    break
            if u_min == -1: 
                print('error ..')
                print('v_i %s and z %s'%(v_i, z))
                print(U[z_order])
                sys.exit()
            t_z = int(u_min + z_low)
            #if t_z % self.sigma != 0:
            #    t_z = (int(t_z/self.sigma)+1)*self.sigma
            t_z_ = int(t_z / self.sigma)
            if t_z_ > self.eta:
                print('z_low %f, t_z %d, t_z_ %d, u_min %d' %(z_low, t_z, t_z_, u_min))
                print('t_z_ larger eq than self.eta, continue')
                print(U[z_order])
                continue
            T_s = min(int(t_z/self.sigma), T_s)
            UZ[l][t_z_] = 0
            temp = []
            last_te = -1
            for c in pdf:
                te = u_min + float(c)
                #if te % self.sigma !=0 : te = (te/self.sigma+1)*self.sigma
                te_ = int(te / self.sigma)
                if te_ <= T_e and UZ[l][te_] < 1:
                    temp.append((te, te_))
                    #uk = self.sigma*(self.eta-1) - float(c)
                    uk = self.sigma*(self.eta) - float(c)
                    uk_ = int(uk/self.sigma)
                    if last_te == -1:
                        UZ[l][te_] = pdf[c]*U[z_order][uk_]
                        last_te = te_
                    else:
                        UZ[l][te_] = UZ[l][last_te] + pdf[c]*U[z_order][uk_]
                        last_te = te_
            if len(temp) == 0:
                if len(success) > 0:
                    print('len temp is 0, len real is larger than 0, continue, pid %d, len success %d'%(os.getpid(), len(success)))
                    continue
                else:
                    print('len temp is 0, len real is 0, exit, pid %d, len success %d '%(os.getpid(), len(success)))
                    sys.exit()
            (te1, te_1) = temp[0]
            (ten, te_n) = temp[-1]
            last_te1 = int(te_1)
            ta1 = -1
            for (ta, ta1) in temp[1:]: # fullfill UZ
                while last_te1 < ta1:
                    UZ[l][last_te1] = UZ[l][int(te_1)]
                    last_te1 += 1
                last_te1 = int(ta1)
            if ta1 != -1:
                while last_te1 < self.eta:
                    UZ[l][last_te1] = UZ[l][int(ta1)]
                    last_te1 += 1
            T_e_ = min(te_n, T_e_)
        T_e = T_e_
        if T_s > T_e:
            print('T_s larger than T_e , error')
            #print(T_s)
            #print(T_e)
            #continue
            sys.exit()
        for T_prime in range(self.eta):
            if T_prime < T_s:
                U[v_order][T_prime] = 0
            elif T_prime >= T_e:
                U[v_order][T_prime] = 1
            else: 
                max_ = -1
                for k in range(len(real_success)):
                    max_ = max(UZ[k][T_prime], max_)
                U[v_order][T_prime] = max_
        #print('T_s %d, T_e %d' %(T_s, T_e))
        del UZ
        #if vi == '447866':
        #    print('vi %s'%vi)
        #    print(U[vi])
        return True


    def rout_(self, v_d, edge_desty, edges, nodes, nodes_order, G):
        N = len(nodes)
        U = np.zeros(N * self.eta).reshape(N, self.eta)
        U.fill(-1)
        Q = []
        iset = set()
        print('pid %d, v d %s' %(os.getpid(), v_d))
        if len(Q) != 0:
            print('error, Q is not empty, exit')
            sys.exit()
        Q.append(v_d)
        iset.add(v_d)
        for j in range(self.eta):
            U[nodes_order[v_d]][j] = 1.0
        lsucess, lorder = '', -1
        flag = True
        while len(Q) != 0:
            v_i = Q.pop(0)
            v_order = nodes_order[v_i]
            #print('v_order %d'%v_order)
            #print('len of predecessors: %d'%(len(list(self.G.predecessors(v_i)))))
            if v_i == v_d:
                U[v_order].fill(1) # = 1
                iset.add(v_i)
            else:
                self.computeU(v_i, v_order, edge_desty, nodes_order, U, Q, G, iset)
        
            #print(list(G.predecessors(v_i)))
            for v in list(G.predecessors(v_i)):
                if U[nodes_order[v]][0] == -1:
                    if v not in Q :
                        Q.append(v)
        print('pid %d, v_d %s, len iset %d'%(os.getpid(), v_d, len(iset)))
        return U
    
    def rout(self, sub_nodes, edge_desty, edges, nodes, gnodes, nodes_order, G):
        print('hehe')
        print('pid %d'%os.getpid())
        #rout_res = []
        #flag = False
        for v_d_ in sub_nodes:
            U_1 = self.rout_(v_d_, edge_desty, edges, nodes, nodes_order, G)
            #if not flag:
            #U_2 = np.delete(U_1, final_inx, 0)
            #del U_1
            #self.save_matrix(v_d_, U_1, nodes_p)
            self.save_matrix(v_d_, U_1, nodes)
            #flag = True
            #np.save(self.umatrix_path+v_d_, U_2)
            #self.write_file(v_d_)
            #rout_res.append((v_d_, U_, iset_))
            #print(v_d_)
        #print('len rout res %d'%len(rout_res))
        #return rout_res
    
    def write_json(self, dicts, name):
        with open(self.subpath+name, 'w' ) as fw:
            json.dump(dicts, fw, indent=4)

    def write_file(self, name, ilist):
        strs = '\n'.join(str(ilist_) for ilist_ in ilist)
        fn = open(self.upath+name, 'w')
        fn.write(strs)
        fn.close()

    def save_matrix(self, name, U_2, nodes):
        strs = ''
        for ix, ky in enumerate(nodes):
            U_ = U_2[ix]
            strs += ky+';'+';'.join(str(l) for l in U_) + '\n'
        fw = open(self.umatrix_path+name, 'w')
        fw.write(strs)
        fw.close()

    def collect_res(self, sub_res):
        self.result.extend(sub_res)

    def main(self, v_s, v_d):
        edge_desty = self.get_dict()
        edges, nodes, edge_desty, G = self.get_graph(edge_desty)
        gnodes = self.get_nodes()
        '''
        las = -1.0
        for edge in edge_desty:
            ekey = list(edge_desty[edge].keys())
            if len(ekey) == 1:
                las = max(las, float(ekey[0]))
            #print(edge)
            #print(edge_desty[edge])
        print(las)
        sys.exit() '''
        nodes_order, nodes_order_r = {}, {}
        self.result = []
        final_inx = []
        i = 0
        for node in nodes:
            nodes_order[node] = i
            #nodes_order_r[i] = node
            i += 1
        gnodes_other = list(set(gnodes) - set(os.listdir(self.umatrix_path)))
        #N = len(gnodes)
        N = len(gnodes_other)
        #self.write_json(nodes_order, 'nodes_order.json')
        if N % self.process_num == 0:
            t_inx = int(N/self.process_num)
        else:
            t_inx = int(N/self.process_num)+1
        #self.rout(nodes_p, edge_desty, edges, nodes, nodes_p, nodes_order, G, final_inx)
        
        pool = multiprocessing.Pool(self.process_num)
        print('begin ... ')
        for len_thr in range(self.process_num):
            mins = min((len_thr+1)*t_inx, N)
            #sub_array = gnodes[len_thr*t_inx:mins]
            sub_array = gnodes_other[len_thr*t_inx:mins]
            #args_ = [sub_array, vedge_desty]#, edge_desty, path_desty, edges]
            print(len(sub_array))
            pool.apply_async(self.rout, args=(sub_array, edge_desty, edges, nodes, gnodes, nodes_order, G))
            #pool.apply_async(self.rout, args=(sub_array, edge_desty, edges, nodes, nodes_order, G), callback=self.collect_res)
            #self.rout(sub_array, edge_desty, edges, nodes, nodes_order, G)
            #pool.apply_async(self.rout, args=(sub_array, edge_desty,  edges, nodes, nodes_order, G), callback=self.collect_res)
        pool.close()
        pool.join()
        
        print('len result %d'%len(self.result))
        print('process end ...')
        #print(self.result)
        #for res in self.result:
        #    self.save_matrix(res[0], res[1], nodes)
        #    self.write_file(res[0], res[2])
        #print('store end ...')

if __name__ == '__main__':
    threads_num = 15
    process_num = 20
    time_budget = 1000
    maxsize = 10000
    eta, sigma = 333, 30
    eta, sigma = 333, 30
    eta, sigma = 170, 60
    eta, sigma = 200, 10
    dinx = 3
    subpath = './res%d/'%dinx
    filename = '../../data/AAL_short_%d.csv'%dinx
    fpath_desty = 'KKdesty_num_%d.json'%threads_num
    #fvedge_desty = 'M_vedge_desty_num_%d.json'%threads_num
    fvedge_desty = 'KK_vedge_desty2.json'
    fedge_desty = 'M_edge_desty.json'
    graph_store_name = 'KKgraph_%d.txt'%threads_num
    degree_file = 'KKdegree2_%d.json'%threads_num
    umatrix_path = subpath+'u_mul_matrix_sig%d/'%sigma
    upath = subpath+'u_mul_path_sig%d/'%sigma
    speed_file = '../../data/AAL_NGR'
    rout = Rout(subpath, graph_store_name, time_budget, filename, fpath_desty, fvedge_desty, maxsize, degree_file, fedge_desty, umatrix_path, upath, process_num, speed_file, eta, sigma)
    aa = '271590-352865;352865-352862'
    v_s = '271590'
    v_d = '352862'
    rout.main(v_s, v_d)

