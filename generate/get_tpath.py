import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os, copy
import threading
import multiprocessing 
from multiprocessing import Process
import json
import operator

class TPath():
    def __init__(self, filename, subpath, dinx, process_num):
        self.filename  = filename
        self.subpath = subpath
        self.dinx = dinx
        self.process_num = process_num
        self.path_time_freq = {}
        self.w_data = multiprocessing.Manager().dict()

    def load(self, ):
        data = pd.read_csv(self.filename)
        return data

    def plot(self, data):
        x = np.arange(len(data))
        plt.plot(x[10000:], data[10000:])
        plt.savefig('./fig/freqs3.png')
        plt.close()

    def get_edge_freq(self, data):
        seg_ids = data.seg_id.values
        N = len(seg_ids)
        edge_freq = {}
        for i in range(N):
            if seg_ids[i] not in edge_freq:
                edge_freq[seg_ids[i]] = 1
            else:
                edge_freq[seg_ids[i]] += 1
        return edge_freq

    def get_edge_time_freq(self, data):
        seg_ids, times = data.seg_id.values, data.travel_time.values
        N = len(seg_ids)
        edge_time_freq = {}
        for i in range(N):
            if seg_ids[i] not in edge_time_freq:
                edge_time_freq[seg_ids[i]] = {}
                edge_time_freq[seg_ids[i]][times[i]] = 1
            else:
                if times[i] not in edge_time_freq[seg_ids[i]]:
                    edge_time_freq[seg_ids[i]][times[i]] = 1
                else:
                    edge_time_freq[seg_ids[i]][times[i]] = +1
        for edge in edge_time_freq:
            all_value = sum(edge_time_freq[edge])
            edge_time_freq[edge] = dict(sorted(edge_time_freq[edge].items(), key=operator.itemgetter(1), reverse=True))
            
        return edge_time_freq

    def get_dataframe(self, dict_data):
        return pd.DataFrame({'seg_id': dict_data.keys(), 'seg_freq': dict_data.values()})

    def temp_(self, data, edge_freq):
        A = []
        for v in data.seg_id.values:
            #print(edge_freq[v])
            A.append(edge_freq[v])
        data['seg_id_freq'] = pd.DataFrame(A)
        return data

    def full_(self, data):
        N = len(data)
        print('lens N: %d'%N)
        full_len = {} #full path's freq
        last_ntid, path_key = -1, ''
        path_len = {} #full path's length
        lens = 0
        path_time = {}
        one_time = []
        for data_ in data.values:
            #print(data_)
            ntid, path, inx = data_[1], data_[-3], data_[-2]
            t_time = data_[-1]
            if ntid != last_ntid:
                if path_key != '':
                    if path_key in full_len:
                        full_len[path_key] += 1
                    else:
                        full_len[path_key] = 1
                        path_len[path_key] = lens
                    if path_key in path_time:
                        path_time[path_key].append(one_time)
                    else:
                        path_time[path_key] = [one_time]

                path_key = path
                lens = 1
                one_time = [t_time]
            else:
                path_key += '-' + path
                lens += 1
                one_time.append(t_time)
            last_ntid = ntid
        if path_key in full_len:
            full_len[path_key] += 1
        else:
            full_len[path_key] = 1
            path_len[path_key] = lens
        if path_key in path_time:
            path_time[path_key].append(one_time)
        else:
            path_time[path_key] = [one_time]
        return full_len, path_len, path_time
    
    def cut_it(self, path_freq, path_len, path_time, k):
        keys = path_len.keys()
        freq_dict = {}
        kk = 2 * k -1

        def cut_it_(paths):
            paths = paths.split('-')
            pls = []
            lens1 = int((len(paths)+1)/2)
            p_lens = lens1 - k + 1
            for i in range(p_lens):
                skey = '-'.join(paths[i*2 + j] for j in range(2 * k))
                pls.append(skey)
            return pls 

               
        def is_full(f_index, inx, s_len):
            iset = set()
            for f_inx in f_index:
                if f_inx != inx:
                    for i in range(k):
                        iset.add(f_inx + i)
                        #print(f_inx+i)
            if len(iset) < s_len:
                return False
            elif len(iset) == s_len:
                return True
            else:
                print('error happend, please check code ...')
                print('error_1 %d %d %d %d' % (len(iset), s_len, inx, k))
                print(f_index)
                print(iset)
                return False


        for key in keys:
            if path_len[key] < k:
                continue
            elif path_len[key] == k:
                freq_dict[key] = path_freq[key]
            else:
                path1 = cut_it_(key)
                for p1 in path1:
                    if p1 in freq_dict:
                        freq_dict[p1] += 1
                    else:
                        freq_dict[p1] = 1
        
        cut_path = {}
        path_time_freq = {}
        for key in keys:
            if path_len[key] < k:
                cut_path[key] = path_freq[key]
            elif path_len[key] == k:
                cut_path[key] = path_freq[key]
                path_time_ = path_time[key]
                p_time = [sum(p_t) for p_t in path_time_]
                if key not in path_time_freq:
                    path_time_freq[key] = {}
                    for p_t in p_time:
                        if p_t not in path_time_freq[key]:
                            path_time_freq[key][p_t] = 1
                        else:
                            path_time_freq[key][p_t] += 1
                else:
                    for p_t in p_time:
                        if p_t not in path_time_freq[key]:
                            path_time_freq[key][p_t] = 1
                        else:
                            path_time_freq[key][p_t] += 1

            else:
                path1 = cut_it_(key)
                freqs = [freq_dict[p1] for p1 in path1]
                index = np.argsort(freqs)
                f_index = list(index)
                s_len = len(index) + k - 1
                for f_inx in f_index:
                    k1 = path1[f_inx]
                    if k1 in cut_path:
                        cut_path[k1] += 1
                    else:
                        cut_path[k1] = 1
                path_time_ = path_time[key]
                for f_inx in f_index:
                    k1 = path1[f_inx]
                    p_time = [sum(p_t[f_inx:f_inx+k+1]) for p_t in path_time_]
                    if k1 not in path_time_freq:
                        path_time_freq[k1] = {}
                        for p_t in p_time:
                            if p_t not in path_time_freq[k1]:
                                path_time_freq[k1][p_t] = 1
                            else:
                                path_time_freq[k1][p_t] += 1
                    else:
                        for p_t in p_time:
                            if p_t not in path_time_freq[k1]:
                                path_time_freq[k1][p_t] = 1
                            else:
                                path_time_freq[k1][p_t] += 1

        for k_1 in path_time_freq:
            path_time_freq[k_1] = dict(sorted(path_time_freq[k_1].items(), key=operator.itemgetter(1), reverse=True))
        flag = self.write2(cut_path, k)
        return flag, cut_path, path_time_freq

    def write(self, paths, k):
        B = [1000, 500, 200, 100, 50, 30, 20, 10, 5, 1, 1]
        A = [[0]*k for i in range(len(B))]
        for key in paths:
            lens = int((len(key.split('-')) + 1)/2)
            for i in range(len(B) -1):
                if paths[key] > B[i]:
                    A[i][lens-1] += 1
            if paths[key] == B[i]:
                A[-1][lens-1] += 1
        strs = '>,'
        strs += ','.join(str(i+1)+'-edge' for i in range(k)) + '\n'
        for i in range(len(B)-1):
            strs += '>%d,'%B[i]
            strs += ','.join(str(A[i][j]) for j in range(k)) + '\n' 
        strs += '=%d,'%B[-1]
        strs += ','.join(str(A[-1][j]) for j in range(k))
        fname = './full_cut/stat_10_k%d.csv'%k
        with open(fname, 'w') as files:
            files.write(strs)
            files.close()

    def write2(self, paths, k):
        B = [1000, 500, 200, 100, 50, 30, 20, 10, 5, 1, 1]
        A = [0] * len(B)
        for key in paths:
            lens = int((len(key.split('-')) + 1)/2)
            if lens == k:
                for i in range(len(B)-1):
                    if paths[key] > B[i]:
                        A[i] += 1
                        break
                if paths[key] == B[i]:
                    A[-1] += 1
        titles = '%d-edge'%k
        if sum(A[:5]) == 0:
            #fname = './full_cut/all_stat_10_k%d.csv'%k
            #self.w_data.to_csv(fname, index=None)
            return False
        else:
            self.w_data[titles] = A
            return True

    def write_json(self, js_dict, fname):
        #json.dumps(js_dict, fname )
        with open(fname, 'w') as fw:
            json.dump(js_dict, fw, indent=4)

    def write_json2(self, js_dict, cut_path, fname):
        #json.dumps(js_dict, fname )
        fre_n = self.dinx
        js_dict_ = copy.deepcopy(js_dict)
        for key in js_dict.keys():
            if cut_path[key] <= fre_n:
                del js_dict_[key]
        if len(js_dict_) < 1: return
        with open(fname, 'w') as fw:
            json.dump(js_dict_, fw, indent=4)


    def thread_fun(self, path_freq, path_len, path_time, inxs_array):
        for inx_k in inxs_array:
            values, cut_path, path_time_freq = self.cut_it(path_freq, path_len, path_time, inx_k)
            if not values:
                print(inx_k)
                #break
            else:
                fname = self.subpath+'path_travel_time_%d.json'%inx_k
                self.write_json2(path_time_freq, cut_path, fname)
            #print('hehe')
        #return path_time_freq
        self.path_time_freq = path_time_freq
        return path_time_freq


    def main(self, ):
        print('load data ...')
        data = self.load()
        print('edge freq ...')
        edge_freq = self.get_edge_freq(data)
        print('get path len and freq ...')
        path_freq, path_len, path_time = self.full_(data)#full_len is the frequency of a path; path_len is the length of a path. 
        k = 0
        B = [1000, 500, 200, 100, 50, 30, 20, 10, 5, 1, 1]
        t1 = ['>%d'%B[l] for l in range(len(B)-1)]
        t1.append('=1')
        self.w_data = multiprocessing.Manager().dict()
        print('cut path ...')
        inxs = [l for l in range(2, 61)]
        print(inxs)
        threads_num = self.process_num
        t_inxs = int(len(inxs) / threads_num) +1
        thread_array = []
        for len_thr in range(threads_num):
            mins = min((len_thr+1)*t_inxs, len(inxs))
            inxs_array = inxs[len_thr * t_inxs : mins]
            print(inxs_array)
            threads_ = Process(target=self.thread_fun, args=(path_freq, path_len, path_time, inxs_array))
            thread_array.append(threads_)
            print('start thread %d' %len_thr)
            threads_.start()

        for len_thr in range(threads_num):
            thread_array[len_thr].join()
        print('write file ...')
        fname = self.subpath+'AAL_stat_%d.csv'%self.dinx
        ww_data = {}
        kkeys = ['%d-edge'%kke for kke in range(2, 100)]
        for kk in kkeys:
            if kk in self.w_data:
                ww_data[kk] = self.w_data[kk]
        ww_data = pd.DataFrame.from_dict(ww_data)
        ww_data.insert(loc=0, column='>', value=t1)
        ww_data.to_csv(fname, sep=';', index=None)

        
if __name__ == '__main__':
    dinx = 50
    process_num = 20
    filename = '../../data/AAL_short_%d.csv'%dinx
    subpath = './res%d/'%dinx
    tpath = TPath(filename, subpath, dinx, process_num)
    tpath.main()


