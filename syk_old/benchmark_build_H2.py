from sys import argv
from math import sqrt
from os.path import join

L = int(argv[1]) # number of spins (not # of majoranas!)
# n = int(argv[2]) # number of disorder samples
msc_dir = '/n/home01/ytan/scratch/deviation_ee/msc_syk'

from dynamite import config
config.L = L

from dynamite.operators import op_sum, op_product, Operator
from dynamite.extras import majorana
from itertools import combinations
import numpy as np
# from scipy.sparse.linalg import trace

from timeit import default_timer
from datetime import timedelta

from dynamite.tools import track_memory, get_max_memory_usage, get_cur_memory_usage
track_memory()

class EventTimer:
    def __init__(self):
        self.global_start = default_timer()
        self.sub_start = None

    @classmethod
    def _td(cls,start,end):
        return timedelta(0,end-start)

    def end_event(self,t=None):

        if t is None:
            t = default_timer()

        print(' (%s, cur %s Gb, max %s Gb)' % (self._td(self.sub_start,t),
                                               str(get_cur_memory_usage()/1E9),
                                               str(get_max_memory_usage()/1E9)),sep='')
        ss = self.sub_start
        self.sub_start = None
        return t - ss

    def begin_event(self,name):
        new_start = default_timer()
        if self.sub_start is not None:
            d = self.end_event(new_start)
        else:
            d = None
        print(self._td(self.global_start,new_start),
              name,end='')
        self.sub_start = new_start
        return d

et = EventTimer()
et.begin_event('Start')

N = 2*L # number of Majoranas
J = 1.0
j = 0 # sample index

start = default_timer()

###########################################
# Test for complete traceless part
###########################################

# load the brute force H2 (with trace)
H2_with_trace = Operator.load(join(msc_dir,'H2_with_trace_'+str(L)+'_'+str(j)+'.msc'))
H2_mat_with_trace = H2_with_trace.to_numpy().toarray() # if only do to_numpy, it will be a sparse matrix
H2_mat_traceless = H2_mat_with_trace - np.trace(H2_mat_with_trace)/len(H2_mat_with_trace)*np.eye(len(H2_mat_with_trace))
np.save('H2_trace_'+str(L)+'_'+str(j)+'.npy', np.trace(H2_mat_with_trace))
np.save('H2_mat_with_trace_'+str(L)+'_'+str(j)+'.npy', H2_mat_with_trace)
np.save('H2_mat_traceless_'+str(L)+'_'+str(j)+'.npy', H2_mat_traceless)

# load the brute force H2 build using matrix multiplication (with trace)
H2_mat_with_trace_mat_mul = Operator.load(join(msc_dir,'H2_with_trace_mat_mul'+str(L)+'_'+str(j)+'.msc'))
H2_mat_with_trace_mat_mul = H2_mat_with_trace_mat_mul.to_numpy().toarray()
H2_mat_traceless_mat_mul = H2_mat_with_trace_mat_mul - np.trace(H2_mat_with_trace_mat_mul)/len(H2_mat_with_trace_mat_mul)*np.eye(len(H2_mat_with_trace_mat_mul))
# np.save('H2_trace_mat_mul_'+str(L)+'_'+str(j)+'.npy', np.trace(H2_mat_with_trace))
# np.save('H2_mat_with_trace_mat_mul_'+str(L)+'_'+str(j)+'.npy', H2_mat_with_trace_mat_mul)
# np.save('H2_mat_traceless_mat_mul_'+str(L)+'_'+str(j)+'.npy', H2_mat_traceless_mat_mul)

# load the cleverly decomposed H2 (without trace)
# H2_5uniq = Operator.load(join(msc_dir,'H2_5uniq_method2_'+str(L)+'_'+str(j)+'.msc'))
H2_6uniq = Operator.load(join(msc_dir,'H2_6uniq_'+str(L)+'_'+str(j)+'.msc'))
# H2_7uniq = Operator.load(join(msc_dir,'H2_7uniq_method2_'+str(L)+'_'+str(j)+'.msc'))
H2_8uniq = Operator.load(join(msc_dir,'H2_8uniq_'+str(L)+'_'+str(j)+'.msc'))
H2_sum_compo = op_sum([H2_6uniq, H2_8uniq])
H2_mat_sum_compo = H2_sum_compo.to_numpy().toarray()
np.save('H2_mat_sum_compo_'+str(L)+'_'+str(j)+'.npy', H2_mat_sum_compo)

print('\n shape: ', H2_mat_with_trace.shape)
print('compare bruteforce (mat mul) and bruteforace (single loop): ', np.allclose(H2_mat_traceless_mat_mul, H2_mat_traceless))
print('compare bruteforce (mat mul) and clever decomposition method: ', np.allclose(H2_mat_traceless_mat_mul, H2_mat_sum_compo))

###########################################
# Test 6-unique element part only
###########################################

# # load H2_6uniq using method1
# H2_6uniq_method1 = Operator.load(join(msc_dir,'H2_6uniq_'+str(L)+'_'+str(j)+'.msc'))
# H2_mat_6uniq_method1 = H2_6uniq_method1.to_numpy().toarray()
# np.save('H2_mat_6uniq_method1_'+str(L)+'_'+str(j)+'.npy', H2_mat_6uniq_method1)

# # load H2_6uniq using method2
# H2_6uniq_method2 = Operator.load(join(msc_dir,'H2_6uniq_method2_'+str(L)+'_'+str(j)+'.msc'))
# H2_mat_6uniq_method2 = H2_6uniq_method2.to_numpy().toarray()
# np.save('H2_mat_6uniq_method2_'+str(L)+'_'+str(j)+'.npy', H2_mat_6uniq_method2)

# print('\n compare method1 and method2: ', np.allclose(H2_mat_6uniq_method1, H2_mat_6uniq_method2))

###########################################
# Test 8-unique element part only
###########################################

# load H2_8uniq using method1
# H2_8uniq_method1 = Operator.load(join(msc_dir,'H2_8uniq_'+str(L)+'_'+str(j)+'.msc'))
# H2_mat_8uniq_method1 = H2_8uniq_method1.to_numpy().toarray()
# np.save('H2_mat_8uniq_method1_'+str(L)+'_'+str(j)+'.npy', H2_mat_8uniq_method1)

# load H2_8uniq using method2
# H2_8uniq_method2 = Operator.load(join(msc_dir,'H2_8uniq_method2_'+str(L)+'_'+str(j)+'.msc'))
# H2_mat_8uniq_method2 = H2_8uniq_method2.to_numpy().toarray()
# np.save('H2_mat_8uniq_method2_'+str(L)+'_'+str(j)+'.npy', H2_mat_8uniq_method2)

# print('\n compare method1 and method2: ', np.allclose(H2_mat_8uniq_method1, H2_mat_8uniq_method2))

et.begin_event('End\n\n')
print('Total max memory usage:',get_max_memory_usage()/(1E9),'Gb')

