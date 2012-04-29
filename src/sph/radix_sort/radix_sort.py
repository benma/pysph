# https://github.com/enjalot/adventures_in_opencl/tree/master/experiments/radix/nv

import pyopencl as cl
import numpy as np
import struct, os

mf = cl.mem_flags

class RadixSort(object):
    def __init__(self, ctx, queue, max_elements, dtype):
        self.ctx = ctx
        self.queue = queue
        
        self.WARP_SIZE = 32
        self.WORKGROUP_SIZE = 256
        self.MIN_LARGE_ARRAY_SIZE = 4 * self.WORKGROUP_SIZE
        cta_size = 128
        self.cta_size = cta_size
        
        self.dtype_size = np.nbytes[dtype]

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        with open('%s/Scan_b.cl' % cur_dir) as f:
            self.scan_prg = cl.Program(self.ctx, f.read()).build()
        with open('%s/RadixSort.cl' % cur_dir) as f:
            self.radix_prg = cl.Program(self.ctx, f.read()).build()

        if (max_elements % (cta_size * 4)) == 0:
            num_blocks = max_elements / (cta_size * 4)
        else:
            num_blocks = max_elements / (cta_size * 4) + 1

        self.d_temp_keys = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.dtype_size * max_elements)
        self.d_temp_values = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.dtype_size * max_elements)

        self.d_counters = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.dtype_size * self.WARP_SIZE * num_blocks)
        self.d_counters_sum = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.dtype_size * self.WARP_SIZE * num_blocks)
        self.d_block_offsets = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.dtype_size * self.WARP_SIZE * num_blocks)

        numscan = max_elements/2/cta_size*16
        if numscan >= self.MIN_LARGE_ARRAY_SIZE:
            #MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE 1024
            self.scan_buffer = cl.Buffer(self.ctx, mf.READ_WRITE, size = self.dtype_size * numscan / 1024)

    def sort(self, d_key, d_val, N):
        key_bits = self.dtype_size * 8
        bit_step = 4
        i = 0
        while key_bits > i*bit_step:
            self.step(d_key, d_val, bit_step, i*bit_step, N);
            i += 1;

    def step(self, d_key, d_val, nbits, startbit, num):
        self.blocks(d_key, d_val, nbits, startbit, num)
        self.queue.finish()

        self.find_offsets(startbit, num)
        self.queue.finish()

        array_length = num/2/self.cta_size*16
        if array_length < self.MIN_LARGE_ARRAY_SIZE:
            self.naive_scan(num)
        else:
            self.scan(self.d_counters_sum, self.d_counters, 1, array_length);
        self.queue.finish()
     
        self.reorder(d_key, d_val, startbit, num)
        self.queue.finish()


    def blocks(self, d_key, d_val, nbits, startbit, num):
        totalBlocks = num/4/self.cta_size
        global_size = (self.cta_size*totalBlocks,)
        local_size = (self.cta_size,)
        blocks_args = (d_key,
                       d_val,
                       self.d_temp_keys,
                       self.d_temp_values,
                       np.uint32(nbits),
                       np.uint32(startbit),
                       np.uint32(num),
                       np.uint32(totalBlocks),
                       cl.LocalMemory(4*self.cta_size*self.dtype_size),
                       cl.LocalMemory(4*self.cta_size*self.dtype_size),
            )
        self.radix_prg.radixSortBlocksKeysValues(self.queue, global_size, local_size, *blocks_args)


    def find_offsets(self, startbit, num):
        totalBlocks = num/2/self.cta_size
        global_size = (self.cta_size*totalBlocks,)
        local_size = (self.cta_size,)
        offsets_args = (self.d_temp_keys,
                        self.d_temp_values,
                        self.d_counters,
                        self.d_block_offsets,
                        np.uint32(startbit),
                        np.uint32(num),
                        np.uint32(totalBlocks),
                        cl.LocalMemory(2*self.cta_size*self.dtype_size),
            )
        self.radix_prg.findRadixOffsets(self.queue, global_size, local_size, *offsets_args)


    def naive_scan(self, num):
        nhist = num/2/self.cta_size*16
        global_size = (nhist,)
        local_size = (nhist,)
        extra_space = nhist / 16 #NUM_BANKS defined as 16 in RadixSort.cpp
        shared_mem_size = self.dtype_size * (nhist + extra_space)
        scan_args = (self.d_counters_sum,
                     self.d_counters,
                     np.uint32(nhist),
                     cl.LocalMemory(2*shared_mem_size)
            )
        self.radix_prg.scanNaive(self.queue, global_size, local_size, *scan_args)


    def scan(self, dst, src, batch_size, array_length):
        self.scan_local1(dst, 
                         src, 
                         batch_size * array_length / (4 * self.WORKGROUP_SIZE),
                         4 * self.WORKGROUP_SIZE)
        self.queue.finish()
        self.scan_local2(dst, 
                         src, 
                         batch_size,
                         array_length / (4 * self.WORKGROUP_SIZE)
            )
        self.queue.finish()
        self.scan_update(dst, batch_size * array_length / (4 * self.WORKGROUP_SIZE))
        self.queue.finish()

    
    def scan_local1(self, dst, src, n, size):
        global_size = (n * size / 4,)
        local_size = (self.WORKGROUP_SIZE,)
        scan_args = (dst,
                     src,
                     cl.LocalMemory(2 * self.WORKGROUP_SIZE * self.dtype_size),
                     np.uint32(size)
            )
        self.scan_prg.scanExclusiveLocal1(self.queue, global_size, local_size, *scan_args)

    def scan_local2(self, dst, src, n, size):
        elements = n * size
        dividend = elements
        divisor = self.WORKGROUP_SIZE
        if dividend % divisor == 0:
            global_size = (dividend,)
        else: 
            global_size = (dividend - dividend % divisor + divisor,)

        local_size = (self.WORKGROUP_SIZE, )
        scan_args = (self.scan_buffer,
                     dst,
                     src,
                     cl.LocalMemory(2 * self.WORKGROUP_SIZE * self.dtype_size),
                     np.uint32(elements),
                     np.uint32(size)
            )
        self.scan_prg.scanExclusiveLocal2(self.queue, global_size, local_size, *scan_args)


    def scan_update(self, dst, n):
        global_size = (n * self.WORKGROUP_SIZE,)
        local_size = (self.WORKGROUP_SIZE,)
        scan_args = (dst,
                     self.scan_buffer)
        self.scan_prg.uniformUpdate(self.queue, global_size, local_size, *scan_args)

    def reorder(self, d_key, d_val, startbit, num):
        totalBlocks = num/2/self.cta_size
        global_size = (self.cta_size*totalBlocks,)
        local_size = (self.cta_size,)
        reorder_args = (d_key,
                        d_val,
                        self.d_temp_keys,
                        self.d_temp_values,
                        self.d_block_offsets,
                        self.d_counters_sum,
                        self.d_counters,
                        np.uint32(startbit),
                        np.uint32(num),
                        np.uint32(totalBlocks),
                        cl.LocalMemory(2*self.cta_size*self.dtype_size),
                        cl.LocalMemory(2*self.cta_size*self.dtype_size)
                    )
        self.radix_prg.reorderDataKeysValues(self.queue, global_size, local_size, *reorder_args)


if __name__ == '__main__':
    N = 2<<10
    keys = np.random.randint(1, 300, size=N).astype(np.uint32)
    vals = np.random.randint(1, 300, size=N).astype(np.uint32)

    indices = keys.argsort()
    sorted_keys = keys[indices]
    sorted_vals = keys[indices]
    
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    d_keys = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=keys)
    d_vals = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vals)

    s = RadixSort(ctx, queue, N, keys.dtype)
    from time import time
    t = time()
    s.sort(d_keys, d_vals, N)
    print "%is" % ((time()-t)*1000)

    cl.enqueue_copy(queue, keys, d_keys)
    cl.enqueue_copy(queue, vals, d_vals)
    queue.finish()
    
    print np.linalg.norm(keys-sorted_keys)
    print np.linalg.norm(vals-sorted_vals)
