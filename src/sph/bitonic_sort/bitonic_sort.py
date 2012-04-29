import pyopencl as cl
import numpy as np
import os

class BitonicSort(object):
    def __init__(self, ctx, queue, local_size_limit=None):
        self.ctx = ctx

        if local_size_limit is None:
            local_size_limit = min(device.max_work_group_size for device in self.ctx.devices)//2
        
        self.queue = queue
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        with open('%s/BitonicSort_b.cl' % cur_dir) as f:
            self.prg = cl.Program(ctx, f.read() % { 'local_size_limit': local_size_limit }).build()
        self.local_size_limit = local_size_limit

    def sort_in_place(self, d_key, d_val, array_length, dir):
        self.sort(d_key, d_val, d_key, d_val, array_length, dir)

    def sort(self, d_dst_key, d_dst_val, d_src_key, d_src_val, array_length, dir):
        if array_length < self.local_size_limit:
            batch = self.local_size_limit // array_length
        else:
            batch = 1
        
        self._sort(d_dst_key, d_dst_val, d_src_key, d_src_val, batch, array_length, dir)
        

    def _sort(self, d_dst_key, d_dst_val, d_src_key, d_src_val, batch, array_length, dir):
        """
        dir: 0 for descending sort, 1 for ascending.
        """
        assert array_length >= 2

        queue = self.queue
        prg = self.prg
        local_size_limit = self.local_size_limit
        

        def int_log2(L):
            if not L:
                return 0

            log2 = 0
            while L & 1 == 0:
                L >>= 1
                log2 += 1
            return log2


        # only power-of-two array lengths are supported
        log2L = int_log2(array_length)
        assert 2**log2L == array_length

        if array_length <= local_size_limit:
            assert (batch * array_length) % local_size_limit == 0

            local_work_size = (local_size_limit // 2, )
            global_work_size = (batch * array_length // 2, )

            kernel_args = (
                d_dst_key,
                d_dst_val,
                d_src_key,
                d_src_val,
                np.uint32(array_length),
                np.uint32(dir),
                )

            prg.bitonicSortLocal(queue, global_work_size, local_work_size, *kernel_args)
            queue.finish()
        else:
            # launch bitonicSortLocal1

            local_work_size = (local_size_limit // 2, )
            global_work_size = (batch * array_length // 2, )

            kernel_args = (
                d_dst_key,
                d_dst_val,
                d_src_key,
                d_src_val,
                )

            prg.bitonicSortLocal1(queue, global_work_size, local_work_size, *kernel_args)
            queue.finish()

            size = 2*local_size_limit
            while size <= array_length:
                stride = size // 2
                while stride > 0:
                    if stride >= local_size_limit:
                        # launch bitonicMergeGlobal

                        local_work_size = (local_size_limit // 4, )
                        global_work_size = (batch * array_length // 2, )

                        kernel_args = (
                            d_dst_key,
                            d_dst_val,
                            d_src_key,
                            d_src_val,
                            np.uint32(array_length),
                            np.uint32(size),
                            np.uint32(stride),
                            np.uint32(dir),
                            )

                        prg.bitonicMergeGlobal(queue, global_work_size, local_work_size, *kernel_args)
                        queue.finish()
                    else:
                        # launch bitonicMergeLocal
                        local_work_size = (local_size_limit // 2, )
                        global_work_size = (batch * array_length // 2, )

                        kernel_args = (
                            d_dst_key,
                            d_dst_val,
                            d_src_key,
                            d_src_val,
                            np.uint32(array_length),
                            np.uint32(stride),
                            np.uint32(size),
                            np.uint32(dir),
                            )

                        prg.bitonicMergeLocal(queue, global_work_size, local_work_size, *kernel_args)
                        queue.finish()

                    stride >>= 1

                size <<= 1

if __name__ == '__main__':
    N = 2<<20
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

    s = BitonicSort(ctx, queue)
    from time import time
    t = time()
    s.sort_in_place(d_keys, d_vals, N, 1)
    print "%ims" % ((time()-t)*1000)

    cl.enqueue_copy(queue, keys, d_keys)
    cl.enqueue_copy(queue, vals, d_vals)
    queue.finish()
    print np.linalg.norm(keys-sorted_keys)
    print np.linalg.norm(vals-sorted_vals)
