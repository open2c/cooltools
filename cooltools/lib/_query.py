from collections import defaultdict
import numpy as np
import pandas as pd

# from scipy.sparse import coo_matrix
from cooler.core import _IndexingMixin


def arg_prune_partition(seq, step):
    """
    Take a monotonic sequence of integers and downsample it such that they
    are at least ``step`` apart (roughly), preserving the first and last
    elements. Returns indices, not values.

    """
    lo, hi = seq[0], seq[-1]
    num = 2 + (hi - lo) // step
    cuts = np.linspace(lo, hi, num, dtype=int)
    return np.unique(np.searchsorted(seq, cuts))


class CSRSelector(_IndexingMixin):
    """
    Instantiates 2D range queries.

    Example
    -------
    >>> selector = CSRSelector(h5, (100, 100), 'count', 10000)
    >>> query = selector[lo1:hi1, lo2:hi2]

    """

    def __init__(self, grp, shape, field, chunksize):
        self.grp = grp
        self.shape = shape
        self.field = field
        self.chunksize = chunksize
        self.offset_selector = grp["indexes"]["bin1_offset"]
        self.bin1_selector = grp["pixels"]["bin1_id"]
        self.bin2_selector = grp["pixels"]["bin2_id"]
        self.data_selector = grp["pixels"][field]

    def _make_getchunk(self, ispan, jspan):
        # Factory for function that executes any piece of a 2D range query by
        # index.

        bin1_selector = self.bin1_selector
        bin2_selector = self.bin2_selector
        data_selector = self.data_selector
        field = self.field
        i0, i1 = ispan
        j0, j1 = jspan

        # coarsegrain the offsets to extract a big chunk of rows at a time
        if (i1 - i0 < 1) or (j1 - j0 < 1):
            offsets = []
            loc_pruned_offsets = []
        else:
            offsets = self.offset_selector[i0 : i1 + 1]
            loc_pruned_offsets = arg_prune_partition(offsets, self.chunksize)

        self._loc_pruned_offsets = loc_pruned_offsets
        # i0 -- matrix row number offset
        # o0 -- corresponding pixel id offset = offsets[0]

        # let's take the downsampled subset of pixel id offsets [o0, ...., o1]
        # each successive pair corresponds to a "piece" of the query
        def getchunk(chunk_id, include_index=False):
            out = {"bin1_id": [], "bin2_id": [], field: []}
            if include_index:
                out["__index"] = []

            # extract a chunk of on-disk rows
            oi, of = loc_pruned_offsets[chunk_id], loc_pruned_offsets[chunk_id + 1]
            p0, p1 = offsets[oi], offsets[of]
            slc = slice(p0, p1)

            bin2_extracted = bin2_selector[slc]
            data_extracted = data_selector[slc]
            if include_index:
                ind_extracted = np.arange(slc.start, slc.stop)

            # go row by row and filter
            for i in range(oi, of):
                # correct the offsets
                lo = offsets[i] - p0
                hi = offsets[i + 1] - p0

                # this row
                bin2 = bin2_extracted[lo:hi]

                # filter for the range of j values we want
                mask = (bin2 >= j0) & (bin2 < j1)
                cols = bin2[mask]

                # apply same mask for data
                data = data_extracted[lo:hi][mask]

                # shortcut for row data
                rows = np.full(len(cols), i0 + i, dtype=bin1_selector.dtype)

                out["bin1_id"].append(rows)
                out["bin2_id"].append(cols)
                out[field].append(data)
                if include_index:
                    out["__index"].append(ind_extracted[lo:hi][mask])

            if len(out):
                for k in out.keys():
                    out[k] = np.concatenate(out[k], axis=0)
            else:
                out["bin1_id"] = np.array([], dtype=bin1_selector.dtype)
                out["bin2_id"] = np.array([], dtype=bin2_selector.dtype)
                out[field] = np.array([], dtype=data_selector.dtype)
                if include_index:
                    out["__index"] = np.array([], dtype=np.int64)

            return out

        return getchunk, loc_pruned_offsets

    def __getitem__(self, key):
        s1, s2 = self._unpack_index(key)
        ispan = self._process_slice(s1, self.shape[0])
        jspan = self._process_slice(s2, self.shape[1])
        getchunk, loc_pruned_offsets = self._make_getchunk(ispan, jspan)
        return RangeQuery(self, ispan, jspan, self.field, getchunk, loc_pruned_offsets)


class RangeQuery(object):
    """
    Executor that fulfills a partitioned 2D range query using a variety of outputs.

    """

    def __init__(self, selector, ispan, jspan, field, getchunk, loc_pruned_offsets):
        self.selector = selector
        self.ispan = ispan
        self.jspan = jspan
        self.field = field
        self.n_chunks = len(loc_pruned_offsets) - 1
        self._locs = loc_pruned_offsets
        self._getchunk = getchunk

    def read_chunk(self, i, include_index=False):
        """Read any chunk of the partitioned query as a dictionary."""
        if not 0 <= i < self.n_chunks:
            raise IndexError(i)
        return self._getchunk(i, include_index)

    def read_chunked(self, include_index=False):
        """Iterator over chunks (as dictionaries)."""
        for i in range(self.n_chunks):
            yield self._getchunk(i, include_index)

    def read(self, include_index=False):
        """Read the complete range query as a dictionary"""
        result = list(self.read_chunked(include_index))
        return {
            k: np.concatenate([d[k] for d in result], axis=0)
            for k in ["bin1_id", "bin2_id", self.field]
        }

    def __repr__(self):
        return (
            "{self.__class__.__name__}"
            '({self.ispan}, {self.jspan}, "{self.field}", ...) '
            "[{n} piece(s)]"
        ).format(self=self, n=self.n_chunks)
