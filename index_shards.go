package faiss

/*
#include <faiss/c_api/IndexShards_c.h>
#include <faiss/c_api/index_factory_c.h>
#include <faiss/c_api/Index_c.h>
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"runtime"
)

type IndexShards struct {
	ptr       *C.FaissIndexShards
	dimension int
}

func NewIndexShards(dimension int, threaded bool, successive bool) (*IndexShards, error) {
	var ptr *C.FaissIndexShards
	status := C.faiss_IndexShards_new_with_options(&ptr, C.idx_t(dimension), boolToCInt(threaded), boolToCInt(successive))
	if status != 0 {
		return nil, fmt.Errorf("failed to create IndexShards: %d", status)
	}

	idx := &IndexShards{ptr: ptr, dimension: dimension}
	runtime.SetFinalizer(idx, (*IndexShards).Free)
	return idx, nil
}

func (idx *IndexShards) AddShard(shard Index) error {
	if idx.ptr == nil {
		return fmt.Errorf("index is closed")
	}
	if shard == nil || shard.cPtr() == nil {
		return fmt.Errorf("invalid shard index")
	}

	status := C.faiss_IndexShards_add_shard(idx.ptr, shard.cPtr())
	if status != 0 {
		return fmt.Errorf("failed to add shard: %d", status)
	}

	return nil
}

func (idx *IndexShards) Search(x []float32, k int) (distances []float32, labels []int64, err error) {
	if idx.ptr == nil {
		return nil, nil, fmt.Errorf("index is closed")
	}
	if len(x) == 0 {
		return nil, nil, fmt.Errorf("empty query vectors")
	}
	if k <= 0 {
		return nil, nil, fmt.Errorf("invalid n or k parameters")
	}
	n := len(x) / idx.dimension

	distances = make([]float32, n*k)
	labels = make([]int64, n*k)

	status := C.faiss_Index_search(
		idx.ptr,
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.idx_t(k),
		(*C.float)(&distances[0]),
		(*C.idx_t)(&labels[0]),
	)

	if status != 0 {
		return nil, nil, fmt.Errorf("search failed: %d", status)
	}

	return distances, labels, nil
}

// Free releases the memory associated with the index
func (idx *IndexShards) Free() {
	if idx.ptr != nil {
		C.faiss_IndexShards_free(idx.ptr)
		idx.ptr = nil
	}
}

func boolToCInt(b bool) C.int {
	if b {
		return C.int(1)
	}
	return C.int(0)
}
