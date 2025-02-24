package main

import (
	"fmt"
	"github.com/DataIntelligenceCrew/go-faiss"
	"log"
	"math/rand"
)

func main() {
	d := 64     // dimension
	nb := 50000 // database size
	// Create a new IndexShards with dimension 64
	// threaded=true for parallel processing
	// successive=false for distributed sharding
	shards, err := faiss.NewIndexShards(d, true, false)
	if err != nil {
		log.Fatalf("Failed to create IndexShards: %v", err)
	}
	defer shards.Free()

	// Create two flat index shards
	shard1, err := faiss.NewIndexFlatL2(d)
	if err != nil {
		log.Fatalf("Failed to create shard1: %v", err)
	}
	defer shard1.Delete()
	xb1 := make([]float32, d*nb)
	for i := 0; i < nb; i++ {
		for j := 0; j < d; j++ {
			xb1[i*d+j] = rand.Float32()
		}
		xb1[i*d] += float32(i) / 1000
	}
	fmt.Println("IsTrained() =", shard1.IsTrained())
	shard1.Add(xb1)
	fmt.Println("Ntotal() =", shard1.Ntotal())

	shard2, err := faiss.NewIndexFlatL2(64)
	if err != nil {
		log.Fatalf("Failed to create shard2: %v", err)
	}
	defer shard2.Delete()

	xb2 := make([]float32, d*nb)
	for i := 0; i < nb; i++ {
		for j := 0; j < d; j++ {
			xb2[i*d+j] = rand.Float32()
		}
		xb2[i*d] += float32(i) / 1000
	}
	fmt.Println("IsTrained() =", shard2.IsTrained())
	shard2.Add(xb2)
	fmt.Println("Ntotal() =", shard2.Ntotal())

	// Add shards to IndexShards
	if err := shards.AddShard(shard1); err != nil {
		log.Fatalf("Failed to add shard1: %v", err)
	}
	if err := shards.AddShard(shard2); err != nil {
		log.Fatalf("Failed to add shard2: %v", err)
	}

	nq := 1000 // number of queries
	xq := make([]float32, d*nq)
	for i := 0; i < nq; i++ {
		for j := 0; j < d; j++ {
			xq[i*d+j] = rand.Float32()
		}
		xq[i*d] += float32(i) / 1000
	}
	// Search for nearest neighbors
	k := 4 // number of nearest neighbors to retrieve
	distances, labels, err := shards.Search(xq, k)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	// Print results
	fmt.Println("Search Results:")
	for i := 0; i < k; i++ {
		fmt.Printf("Neighbor %d: Distance=%f, Label=%d\n",
			i, distances[i], labels[i])
	}
}
