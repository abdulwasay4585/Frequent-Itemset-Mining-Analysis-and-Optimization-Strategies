import time
import numpy as np
from collections import defaultdict
import multiprocessing as mp
from functools import partial

class FIMBase:
    def __init__(self, transactions, min_sup):
        self.transactions = transactions
        self.num_transactions = len(transactions)
        self.min_sup_count = int(min_sup * self.num_transactions)
        self.frequent_itemsets = {}

    def get_frequent_1_itemsets(self):
        counts = defaultdict(int)
        for transaction in self.transactions:
            for item in transaction:
                counts[item] += 1
        
        frequent_1 = {tuple([item]): count for item, count in counts.items() if count >= self.min_sup_count}
        return frequent_1

class Apriori(FIMBase):
    def run(self):
        start_time = time.time()
        L1 = self.get_frequent_1_itemsets()
        self.frequent_itemsets[1] = L1
        
        k = 2
        while True:
            candidates = self.generate_candidates(list(self.frequent_itemsets[k-1].keys()), k)
            if not candidates:
                break
            
            frequent_k = self.count_support(candidates)
            if not frequent_k:
                break
            
            self.frequent_itemsets[k] = frequent_k
            k += 1
        
        return self.frequent_itemsets, time.time() - start_time

    def generate_candidates(self, prev_frequent, k):
        candidates = []
        n = len(prev_frequent)
        for i in range(n):
            for j in range(i + 1, n):
                l1 = sorted(prev_frequent[i])
                l2 = sorted(prev_frequent[j])
                if l1[:-1] == l2[:-1]:
                    candidate = tuple(sorted(list(set(l1) | set(l2))))
                    if self.should_prune(candidate, prev_frequent):
                        continue
                    candidates.append(candidate)
        return candidates

    def should_prune(self, candidate, prev_frequent):
        from itertools import combinations
        subsets = combinations(candidate, len(candidate) - 1)
        for subset in subsets:
            if tuple(sorted(subset)) not in prev_frequent:
                return True
        return False

    def count_support(self, candidates):
        counts = {c: 0 for c in candidates}
        for transaction in self.transactions:
            transaction_set = set(transaction)
            for candidate in candidates:
                if set(candidate).issubset(transaction_set):
                    counts[candidate] += 1
        
        return {c: count for c, count in counts.items() if count >= self.min_sup_count}

class NS_FIM(FIMBase):
    """
    SOTA 2022 inspired: Non-Sparse Frequent Itemset Mining.
    Uses Vertical Bitset representation for fast intersection.
    """
    def __init__(self, transactions, min_sup):
        super().__init__(transactions, min_sup)
        self.item_bitsets = {}
        self.all_items = sorted(list(set([item for t in transactions for item in t])))
        self.num_items = len(self.all_items)
        
        # Build bitsets for each item
        for item in self.all_items:
            bitset = np.zeros(self.num_transactions, dtype=bool)
            for i, transaction in enumerate(self.transactions):
                if item in transaction:
                    bitset[i] = True
            self.item_bitsets[item] = bitset

    def run(self):
        start_time = time.time()
        L1_items = []
        L1 = {}
        for item in self.all_items:
            sup = np.sum(self.item_bitsets[item])
            if sup >= self.min_sup_count:
                L1[tuple([item])] = sup
                L1_items.append(item)
        
        self.frequent_itemsets[1] = L1
        self.mine_recursive(tuple(), L1_items, 1)
        
        return self.frequent_itemsets, time.time() - start_time

    def mine_recursive(self, prefix, items, k):
        for i, item in enumerate(items):
            new_itemset = prefix + tuple([item])
            
            # Intersection of bitsets
            if not prefix:
                current_bitset = self.item_bitsets[item]
            else:
                # We can cache bitsets of prefixes, but for simplicity:
                current_bitset = self.get_bitset(new_itemset)
            
            support = np.sum(current_bitset)
            if support >= self.min_sup_count:
                if k not in self.frequent_itemsets:
                    self.frequent_itemsets[k] = {}
                self.frequent_itemsets[k][new_itemset] = support
                
                # Further items to combine
                suffix_items = items[i+1:]
                if suffix_items:
                    self.mine_recursive(new_itemset, suffix_items, k + 1)

    def get_bitset(self, itemset):
        res = self.item_bitsets[itemset[0]].copy()
        for i in range(1, len(itemset)):
            res &= self.item_bitsets[itemset[i]]
        return res

class OptimizedApriori(Apriori):
    def __init__(self, transactions, min_sup, parallel=False):
        super().__init__(transactions, min_sup)
        self.parallel = parallel
        # Precompute bitsets for optimization
        self.item_bitsets = {}
        all_items = sorted(list(set([item for t in transactions for item in t])))
        for item in all_items:
            bitset = np.zeros(self.num_transactions, dtype=bool)
            for i, transaction in enumerate(transactions):
                if item in transaction:
                    bitset[i] = True
            self.item_bitsets[item] = bitset

    def count_support(self, candidates):
        if not self.parallel:
            return self.count_support_bitset(candidates)
        else:
            return self.count_support_parallel(candidates)

    def count_support_bitset(self, candidates):
        frequent_k = {}
        for candidate in candidates:
            bitset = self.item_bitsets[candidate[0]].copy()
            for i in range(1, len(candidate)):
                bitset &= self.item_bitsets[candidate[i]]
            
            sup = np.sum(bitset)
            if sup >= self.min_sup_count:
                frequent_k[candidate] = sup
        return frequent_k

    def count_support_parallel(self, candidates):
        # Implementation of parallel support counting
        num_cores = mp.cpu_count()
        chunk_size = max(1, len(candidates) // num_cores)
        chunks = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]
        
        with mp.Pool(num_cores) as pool:
            results = pool.map(partial(self._count_chunk, self.item_bitsets, self.min_sup_count), chunks)
        
        frequent_k = {}
        for res in results:
            frequent_k.update(res)
        return frequent_k

    @staticmethod
    def _count_chunk(item_bitsets, min_sup_count, chunk):
        frequent_chunk = {}
        for candidate in chunk:
            bitset = item_bitsets[candidate[0]].copy()
            for i in range(1, len(candidate)):
                bitset &= item_bitsets[candidate[i]]
            sup = np.sum(bitset)
            if sup >= min_sup_count:
                frequent_chunk[candidate] = sup
        return frequent_chunk
