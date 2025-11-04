# Auburn ICPC Reference Sheet 2025

# Algorithms

## Binary Search

```python
from bisect import bisect_left
items = [5,12,1,3,2,11,7,8,9,10,6,4]
items.sort()

def binary_search(items, target):
    i = bisect_left(items, target)
    if i != len(items) and items[i] == target:
        return i
    return -1
print(binary_search(items, 5))
```

## Sorting

Python's default is Timsort. It's a stable sort that finishes in $O(n\cdot \log_2(n))$ time

```python
a = [3,2,4,5,10]
a.sort()
# equivalent
a = sorted(a)

b = [('first', 3, 'a'), ('second', 9, 'b'), ('third', 4, 'd'), ('fourth', 4, 'c'), ('fifth', 4, 'c')]
b.sort(key=lambda x: (x[1], x[2])) 
#[('first', 3, 'a'), ('fourth', 4, 'c'), ('fifth', 4, 'c'), ('third', 4, 'd'), ('second', 9, 'b')]
```

For more complex data, you can sort by a custom key, and even by multiple keys

## Minimum Spanning Trees

### Kruskal's Algorithm

```python
num_vertices, num_edges = 5,5
# src, dest, weight
edges = [
    (1, 2, 1),
    (1, 3, 2),
    (2, 3, 3),
    (2, 4, 4),
    (3, 4, 5)
]

# Use union find to find the minimum spanning tree
parents = [i for i in range(num_vertices+1)]
def find(x):
    if parents[x] == x:
        return x
    parents[x] = find(parents[x])
    return parents[x]
def union(x, y):
    parents[find(x)] = find(y)

total_weight = 0
# take the edges in sorted order (min weight first)
for src, dest, weight in sorted(edges, key=lambda x: x[2]):
    if find(src) != find(dest):
        union(src, dest)
        total_weight += weight
print("Total weight:", total_weight)
```

### Prim's Algorithm
Does not use recursion and will run faster in python

```python
import heapq

num_vertices, num_edges = 5, 5
edges = [
    (1, 2, 1),
    (1, 3, 2),
    (2, 3, 3),
    (2, 4, 4),
    (3, 4, 5)
]

graph = {i: [] for i in range(1, num_vertices + 1)}
for src, dest, weight in edges:
    graph[src].append((dest, weight))
    graph[dest].append((src, weight))

start_node = 1
visited = set()
min_heap = [(0, start_node)]
total_weight = 0

while min_heap:
    weight, current_node = heapq.heappop(min_heap)

    if current_node in visited:
        continue

    visited.add(current_node)
    total_weight += weight

    for neighbor, edge_weight in graph[current_node]:
        if neighbor not in visited:
            heapq.heappush(min_heap, (edge_weight, neighbor))

print(total_weight)
```

## Binary Trees

### Trie

A trie is a great way to hold prefixes for strings or similar data types. If we have a problem that asks a question like "Does this word share a prefix with any previous word," a trie allows us to check this in O(`len(word)`) time.

```python
class Trie:
    def __init__(self, numChars=26):
        self.root = self.TrieNode(numChars)
        self.numChars = numChars
    def insert(self,word):
        n = self.root
        word = word.lower()
        for c in word:
            l = ord(c) - ord('a')
            if n.children[l] == None:
                n.children[l] = self.TrieNode(self.numChars)
            n = n.children[l]
        n.leaf = True
    def search(self,word):
        n = self.root
        word = word.lower()
        for c in word:
            l = ord(c) - ord('a')
            if n.children[l] == None:
                return False
            n = n.children[l]
        return n.leaf
      
  class TrieNode:
      def __init__(self, numChars):
          self.children = [None]*numChars
          self.leaf = False
```

### Heap

Heaps are a way to get the minimum value from a list repeatedly in $O(\log_2(n))$ time. They maintain the "heap invariant," where each item is less than or equal to the items to the left and right in the tree. We normally represent heaps as an array, with item `i`'s left child at `2i` and right at `2i+1`. Use `heapq` when trying to use a heap in python

```python
from heapq import heappush, heappop, heapify

a = [5, 7, 9, 1, 3]
heapify(a)
print(a) #[1, 3, 9, 7, 5]

heappush(a, 4)
print(a) # [1, 3, 4, 7, 5, 9]

print(heappop(a)) # 1
print(a) #[3, 5, 4, 7, 9]
```

## Graph Search

### BFS

Tells us the shortest path between two points. Can also measure that distance and return the path.

```python
from collections import deque
graph = ["O|OOO",
         "O|OOO",
         "O|OOO",
         "OOO|O",
         "OOO|O"]
neighbors = [(1, 0), (0, 1), (-1, 0), (0, -1)]
inbounds = lambda x, y: 0 <= x < len(graph) and 0 <= y < len(graph[0])

queue = deque()
x,y = 0,0
path = [(x,y)]
start = (x, y, path)
queue.append(start)
visited = set()
visited.add((x, y))

end = (4, 4)

while queue:
    x, y, path = queue.popleft()
    if (x, y) == end:
        print("Found path")
        print(path)
        break
    for dx, dy in neighbors:
        new_x = x + dx
        new_y = y + dy
        if inbounds(new_x, new_y) and graph[new_x][new_y] == "O" and (new_x, new_y) not in visited:
            queue.append((x + dx, y + dy, path + [(x + dx, y + dy)]))
            visited.add((x + dx, y + dy))
else:
    print("No path found")
```

### DFS

Tells us if a path exists between two points in a graph. Can also measure the path. 

Two options: Recursive or Iterative. For iterative, use same as above, but replace the queue with `stack = []` and change `queue.popleft()` to `stack.pop()`

```python
graph = ["O|OOO",
         "O|OOO",
         "O|OOO",
         "OOO|O",
         "OOO|O"]

neighbors = [(1, 0), (0, 1), (-1, 0), (0, -1)]
inbounds = lambda x, y: 0 <= x < len(graph) and 0 <= y < len(graph[0])

start = (0, 0, [(0, 0)])
end = (4, 4)

def dfs(x, y, path):
    if (x, y) == end:
        print("Found path")
        print(path)
        return True
    for dx, dy in neighbors:
        new_x = x + dx
        new_y = y + dy
        if inbounds(new_x, new_y) and graph[new_x][new_y] == "O" and (new_x, new_y) not in path:
            if dfs(new_x, new_y, path + [(new_x, new_y)]):
                return True
    return False
print(dfs(0, 0, [(0, 0)]))
```

### Dijkstra

With Dijkstra's algorithm, we compute the shortest way to get between a starting point and any ending point for a (positive) weighted graph. In this example, we keep a weighted adjacency matrix. We use a heap to continuously check the smallest next smallest destination. Whenever we first reach a point, that must be the shortest way to get there, so we can finalize the distance. You could also easily keep track of the path with some modifications.

NOTE: Dijkstra's algorithm does NOT work for negative edge weights. If you have those, use Bellman-Ford. It's slower, but necessary. 

```python
from heapq import heappop, heappush

num_points, start, end = 4, 1, 3
distances = [[0, 1, 3, 14], 
             [2, 0, 4, 22], 
             [3, 10, 0, 7], 
             [13, 8, 2, 0]]
finalized_distances = [-1] * num_points

current_distances = []
heappush(current_distances, (0, start))
while current_distances:
    dist, index = heappop(current_distances)
    if finalized_distances[index] != -1:
        # we've already finalized this node, we should move on
        continue
    # finalize this distance
    finalized_distances[index] = dist
    for i in range(num_points):
        if i == index:
            continue
        if finalized_distances[i] == -1:
            # only try to improve distance if it is not finalized
            heappush(current_distances, (dist + distances[index][i], i))

print(finalized_distances[end])
```

### Bellman-ford

You have to check for negative cycles. Much slower than dijkstra, so be cautious to use it to early

Might be wrong, grabbed online and it looked ok

```python
num_points, start, end = 4, 1, 3
distances = [[0, 1, 3, -2], 
             [2, 0, 4, 22], 
             [3, -2, 0, 7], 
             [13, 8, 2, 0]]

def bellman_ford():
    # initialize distances to infinity
    dist = [float("inf")] * num_points
    # set distance to start to 0
    dist[start] = 0
    # relax edges repeatedly
    for _ in range(num_points):
        for i in range(num_points):
            for j in range(num_points):
                if dist[j] > dist[i] + distances[i][j]:
                    dist[j] = dist[i] + distances[i][j]
    # check for negative cycles
    for i in range(num_points):
        for j in range(num_points):
            if dist[j] > dist[i] + distances[i][j]:
                print("Negative cycle detected")
                return
    print(dist)

bellman_ford()
```

### Topological Sort

You can do topological sorting on a DAG using a modification of recursive DFS.

This WILL NOT work on a graph with cycle

```python
num_vertices, num_edges = 5,5
# means 1 being done allows 2 to be done
# 2 being done allows 3 and 5 to be done
edges = {
    1: [2],
    2: [3,5],
    3: [5],
    4: [3],
    5: []
}

finished_stack = []
visited = [False] * (num_vertices+2)

def dfs(x):
    '''Recursively visit all nodes connected to x'''
    visited[x] = True
    for i in edges[x]:
        if not visited[i]:
            dfs(i)
    # after all nodes connected to x have been visited
    # # add x to the finished stack
    finished_stack.append(x)

# visit all nodes
for i in range(1, num_vertices+1):
    if not visited[i]:
        dfs(i)
# finished_stack in reverse order is the topological order
topo_order = finished_stack[::-1]
print(topo_order)
```

## Longest Path

Finding the longest path on a normal graph is NP-Hard, but on a DAG it's easy.

```python
def longest_path(G):
	# get topological ordering
	V = topsort(G)
	for x in V:
		x.value = max(G.incoming_weight + parent_value)
	x  = max value node
	backtrace its parent
```

This can be applied to Gantt Charts (critical path of a task).

### Finding Eulerian Trials

An Eulerian train is a traversal over a graph where every edge is used exactly once. You quickly find one with Heirholtzer's Algorithm

```python
def eularian_trail_exists(G):
	source,sink = None
	for v in G.nodes:
		if v.out_degree != v.in_degree:
			if v.out_degree == v.in_degree + 1:
				if source is not None:
					source = v
				else:
					return False
		  if v.in_degree == v.out_degree+ 1:
				if sink is not None:
					sink = v
				else:
					return False
	if all([source,sink]) or not any([source,sink]):
		# only ok if we have BOTH or NEITHER
		return True
	return False

def Hierholtzer(G):
	for v in G.nodes:
		v.in_degree = len(G.adjacent(v))
		v.out_degree = len(G.in_adjacent(v))
		source_exists = eularian_trail_exists(G)
		trail = new stack
		if source_exists:
			dfs_stack = []
			dfs_stack.append(source)
			while dfs_stack:
				v = dfs_stack.pop()
				if v.out_degree > 0:
					dfs_stack.append(v)
					dfs_stack.append(G.adjacent(v)[-1])
					G.edges.remove(v, G.adjacent(v)[-1])
					v.out_degree -= 1
				else:
					eularian_trail.append(v)
```

### Union-Find

In order to find all connected components quickly, we can use union-find. This is an important first step for Kruskal's algorithm, and can also be useful for checking how many separate sets exist at the same time.

```python
items = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def find(x):
    if items[x] == x:
        return x
    items[x] = find(items[x])
    return items[x]

def union(x, y):
    items[find(x)] = find(y)

# to combine two sets, union an element from each set
union(1, 5)
print(find(5), find(1)) # 5 5
```

### Checking for Cycle

We can check for cycle using a version of DFS. We keep track of the elements that are currently on the call stack, and we see if we would add the same element again. 

```cpp
num_vertices, num_edges = 5,5
# note the cycle from 1 to 3 to 5 to 1
edges = {
    1: [3],
    2: [],
    3: [5],
    4: [2],
    5: [1]
}

'''Use dfs to detect a cycle in the graph'''
# visited[i] is True if node i has been visited
visited = [False] * (num_vertices+2)
# on_stack[i] is True if node i is on the current dfs stack
on_stack = [False] * (num_vertices+2)
# stack is the current dfs stack
stack = []
# cycle is True if a cycle is detected
cycle = False

def dfs(x):
    '''Recursively visit all nodes connected to x'''
    visited[x] = True
    on_stack[x] = True
    for i in edges[x]:
        if not visited[i]:
            dfs(i)
        elif on_stack[i]:
            # if i is on the current stack, a cycle is detected
            print("CYCLE DETECTED")
            return
    # after all nodes connected to x have been visited
    # # remove x from the stack
    on_stack[x] = False

# visit all nodes
for i in range(1, num_vertices+1):
    if not visited[i]:
        dfs(i)
```

### Strongly Connected Components

An SCC only exists in directed graphs. For any two vertices ($a$ and $b$) in an SCC, we can find a path from $a$ to $b$ AND from $b$ to $a$. This means every vertex in that SCC is reachable from others. 

```python
# kinda transcribed from c++, hopefully correct

# define some negative values for the consts here
adjacency_list: dict[idx, list[idx]]
dfsNumberCounter = 0;
numSCC = 0;
stack = []
dfs_num = collections.defaultdict(lambda: UNVISITED)
dfs_low, visited  = dict(), set()

def tarjanSCC(node):
	dfs_low[node] = dfs_num[node] = dfsNumberCounter
	dfsNumberCounter += 1
	stack.append(node)
	visited.add(node)
	for x in adjacency_list[node]:
		if dfs_num[x] == UNVISITED:
			tarjanSCC(x)
		if x in visited:
			dfs_low[node] = min(dfs_low[node], dfs_low[x])
	if dfs_low[node] == dfs_num[node]:
		numSCC += 1
		while True:
			x = stack.pop()
			visited[x] = False
			if x == node: break

for node in nodes:
	if dfs_num[node] == UNVISITED:
		tarjansSCC(node)
```

## Prefix Sum

If you have a static list of numbers and need something like a sum of all of them, you can precompute a list of cumulative sums and compute them in O(1). Note that this only works when we have an operation like subtraction available. This does not work for the min or max or a range.

```python
a = [1,2,3,4,5,6,7]
prefix_sum = [a[0]]
for i in range(1, len(a)):
    prefix_sum.append(prefix_sum[-1] + a[i])
print(prefix_sum)

def get_sum(left, right):
    if left == 0:
        return prefix_sum[right]
    return prefix_sum[right] - prefix_sum[left-1]
```

## Segment Trees

If a list of numbers changes frequently and we need operations over a range (like sum, min/max, product, gcf/lcm), we can use a segment tree.

```python
a = [8,1,3,9,5,7,10]
tree = [float('inf')] * (4*n)
def build_tree(left, right, c=0):
    if left==right:
        tree[c] = a[left]
        return
    middle = (left+right) // 2
    build_tree(left, middle, c*2+1)
    build_tree(middle+1, right, c*2+2)
    tree[c] = min(tree[2*c+1], tree[2*c+2])
build_tree(0, len(a)-1)

def update_tree(index, new_val, left=0, right=len(a)-1, c=0):
    if left == right == index:
        prev = tree[c]
        tree[c] = new_val
        # print(f"Update {c=} {left=} {right=} from {prev} to {tree[c]}")
    else:
        middle = (left + right) // 2
        if index <= middle:
            update_tree(index, new_val, left, middle, 2*c+1)
        else:
            update_tree(index, new_val, middle+1, right, 2*c+2)
        prev = tree[c]
        tree[c] = min(tree[2*c+1], tree[2*c+2])
        # print(f"Update {c=} {left=} {right=} from {prev} to {tree[c]}")
update_tree(index=4, new_val=1)

def query_tree(query_left, query_right, left=0, right=len(a)-1, c=0):
    print(f"Query {c=} {left=} {right=}")
    if query_left > right or query_right < left:
        return float('inf')
    if query_left <= left and query_right >= right:
        return tree[c]
    middle = (left + right) // 2
    return min(query_tree(query_left, query_right, left, middle, c*2+1), \
               query_tree( query_left, query_right, middle+1, right, c*2+2))
print(a[3:7], query_tree(3,6)) 
```

## String Alignment

We want to know the minimum number of edits to transform one string (A) into another (B).  For character A[i] and B[i], we can either match (they are the same, +2 score), mismatch (replace A[i] with B[i], -1 score), insert a space in A[i] (-1 score), delete a character in A[i] (-1 score). The combination of these actions with the highest score is the best alignment.

```python
A = "ACAATCC"
B = "AGCATGC"
n = len(A)
m = len(B)

table = [[0 for j in range(20)] for i in range(20)] # Needleman Wunsnch's algorithm

for i in range(1, n+1):
    table[i][0] = i * -1
for j in range(1, m+1):
    table[0][j] = j * -1

for i in range(1, n+1):
    for j in range(1, m+1):
        # match = 2 points, mismatch = -1 point
        table[i][j] = table[i - 1][j - 1] + (2 if A[i - 1] == B[j - 1] else -1)
        # insert/delete = -1 point
        table[i][j] = max(table[i][j], table[i - 1][j] - 1); # delete
        table[i][j] = max(table[i][j], table[i][j - 1] - 1); # insert

print("DP table:")
for i in range(0, n+1):
    for j in range(0, m+1):
        print("{:>3}".format(table[i][j]), end='')
    print()

print("Maximum Alignment Score: {}".format(table[n][m]))
```

You can also use this table structure with Dijkstra for the same problem. Set match to 0, and other stuff to +1. Then, find the shortest path from top left to bottom right.

# Math

### Checking for Prime

```python
import math
# O(sqrt(n)) time complexity
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, math.floor(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True
```

### Getting a list of primes

```python
def sieve_of_eratosthenes(n):
    primes = []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return primes
```

### Finding a number's prime factors

```python
def prime_factorization(n, primes_under_sqrt_n):
    prime_factors = []
    for p in primes_under_sqrt_n:
        while n % p == 0:
            prime_factors.append(p)
            n //= p
        if n == 1:
            break
    if n > 1:
        prime_factors.append(n)
    return prime_factors
```

### Prime properties

For an integer N where $N = a^i \times b^j \times ... \times c^k$ 

Number of divisors of integer N:  $(i+1) \times (j+1) \times ... \times (k+1)$

Sum of divisors of integer N: $\frac{a^{i+1} - 1}{a-1} \times \frac{b^{j+1} - 1}{b-1} \times ... \times \frac{c^{k+1} - 1}{c-1}$

### Euler Phi Function

Counts the number of positive integers relatively prime to a number

$\phi(N) = N \times \prod_{p_i}(1 - \frac{1}{p_i})$

```python
def euler_phi(n, primes_under_sqrt_n):
    result = n
    for p in primes_under_sqrt_n:
        if n % p == 0:
            result -= result // p
            # equivalent to result *= (1 - 1 // p)
    if n > 1:
        result -= result // n
    return int(result)

```


### LRU Cache
```python
import functools
@functools.lru_cache(maxsize=None)
```


### Modulo Properties

$(a+b)\%m = ((a\%m) + (b\%m))\%m$ 

$(a-b)\%m = ((a\%m) - (b\%m))\%m$ 

$(a\times b) \%m = ((a\%m) \times (b\%m))\%m$ 

Division is not this way: if you need to do that, you need modular inverse

```python
def gcd(a, b):
    # Recursive Extended Euclidean Algorithm
    if b == 0:
        return a, 1, 0
    g, x_, y_ = gcd(b, a % b)
    x = y_
    y = x_ - y_ * (a//b)
    return g, x, y
def modular_inverse(a, m):
    g, x, _ = gcd(a, m)
    if g == 1:
        return x
    # If gcf != 1, no solution
    return None
```

### Built-in easy stuff

Translate string in any base to integer `int(string, base)` 

Factorial: `math.factorial(number)`

Exponent with modulo: `pow(base, exponent, mod)`

# Fast IO Methods

For python, we use `sys.stdin` for reading quickly line-by-line. You can also `sys.stdin.readline()` if you don't want to iterate over it. Keep in mind that this includes the `\n` character

### Fast Input
```python
import sys
data = list(map(int, sys.stdin.buffer.read().split()))
it = iter(data)
```

### Fast Output
```python
import sys

for line in sys.stdin:
    print(line)

# OR

while True:
    line = sys.stdin.readline()
    if line == "":
        break
    print(line)
```


