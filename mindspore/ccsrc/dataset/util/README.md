# Event
The header file WaitPost.h contains the implementation of an event which is a type of synchronization mechanism that is used to indicate to waiting processes when a particular condition has become true.

An event is created with initial state set to false. It provides the following operations:
* `wait` - causes the suspension of the executing process until the state of the event is set to true. If the state is already set to true has no effect.
* `set` - sets the event's state to true, releasing all waiting processes.
* `clear` - sets the event's state to false.

# Counting Semaphore
The header file Semaphore.h contains the implementation of counting semaphore. Conceptually, a semaphore is a nonnegative integer count. Semaphores are typically used to coordinate access to resources, with the semaphore count initialized to the number of free resources. Threads then atomically increment the count when resources are added and atomically decrement the count when resources are removed.

When the semaphore count becomes zero, indicating that no more resources are present, threads trying to decrement the semaphore block wait until the count becomes greater than zero.

Two operations are provided 
* `P`(). Decrement the semaphore count. If the count is 0, the current thread is blocked.
* `V`(). Increment the semaphore count. Wake up one of the threads that are currently blocked. Note that the current implementation wakes up one of the blocked threads instead of waking up all of them.

# List
It is a doubly linked structure used solely by Buffer Manager. List can used for general purpose. The reason we use a home grown linked list because Buffer Manager manages several linked lists and an element can simultaneously in more than one list. Using STL C++ container is not as efficient as the home grown linked list.

# Consumer/Producer Queue
The header file Queue.h contains a generic implementation of producer/consumer queue. The problem describes two processes, the producer and the consumer, who share a common, fixed-size buffer used as a queue. The producer's job is to generate data, put it into the buffer, and start again. At the same time, the consumer is consuming the data (i.e., removing it from the buffer), one piece at a time. 

It has the following template signature
```
        template<typename T, int SIZE> 
        class Queue {

```
_SIZE_ is the capacity of the queue.
_T_ is the object class that represents the data that are produced and consumed by the producer and consumer respectively.

Initially the Queue is empty and all consumers are blocked.

The implementation of Queue is based on counting semaphore above.

The following operations are provided
* void `push_back`(const T&) used by producer to add data to the queue.
* T `pop_front`() used by consumer to retrieve the data from the queue.


# Memory Pool
Two different kinds of memory pools are provided. While they behave differently, they have identical interfaces
* void * `allocate`(size_t reqSize). It allocates memory from the pool where reqSize is the size of memory requested
* void `deallocate`(void *p) returns the memory previously acquired by allocate pointed to by p back to the memory pool
* void `Reallocate`(void **pp, size_t oldSize, size_t newSize). Enlarge or shrink the memory acquired previously by allocate to the new size. The old pointer is passed in and a new pointer (or maybe the same ond) is returned.

C++ operator **new** and **delete** are also overloaded to make use of the customized memory pools.

Both functions allocate and deallocate can throw `std::bad_alloc` if running out of memory from the arena. It is user's responsibility to catch the out of memory exception.

An allocator header file Allocator.h is created to provided additional support to hook into the C++ STL container such as vector or map to allocate memory from the customized memory pools.

## BuddyArena
The first kind of memory pool is BuddyArena. The corresponding header file is BuddyArena.h.

BuddyArena is a general purpose arena and the constructor takes K (in unit of MB) as input. The default value is 4096 which is 4G if no value is given to the constructor.

BuddyArena is implemented based on Buddy System.

## CircularPool
The second kind of memory pool is CircularPool. The corresponding header file is CircularPool.h.

CircularPool is built upon multiple BuddyArena. Initially there is one BuddyArena. More BuddyArena are gradually added to the memory pool as needed until it reaches the specified maximum capacity. There is no guarantee the newly added BuddyArena is contiguous. Maximum size of allocated block in CircularPool is determined by the maximum block allowed by a BuddyArena. By default the maximum capacity is 32G and each BuddyArena is 4G.. The constructor takes unit of GB as input.

There are one important assumption of this kind of memory pool
* Allocated memory is not kept for the whole duration of the memory pool and will be released soon.

User allocates memory from the _logical_ end of the pool while allocated memory will be returned to the _logical_ head of the pool. When a new BuddyArena is added to the pool, it will become the new logical end. When a BuddyArena becomes full, the next BuddyArena (in a round robin fashion) will become the new tail.



