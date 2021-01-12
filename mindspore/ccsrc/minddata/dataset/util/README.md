This folder contains miscellaneous utilities used by the dataset code. We will describe a couple important classes in this file.

## Thread Management

This picture summarizes a few important classes that we will cover in the next few sections.

![Thread management](https://images.gitee.com/uploads/images/2020/0601/220111_9b07c8fa_7342120.jpeg "task_manager.JPG")

## Task

A Task object corresponds to an instance of std::future returning from std::async. In general, a user will not create a Task object directly. Most work will go through TaskManager's TaskGroup interface which we will cover later in this document. Here are some important members and functions of Task class.

```cpp
std::function<Status()> fnc_obj_;
```

It is the entry function when the thead is spawned. The function does not take any input and will return a Status object. The returned Status object will be saved in this member

```cpp
Status rc_;
```

To retrieve the executed result from the entry function, call the following function

```cpp
Status Task::GetTaskErrorIfAny();
```

Here is roughly the pseudo code of a lifetime of a Task. Some extra works needed to spawn the thread are omitted for the purpose of simplicity. As mentioned previously, a user never spawn a thread directly using a Task class without using any helper.

```cpp
1 Task tk = Task("A name for this thread", []() -> Status {
2   return Status::OK();
3 });
4 RETURN_IF_NOT_OK(tk.Run());
5 RETURN_IF_NOT_OK(tk.Join();)
6 RETURN_IF_NOT_OK(tk.GetTaskErrorIfAny());
```

In the above example line 1 to 3 we use Task constructor to prepare a thread that we are going to create and what it will be running. We also assign a name to this thread. The name is for eye catcher purpose. The second parameter is the real job for this thread to run.
<br/>Line 4 we spawn the thread. In the above example, the thread will execute the lambda function which does nothing but return a OK Status object.
<br/>Line 5 We wait for the thread to complete
<br/>Line 6 We retrieve the result from running the thread which should be the OK Status object.

Another purpose of Task object is to wrap around the entry function and capture any possible exceptions thrown by running the entry function but not being caught within the entry function.

```cpp
  try {
    rc_ = fnc_obj_();
  } catch (const std::bad_alloc &e) {
    rc_ = Status(StatusCode::kOutOfMemory, __LINE__, __FILE__, e.what());
  } catch (const std::exception &e) {
    rc_ = Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, e.what());
  }
```

Note that

```cpp
Status Task::Run();
```

is not returning the Status of running the entry function func_obj_. It merely indicates if the spawn is successful or not. This function returns immediately.

Another thing to point out that Task::Run() is not designed to re-run the thread repeatedly, say after it has returned. Result will be unexpected if a Task object is re-run.

For the function

```cpp
Status Task::Join(WaitFlag wf = WaitFlag::kBlocking);
```

where

```cpp
enum class WaitFlag : int { kBlocking, kNonBlocking };
```

is also not returning the Status of running the entry function func_obj_ like the function Run(). It can return some other unexpected error while waiting for the thread to return.

This function blocks (kBlocking) by default until the spawned thread returns.

As mentioned previously, use the function GetTaskErrorIfAny() to fetch the result from running the entry function func_obj_.

The non-blocking version (kNonBlocking) of Join allows us to force the thread to return if timed out.

```cpp
while (thrd_.wait_for(std::chrono::seconds(1)) != std::future_status::ready) {
    // Do something if the thread is blocked on a conditional variable
}
```

The main use of this form of Join() is after we have interrupted the thread.

A design alternative is to use

```cpp
std::future<Status>
```

to spawn the thread asynchronously and we can get the result using std::future::get(). But get() can only be called once and it is then more convenient to save the returned result in the rc_member for unlimited number of retrieval. As we shall see later, the value of rc_ will be propagated to high level classes like TaskGroup, master thread.

Currently it is how the thread is defined in Task class

```cpp
std::future<void> thrd_;
```

and spawned by this line of code.

```cpp
thrd_ = std::async(std::launch::async, std::ref(*this));
```

Every thread can access its own Task object using the FindMe() function.

```cpp
Task * TaskManager::FindMe();
```

There are other attributes of Task such as interrupt which we will cover later in this document.

## TaskGroup

The first helper in managing Task objects is TaskGroup. Technically speaking a TaskGroup is a collection of related Tasks. As of this writing, every Task must belong to a TaskGroup. We spawn a thread using the following function

```cpp
Status TaskGroup::CreateAsyncTask(const std::string &my_name, const std::function<Status()> &f, Task **pTask = nullptr);
```

The created Task object is added to the TaskGroup object. In many cases, user do not need to get a reference to the newly created Task object. But the CreateAsyncTask can return one if requested.

There is no other way to add a Task object to a TaskGroup other than by calling TaskGroup::CreateAsyncTask. As a result, no Task object can belong to multiple TaskGroup's by design. Every Task object has a back pointer to the TaskGroup it belongs to :

```cpp
TaskGroup *Task::MyTaskGroup();
```

Task objects in the same TaskGroup will form a linked list with newly created Task object appended to the end of the list.

Globally we support multiple TaskGroups's running concurrently. TaskManager (discussed in the next section) will chain all Task objects from all TaskGroup's in a single LRU linked list.

### HandShaking

As of this writing, the following handshaking logic is required. Suppose a thread T1 create another thread, say T2 by calling TaskGroup::CreateAsyncTask. T1 will block on a WaitPost area until T2 post back signalling T1 can resume.

```cpp
// Entry logic of T2
auto *myTask = TaskManager::FindMe();
myTask->Post();
```

If T2 is going to spawn more threads, say T3 and T4, it is *highly recommended* that T2 wait for T3 and T4 to post before it posts back to T1.

The purpose of the handshake is to provide a way for T2 to synchronize with T1 if necessary.

TaskGroup provides similar functions as Task but at a group level.

```cpp
void TaskGroup::interrupt_all() noexcept;
```

This interrupt all the threads currently running in the TaskGroup. The function returns immediately. We will cover more details on the mechanism of interrupt later in this document.

```cpp
Status TaskGroup::join_all(Task::WaitFlag wf = Task::WaitFlag::kBlocking);
```

This performs Task::Join() on all the threads in the group. This is a blocking call by default.

```cpp
Status TaskGroup::GetTaskErrorIfAny();
```

A TaskGroup does not save records for all the Task::rc_ for all the threads in this group. Only the first error is saved. For example, if thread T1 reports error rc1 and later on T2 reports error rc2, only rc1 is saved in the TaskGroup and rc2 is ignored. TaskGroup::GetTaskErrorIfAny() will return rc1 in this case.

```cpp
int size() const noexcept;
```

This returns the size of the TaskGroup.

## TaskManager

TaskManager is a singleton, meaning there is only one such class object. It is created by another Services singleton object which we will cover it in the later section.

```cpp
TaskManager &TaskManager::GetInstance()
```

provides the method to access the singleton.

TaskManager manages all the TaskGroups and all the Tasks objects ever created.

```cpp
  List<Task> lru_;
  List<Task> free_lst_;
  std::set<TaskGroup *> grp_list_;
```

As mentioned previously, all the Tasks in the same TaskGroup are linked in a linked list local to this TaskGroup. At the TaskManager level, all Task objects from all the TaskGroups are linked in the lru_ list.

When a thread finished its job and returned, its corresponding Task object is saved for reuse in the free_lst_. When a new thread is created, TaskManager will first look into the free_lst_ before allocating memory for the new Task object.

```cpp
  std::shared_ptr<Task> master_;
```

The master thread itself also has a corresponding **fake** Task object in the TaskManager singleton object. But this fake Task is not in any of the List<Task>

### Passing error to the master thread

```cpp
void TaskManager::InterruptGroup(Task &);
void TaskManager::InterruptMaster(const Status &);
Status Status::GetMasterThreadRc();
```

When a thread encounters some unexpected error, it performs the following actions before returning

* It saves the error rc in the TaskGroup it belongs (assuming it is the first error reported in the TaskGroup).
* It interrupts every other threads in the TaskGroup by calling TaskManager::InterruptGroup.
* It interrupts the master thread and copy the error rc to the TaskManager::master_::rc_ by calling TaskManager::InterruptMaster(rc). However, because there can be many TaskGroups running in parallel or back to back, if the TaskManager::master_::rc_ is already set to some error from earlier TaskGroup run but not yet retrieved, the old error code will **not** be overwritten by the new error code.

Master thread can query the result using TaskGroup::GetTaskErrorIfAny or TaskManager::GetMasterThreadRc. The first form is the *preferred* method. For the second form, TaskManager::master_::rc_ will be reset to OK() once retrieved such that future call of TaskManager::InterruptMaster() will populate the error to the master thread again.

### WatchDog

TaskManager will spawn an additional thread with "Watchdog" as name catcher. It executes the following function once startup

```cpp
Status TaskManager::WatchDog() {
  TaskManager::FindMe()->Post();
  errno_t err = sem_wait(&sem_);
  if (err == -1) {
    RETURN_STATUS_UNEXPECTED("Errno = " + std::to_string(errno));
  }
  // We are woken up by control-c and we are going to stop all threads that are running.
  // In addition, we also want to prevent new thread from creating. This can be done
  // easily by calling the parent function.
  RETURN_IF_NOT_OK(ServiceStop());
  return Status::OK();
}
```

Its main purpose is to handle Control-C and stop all the threads from running by interrupting all of them. We will cover more on the function call ServiceStop() when we reach the section about Service class.

WatchDog has its own TaskGroup to follow the protocol but it is not in the set of all the TaskGroup.

## Interrupt

C++ std::thread and std::async do not provide a way to stop a thread. So we implement interrupt mechanism to stop a thread from running and exit.

The initial design can be considered as a polling method. A bit or a flag may be set in some global shared area. The running thread will periodically check this bit/flag. If it is set, interrupt has been sent and the thread will quit. This method has a requirement that even if the thread is waiting on a std::conditional_variable, it can't do an unconditional wait() call. That is, it must do a wait_for() with a time out. Once returned from the wait_for() call, the thread must check if it is woken up due to time out or due to the condition is satisfied.

The cons of this approach is the performance cost and we design a pushing method approach.

To begin with we define an abstract class that describe objects that are interruptible.

```cpp
class IntrpResource { ... };
```

It has two states:

```cpp
 enum class State : int { kRunning, kInterrupted };
```

either it is in the state of running or being interrupted.
There are two virtual functions that any class inherit can override

```cpp
virtual Status Interrupt();
virtual void ResetIntrpState();
```

Interrupt() in the base class change the state of the object to kInterrupted. ResetIntrpState() is doing the opposite to reset the state. Any class that inherits the base class can implement its own Interrupt(), for example, we will later on see how a CondVar class (a wrapper for std::condition_variable) deals with interrupt on its own.

All related IntrpResource can register to a

```cpp
class IntrpService {...}
```

It provides the public method

```cpp
 void InterruptAll() noexcept;
```

which goes through all registered IntrpResource objects and call the corresponding Interrupt().

A IntrpResource is always associated with a TaskGroup:

```cpp
class TaskGroup {
  ...
  std::shared_ptr<IntrpService> intrp_svc_;
  ...
};
```

As of this writing, both push and poll methods are used. There are still a few places (e.g. a busy while loop) where a thread must periodically check for interrupt.

## CondVar

A CondVar class is a wrapper of std::condition_variable

```cpp
  std::condition_variable cv_;
```

and is interruptible :

```cpp
class CondVar : public IntrpResource { ... }
```

It overrides the Interrupt() method with its own

```cpp
void CondVar::Interrupt() {
  IntrpResource::Interrupt();
  cv_.notify_all();
}
```

It provides a Wait() method and is equivalent to std::condition_variable::wait.

```cpp
Status Wait(std::unique_lock<std::mutex> *lck, const std::function<bool()> &pred);
```

The main difference is Wait() is interruptible. Thread returning from Wait must check Status return code if it is being interrupted.

Note that once a CondVar is interrupted, its state remains interrupted until it is reset.

## WaitPost

A WaitPost is an implementation of <a href="https://en.wikipedia.org/wiki/Event_(synchronization_primitive)">Event</a>. In brief, it consists of a boolean state and provides methods to synchronize running threads.

* Wait(). If the boolean state is false, the calling threads will block until the boolean state becomes true or an interrupt has occurred.
* Set(). Change the boolean state to true. All blocking threads will be released.
* Clear(). Reset the boolean state back to false.

WaitPost is implemented on top of CondVar and hence is interruptible, that is, caller of

```cpp
Status Wait();
```

must check the return Status for interrupt.

The initial boolean state is false when a WaitPost object is created. Note that once a Set() call is invoked, the boolean state remains true until it is reset.

## List

A List is the implementation of doubly linked list. It is not thread safe and so user must provide methods to serialize the access to the list.

The main feature of List is it allows an element to be inserted into multiple Lists. Take the Task class as an example. It can be in its TaskGroup list and at the same time linked in the global TaskManager task list. When a Task is done, it will be in the free list.

```cpp
class Task {
  ...
  Node<Task> node;
  Node<Task> group;
  Node<Task> free;
  ...
};
class TaskGroup {
  ...
  List<Task> grp_list_;
  ...
};
class TaskManager {
  ...
  List<Task> lru_;
  List<Task> free_lst_;
  ...
};
```

where Node<T> is defined as

```cpp
template <typename T>
struct Node {
  using value_type = T;
  using pointer = T *;
  pointer prev;
  pointer next;

  Node() {
    prev = nullptr;
    next = nullptr;
  }
};
```

The constructor List class will take Node<> as input so it will follow this Node element to form a doubly linked chain. For example, List<Task> lru_takes Task::node in its constructor while TaskGroup::grp_list_ takes Task::group in its constructor. This way we allow a Task to appear in two distinct linked lists.

## Queue

A Queue is a thread safe solution to producer-consumer problem. Every queue is of finite capacity and its size must be provided to the constructor of the Queue. Few methods are provided

* Add(). It appends an element to queue and will be blocked if the queue is full or an interrupt has occurred.
* EmplaceBack(). Same as an Add() but construct the element in place.
* PopFront(). Remove the first element from the queue and will be blocked if the queue is empty or an interrupt has occurred.

Queue is implemented on top of CondVar class and hence is interruptible. So callers of the above functions must check for Status return code for interrupt.

## Locking

C++11 does not provide any shared lock support. So we implement some simple locking classes for our own benefits.

### SpinLock

It is a simple exclusive lock based on CAS (compared and swap). The caller repeatedly trying (and hence the name spinning) to acquire the lock until successful. It is best used when the critical section is very short.

SpinLock is not interruptible.

There is helper class LockGuard to ensure the lock is released if it is acquired.

### RWLock

It is a simple Read Write Lock where the implementation favors writers. Reader will acquire the lock in S (share) mode while writer will acquire the lock in X (exclusive) mode. X mode is not compatible with S and X. S is compatible with S but not X. In addition, we also provide additional functions

* Upgrade(). Upgrade a S lock to X lock.
* Downgrade(). Downgrade a X lock to S lock.

RWLock is not interruptible.

Like LockGuard helper class, there are helper classes SharedLock and UniqueLock to release the lock when the lock goes out of scope.

## Treap

A Treap is the combination of BST (Binary Search Tree) and a heap. Each key is given a priority. The priority for any non-leaf node is greater than or equal to the priority of its children.

Treap supports the following basic operations

* To search for a given key value. Standard binary search algorithm is applied, ignoring the priorities.
* To insert a new key X into the treap. Heap properties of the tree is maintained by tree rotation.
* To delete a key from a treap. Heap properties of the tree is maintained by tree rotation.

## MemoryPool

A MemoryPool is an abstract class to allow memory blocks to be dynamically allocated from a designated memory region. Any class that implements MemoryPool must provide the following implementations.

```cpp
  // Allocate a block of size n
  virtual Status Allocate(size_t, void **) = 0;

  // Enlarge or shrink a block from oldSz to newSz
  virtual Status Reallocate(void **, size_t old_sz, size_t new_sz) = 0;

  // Free a pointer
  virtual void Deallocate(void *) = 0;
```

There are several implementations of MemoryPool

### Arena

Arena is a fixed size memory region which is allocated up front. Each Allocate() will sub-allocate a block from this region.

Internally free blocks are organized into a Treap where the address of the block is the key and its block size is the priority. So the top of the tree is the biggest free block that can be found. Memory allocation is always fast and at a constant cost. Contiguous free blocks are merged into one single free block. Similar algorithm is used to enlarge a block to avoid memory copy.

The main advantage of Arena is we do not need to free individual memory block and simply free the whole region instead.

### CircularPool

It is still an experimental class. It consists of one single Arena or multiple Arenas. To allocate memory we circle through the Arenas before new Arena is added. It has an assumption that memory is not kept for too long and will be released at some point in the future, and memory allocation strategy is based on this assumption.

## B+ tree

We also provide B+ tree support. Compared to std::map, we provide the following additional features

* Thread safe
* Concurrent insert/update/search support.

As of this writing, no delete support has been implemented yet.

## Service

Many of the internal class inherit from a Service abstract class. A Service class simply speaking it provides service. A Service class consists of four states

```cpp
enum class STATE : int { kStartInProg = 1, kRunning, kStopInProg, kStopped };
```

Any class that inherits from Service class must implement the following two methods.

```cpp
  virtual Status DoServiceStart() = 0;
  virtual Status DoServiceStop() = 0;
```

### Service::ServiceStart()

This function brings up the service and moves the state to kRunning. This function is thread safe. If another thread is bringing up the same service at the same time, only one of them will drive the service up. ServiceStart() will call DoServiceStart() provided by the child class when the state reaches kStartInProg.
An example will be TaskManager which inherits from Service. Its implementation of DoServiceStart will be to spawn off the WatchDog thread.

### Service::ServiceStop()

This function shut down the service and moves the state to kStopped. This function is thread safe. If another thread is bringing down the same service at the same time, only one of them will drive the service down. ServiceStop() will call DoServiceStop() provided by the child class when the states reaches kStopInProg.
As an example, Both TaskManager and TaskGroup during service shutdown will generates interrupts to all the threads.

### State checking

Other important use of Service is to synchronize operations. For example, TaskGroup::CreateAsyncTask will return interrupt error if the current state of TaskGroup is not kRunning. This way we can assure no new thread is allowed to create and added to a TaskGroup while the TaskGroup is going out of scope. Without this state check, we can have Task running without its TaskGroup, and may run into situation the Task is blocked on a CondVar and not returning.

## Services

Services is a singleton and is the first and only one singleton created as a result of calling

```cpp
mindspore::dataset::GlobalInit();
```

The first thing Services singleton do is to create a small 16M circular memory pool. This pool is used by many important classes to ensure basic operation will not fail due to out of memory. The most important example is TaskManager. Each Task memory is allocated from this memory pool.

The next thing Services do is to spawn another singletons in some specific orders. One of the problems of multiple singletons is we have very limited control on the order of creation and destruction of singletons. Sometimes we need to control which singleton to allocate first and which one to deallocate last. One good example is logger. Logger is usually the last one to shutdown.

Services singleton has a requirement on the list of singletons it bring up. They must inherit the Service class. Services singleton will bring each one up by calling the corresponding ServiceStart() function. The destructor of Services singleton will call ServiceStop() to bring down these singletons. TaskManager is a good example. It is invoked by Services singleton.

Services singleton also provide other useful services like

* return the current hostname
* return the current username
* generate a random string

## Path

Path class provides many operating system specific functions to shield the user to write functions for different platforms. As of this writing, the following functions are provided.

```cpp
  bool Exists();
  bool IsDirectory();
  Status CreateDirectory();
  Status CreateDirectories();
  std::string Extension() const;
  std::string ParentPath();
```

Simple "/" operators are also provided to allow folders and/or files to be concatenated and work on all platforms including Windows.
