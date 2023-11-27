# mindspore.multiprocessing

## Overview

mindspore.multiprocessing provides the ability to create multiple processes. Its internal implementation inherits from Python native multiprocessing module and overloads some of the interfaces to ensure that the MindSpore framework works properly when creating multiple processes by fork.

When using mindspore.multiprocessing and creating multiprocesses by fork, the framework internally cleans up and resets threads, locks, and other resources to ensure that the framework functions properly.

- When fork occurs, the parent process waits for the internal thread task to finish executing before the fork, and actively holds the GIL lock in the current thread to avoid the child process after the fork from being unable to hold the GIL lock.
- After fork occurs, the parent process releases the actively held GIL lock.
- After fork occurs, the child process releases the actively held GIL lock, then restores and cleans up resources such as threads within the framework, and resets the backend to `CPU`.

The above process is not triggered when creating a multiprocess without using fork. At this point, the process of creating a multiprocess using mindspore.multiprocessing is exactly the same as that of using Python native multiprocessing module.

Since mindspore.multiprocessing inherits from Python native multiprocessing module, its interface usage is fully compatible with the native module, so users can directly change `import multiprocessing` to `import mindspore. multiprocessing`. Therefore, the description of the native interfaces is not repeated here. For a detailed description and the usage description of the interface, please refer to [multiprocessing](https://docs.python.org/3/library/multiprocessing.html).

## Usage Description

A sample of creating multiple processes using mindspore.multiprocessing is shown below:

``` python
from mindspore import Tensor, ops
import mindspore.multiprocessing as multiprocessing

def child_process():
    y = ops.log(Tensor(2.0))
    print("ops.log(Tensor(2.0))=", y)

if __name__ == '__main__':
    p = multiprocessing.Process(target=child_process)
    p.start()
    p.join()
```

The result is shown below:

``` log
ops.log(Tensor(2.0))= 0.6931472
```

A sample of creating a process pool using mindspore.multiprocessing is shown below:

``` python
from mindspore import Tensor, ops
import mindspore.multiprocessing as multiprocessing

def child_process(x):
    return ops.log(x)

if __name__ == '__main__':
    with multiprocessing.Pool(processes=2) as pool:
        inputs = [Tensor(2.0), Tensor(2.0)]
        outputs = pool.map(child_process, inputs)
        print("ops.log(Tensor(2.0))=", outputs)
```

The result is shown below:

``` log
ops.log(Tensor(2.0))= [Tensor(shape=[], dtype=Float32, value= 0.693147), Tensor(shape=[], dtype=Float32, value= 0.693147)]
```

When the user creates a multiprocess by fork after executing a computing task, if the module used is not mindspore.multiprocessing, there may be a problem that the child process is stuck due to the loss of the frame thread. Changing to use the fork of the mindspore.multiprocessing module to create multiprocesses can solve this problem.

> The fork of creating child processes is only supported on POSIX systems (e.g. Linux and macOS), but not on Windows.

``` python
from mindspore import Tensor, ops
# Child process may be stuck when using 'import multiprocessing' .
import mindspore.multiprocessing as multiprocessing

def child_process(q):
    y = ops.log(Tensor(2.0))
    q.put(y)
    return

if __name__ == '__main__':
    multiprocessing.set_start_method("fork", force=True)
    print("parent process:ops.log(Tensor(2.0))=", ops.log(Tensor(2.0)))
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=child_process, args=(q,))
    p.start()
    print("child process:ops.log(Tensor(2.0))=", q.get())
    p.join()
```

The result is shown below:

``` log
parent process:ops.log(Tensor(2.0))= 0.6931472
child process:ops.log(Tensor(2.0))= 0.6931472
```

When the backend is `Ascend`, the process has exclusive access to the card resources. If a user creates a child process after performing a computation task, and the child process uses the resources of the parent process to perform another computation task, it may report an error due to resource inaccessibility. After modifying the child process creation method to fork, the framework will reset the backend of the child process to `CPU` to avoid resource conflicts.

``` python
from mindspore import Tensor, ops, context
import mindspore.multiprocessing as multiprocessing

def child_process(q):
    y = ops.log(Tensor(2.0))
    q.put(y)
    return

if __name__ == '__main__':
    context.set_context(device_target="Ascend")
    # Child process may not be able to acquire resources when start_method is set to 'spawn' .
    multiprocessing.set_start_method("fork", force=True)
    print("parent process:ops.log(Tensor(2.0))=", ops.log(Tensor(2.0)))
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=child_process, args=(q,))
    p.start()
    print("child process:ops.log(Tensor(2.0))=", q.get())
    p.join()
```

The result is shown below:

``` log
parent process:ops.log(Tensor(2.0))= 0.6931472
child process:ops.log(Tensor(2.0))= 0.6931472
```

The action of creating a multiprocess can be placed in front of executing any computational tasks an the backend is modified manually in a child process.

``` python
from mindspore import Tensor, ops, context
import mindspore.multiprocessing as multiprocessing

def child_process(q):
    # Child process may not be able to acquire resources when using the same resources as parent process.
    context.set_context(device_target="CPU")
    y = ops.log(Tensor(2.0))
    q.put(y)
    return

if __name__ == '__main__':
    context.set_context(device_target="Ascend")
    multiprocessing.set_start_method("spawn", force=True)
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=child_process, args=(q,))
    p.start()
    print("child process:ops.log(Tensor(2.0))=", q.get())
    p.join()
    # Child process may not be able to acquire resources when this compute task is executed before the child process is created.
    print("parent process:ops.log(Tensor(2.0))=", ops.log(Tensor(2.0)))
```

The result is shown below:

``` log
child process:ops.log(Tensor(2.0))= 0.6931472
parent process:ops.log(Tensor(2.0))= 0.6931472
```
