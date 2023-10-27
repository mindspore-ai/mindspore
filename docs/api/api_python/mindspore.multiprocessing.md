# mindspore.multiprocessing

## 介绍

mindspore.multiprocessing提供了创建多进程的能力，其内部实现继承自Python原生的multiprocessing模块，并通过对部分接口进行重载，确保以fork方式创建多进程时MindSpore框架的正常使用。

当使用mindspore.multiprocessing并以fork方式创建多进程时，框架内部会对线程、锁等资源进行清理和重置，以保障框架功能正常。

- 在fork发生时，父进程在fork前会先等待内部线程任务执行完毕，并在当前线程主动持有GIL锁，以避免fork后的子进程无法持有GIL锁。
- 在fork发生后，父进程会释放主动持有的GIL锁。
- 在fork发生后，子进程会释放主动持有的GIL锁，然后对框架内部的线程等资源进行恢复和清理，并将后端重置为 `CPU` 。

当不使用fork方式创建多进程时，上述流程不会被触发。此时使用mindspore.multiprocessing创建多进程的执行流程和使用Python原生的multiprocessing模块的流程完全一致。

由于mindspore.multiprocessing继承自Python原生的multiprocessing模块，其接口用法和原生模块完全兼容，用户可以直接把 `import multiprocessing` 修改为 `import mindspore.multiprocessing` 。因此，此处不对原生接口进行重复描述。接口的详细介绍和使用方式请参考[multiprocessing](https://docs.python.org/3/library/multiprocessing.html)。

## 使用说明

一个使用mindspore.multiprocessing创建多进程的样例如下：

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

运行结果如下：

``` log
ops.log(Tensor(2.0))= 0.6931472
```

一个使用mindspore.multiprocessing创建进程池的样例如下：

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

运行结果如下：

``` log
ops.log(Tensor(2.0))= [Tensor(shape=[], dtype=Float32, value= 0.693147), Tensor(shape=[], dtype=Float32, value= 0.693147)]
```

当用户在执行计算任务后，再使用fork方式创建多进程时，若使用的模块不是mindspore.multiprocessing，可能会出现线程丢失导致的子进程卡住的问题。

> 仅在POSIX系统下（如Linux和macOS）支持fork方式创建子进程，Windows下不支持该方式

``` python
from mindspore import Tensor, ops
import multiprocessing # Using native multiprocessing module

def child_process(q):
    y = ops.log(Tensor(2.0)) # Child process will be stuck here
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

改为使用mindspore.multiprocessing模块的fork方式创建多进程，可以解决该问题：

``` python
from mindspore import Tensor, ops
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

运行结果如下：

``` log
parent process:ops.log(Tensor(2.0))= 0.6931472
child process:ops.log(Tensor(2.0))= 0.6931472
```

当后端为 `Ascend` 时，进程会独占卡资源。若用户在执行计算任务后创建子进程，且子进程使用了父进程的资源执行另一个计算任务，可能会因资源无法访问而报错。

``` python
from mindspore import Tensor, ops, context
import mindspore.multiprocessing as multiprocessing

def child_process(q):
    y = ops.log(Tensor(2.0)) # Child process will fail to acquire resources.
    q.put(y)
    return

if __name__ == '__main__':
    context.set_context(device_target="Ascend")
    multiprocessing.set_start_method("spawn", force=True)
    print("parent process:ops.log(Tensor(2.0))=", ops.log(Tensor(2.0)))
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=child_process, args=(q,))
    p.start()
    print("child process:ops.log(Tensor(2.0))=", q.get())
    p.join()
```

修改子进程创建方式为fork后，框架会将子进程的后端重置为 `CPU` ，以避免资源冲突。

``` python
from mindspore import Tensor, ops, context
import mindspore.multiprocessing as multiprocessing

def child_process(q):
    y = ops.log(Tensor(2.0))
    q.put(y)
    return

if __name__ == '__main__':
    context.set_context(device_target="Ascend")
    multiprocessing.set_start_method("fork", force=True)
    print("parent process:ops.log(Tensor(2.0))=", ops.log(Tensor(2.0)))
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=child_process, args=(q,))
    p.start()
    print("child process:ops.log(Tensor(2.0))=", q.get())
    p.join()
```

运行结果如下：

``` log
parent process:ops.log(Tensor(2.0))= 0.6931472
child process:ops.log(Tensor(2.0))= 0.6931472
```

也可以将创建多进程的动作放在执行任何计算任务的前面，并在子进程里手动修改后端。

``` python
from mindspore import Tensor, ops, context
import mindspore.multiprocessing as multiprocessing

def child_process(q):
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
    print("parent process:ops.log(Tensor(2.0))=", ops.log(Tensor(2.0)))
```

运行结果如下：

``` log
child process:ops.log(Tensor(2.0))= 0.6931472
parent process:ops.log(Tensor(2.0))= 0.6931472
```
