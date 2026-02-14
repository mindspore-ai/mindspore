mindspore.runtime.set_cpu_affinity
===================================

.. py:function:: mindspore.runtime.set_cpu_affinity(enable_affinity, affinity_cpu_list=None, module_to_cpu_dict=None, bind_file=None)

    使能线程级绑核功能，给MindSpore的主要模块（主线程、pynative、runtime、minddata）分配特定的CPU核，防止MindSpore线程抢占CPU导致性能不稳定的情况。

    .. note::
        - 提供灵活的绑核配置：

          1. 不指定 `affinity_cpu_list` 时，依据环境上可用的CPU、NUMA节点和设备资源，自动化生成本进程亲和的CPU范围。
          2. 指定 `affinity_cpu_list` 时，依据传入的 `affinity_cpu_list` 手动指定本进程的CPU范围。
          3. 不指定 `module_to_cpu_dict` 时，默认给 ``"main"`` 模块分配CPU范围的核。
          4. 指定 `module_to_cpu_dict` 时，依据传入的 `module_to_cpu_dict` 手动指定各个模块绑定的CPU范围的索引。
          5. 指定 `bind_file`时，优先依据传入的 JSON 文件指定各个模块绑定的CPU范围。
        - 自动化生成绑核策略场景会调用系统命令去获取环境上的CPU、NUMA节点和设备资源，由于环境差异，一些命令无法成功执行；按照环境上可获取资源，生成的自动化绑核策略会有所差异：

          1. `cat /sys/fs/cgroup/cpuset/cpuset.cpus`，获取环境上可用的CPU资源；若执行该命令失败，绑核功能不会生效。
          2. `npu-smi info -m`，获取环境上可用的NPU资源；若执行该命令失败，仅根据可用CPU资源去生成绑核策略，不考虑设备亲和性。
          3. `npu-smi info -t board -i {NPU_ID} -c {CHIP_ID}`，根据设备逻辑ID去获取NPU的详细信息；若执行该命令失败，仅根据可用CPU资源去生成绑核策略，不考虑设备亲和性。
          4. `lspci -s {PCIe_No} -vvv`，获取环境上设备的硬件信息；若执行该命令失败，仅根据可用CPU资源去生成绑核策略，不考虑设备亲和性。
          5. `lscpu`，获取环境上CPU与NUMA节点的信息；若执行该命令失败，仅根据可用CPU资源去生成绑核策略，不考虑设备亲和性。

    参数：
        - **enable_affinity** (bool) - 开关线程级绑核功能。
        - **affinity_cpu_list** (list，可选) - 自定义指定本进程的绑核CPU范围。传入列表需要为 ``["cpuidX-cpuidY"]`` 格式，例如 ``["0-3", "8-11"]``。默认值： ``None``，即使用依据环境自动化生成的绑核策略。允许传入空列表 ``[]``，与 ``None`` 默认行为一致。
        - **module_to_cpu_dict** (dict，可选) - 自定义指定的绑核策略。传入字典的key需要为模块名称字符串，目前支持传入 ``"main"`` 、 ``"runtime"`` 、 ``"pynative"`` 、 ``"minddata"``；value需要为包含 ``int`` 元素的列表，表示绑核CPU范围中的索引，例如 ``{"main": [0,1], "minddata": [6,7]}``。默认值： ``None``，即默认给模块 ``"main"`` 绑核。允许传入空字典 ``{}``，与 ``None`` 默认行为一致。
        - **bind_file** (str，可选) - 统一绑定JSON文件的路径。设置该参数后，当前设备的 ``affinity_cpu_list`` 和 ``module_to_cpu_dict`` 会从该文件中读取并填充，手动配置项将被忽略。绑定JSON文件示例如下：

          .. code-block::
              {
                  "bind_config": {"bind_cpu_mode": "cpu", "bind_memory_mode": "numa", "actor_thread_fix_bind": true},
                  "bind_cpu": {
                  "scheduler": {"main": "40-45"},
                  "device0": {"main": "0-4", "runtime": "5-9", "pynative": "10-14", "minddata": "15-19"},
                  "device1": {"main": "20-24", "runtime": "25-29", "pynative": "30-34", "minddata": "35-39"}
                  },
                  "bind_memory": {"device0": 0, "device1": 1, "scheduler": 2}
              }

          更详细的配置说明，请参考 `msrun启动 <https://www.mindspore.cn/tutorials/zh-CN/master/parallel/msrun_launcher.html>`_ 。

    异常：
        - **TypeError** - 参数 `enable_affinity` 不是bool。
        - **TypeError** - 参数 `affinity_cpu_list` 既不是列表，也不是 ``None``。
        - **TypeError** - 参数 `affinity_cpu_list` 中的元素不是字符串。
        - **ValueError** - 参数 `affinity_cpu_list` 中的元素不符合 ``["cpuidX-cpuidY"]`` 格式。
        - **TypeError** - 参数 `module_to_cpu_dict` 既不是字典，也不是 ``None``。
        - **TypeError** - 参数 `module_to_cpu_dict` 中的key不是字符串。
        - **TypeError** - 参数 `module_to_cpu_dict` 中的value不是列表。
        - **ValueError** - 参数 `module_to_cpu_dict` 中value的元素不是非负整数。
        - **ValueError** - 输入的 `bind_file` 文件路径与 `msrun --bind_numa` 参数指定的路径不一致。
        - **ValueError** - 输入的 `bind_file` 文件的内容哈希值与 `msrun --bind_numa` 参数指定的路径对应的哈希值不匹配。
        - **RuntimeError** - 自定义指定绑核策略场景，分配给某个设备的CPU在环境中不可用。
        - **RuntimeError** - 重复调用了 :func:`mindspore.runtime.set_cpu_affinity` 接口。
