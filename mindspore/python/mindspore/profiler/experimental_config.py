# Copyright 2020-2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Experimental config file."""

__all__ = ["_ExperimentalConfig"]

from mindspore.profiler.common.constant import ProfilerLevel, AicoreMetrics


class _ExperimentalConfig:
    r"""
    The purpose of this class is to configure scalable parameters when using profiling for model
    performance data acquisition.

    Args:
        profiler_level (ProfilerLevel, optional): (Ascend only) The level of profiling.
            Default: ``ProfilerLevel.Level0``.

            - ProfilerLevel.LevelNone: This setting takes effect only when mstx is enabled, indicating that no
              operator data is collected on the device side.
            - ProfilerLevel.Level0: Leanest level of profiling data collection, collects information about the elapsed
              time of the computational operators on the NPU and communication large operator information.
            - ProfilerLevel.Level1: Collect more CANN layer AscendCL data and AICore performance metrics and
              communication mini operator information based on Level0.
            - ProfilerLevel.Level2: Collect GE and Runtime information in CANN layer on top of Level1
        aic_metrics (AicoreMetrics, optional): (Ascend only) Types of AICORE performance data collected,
            when using this parameter, `activities` must include ``ProfilerActivity.NPU`` , and the value
            must be a member of AicoreMetrics. When profiler_level is Level0, the default value is
            ``AicoreMetrics.AiCoreNone``; profiler_level is Level1 or Level2, the default value is:
            ``AicoreMetrics.PipeUtilization``. The data items contained in each metric are as follows:

            - AicoreMetrics.AiCoreNone: Does not collect AICORE data.
            - AicoreMetrics.ArithmeticUtilization: ArithmeticUtilization contains mac_fp16/int8_ratio,
              vec_fp32/fp16/int32_ratio, vec_misc_ratio etc.
            - AicoreMetrics.PipeUtilization: PipeUtilization contains vec_ratio, mac_ratio, scalar_ratio,
              mte1/mte2/mte3_ratio, icache_miss_rate etc.
            - AicoreMetrics.Memory: Memory contains ub_read/write_bw, l1_read/write_bw, l2_read/write_bw,
              main_mem_read/write_bw etc.
            - AicoreMetrics.MemoryL0: MemoryL0 contains l0a_read/write_bw, l0b_read/write_bw, l0c_read/write_bw etc.
            - AicoreMetrics.ResourceConflictRatio: ResourceConflictRatio contains vec_bankgroup/bank/resc_cflt_ratio
              etc.
            - AicoreMetrics.MemoryUB: MemoryUB contains ub_read/write_bw_mte, ub_read/write_bw_vector,
              ub\_read/write_bw_scalar etc.
            - AicoreMetrics.L2Cache: L2Cache contains write_cache_hit, write_cache_miss_allocate, r0_read_cache_hit,
              r1_read_cache_hit etc.
            - AicoreMetrics.MemoryAccess: Statistics on storage access bandwidth and storage capacity of main
              storage and L2 cache etc.
        l2_cache (bool, optional): (Ascend only) Whether to collect L2 cache data, collect when True.
            Default: ``False`` . The l2_cache.csv file is generated in the ASCEND_PROFILER_OUTPUT folder.
        mstx (bool, optional): (Ascend only) Whether to collect light weight profiling data, collect when True.
            Default: ``False`` .
        data_simplification (bool, optional): (Ascend only) Whether to remove FRAMEWORK data and other redundant data.
            If set to True, only the profiler deliverables and raw performance data under the PROF_XXX directory are
            kept to save space. Default value: ``True`` .
        export_type (list, optional): (Ascend only) The data type to export.
            The db and text formats can be exported at the same time. The default value is ``None``,
            indicating that data of the text type is exported.

            - ExportType.Text: Export text type data.
            - ExportType.Db: Export db type data.
        mstx_domain_include (list, optional): (Ascend only) Set the set of enabled domain names when the mstx switch
                  is turned on. The name must be of str type. Default value: ``[]``, indicating that this parameter
                  is not used to control the domain. This parameter is mutually exclusive with the mstx_domain_exclude
                  parameter and cannot be set simultaneously. If both are set, only the mstx_domain_include parameter
                  takes effect.
        mstx_domain_exclude (list, optional): (Ascend only) Set the set of domain names that are not enabled when the
                  mstx switch is turned on. The name must be of str type. Default value: ``[]``, indicating that this
                  parameter is not used to control the domain.
        sys_io (bool, optional): (Ascend only) Whether to collect NIC and RoCE data. Default: ``False``.
        sys_interconnection (bool, optional): (Ascend only) Whether to collect system interconnection data, including
            HCCS data, PCIe data, and Stars Chip Trans. Default: ``False``.
        host_sys (list, optional): Collect the data of system call classes on the host side.
            Default: ``[]``, indicating that system class data on the host side is not collected.
            You need to set `start_profile` of :class:`mindspore.profiler.profile` to ``False``. Currently, only
            the **root user** supports collecting DISK or OSRT data, when collecting DISK or
            OSRT data, it is necessary to install the iotop, perf, and ltrace third-party tools in advance.
            For detailed steps, please refer to `Installing Third-party Tools
            <https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/T&ITools/Profiling/atlasprofiling_16_0136.
            html>`_ .
            After the third-party tool is successfully installed, user permissions need to be configured.
            For detailed steps, please refer to `Configure User Permissions
            <https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/T&ITools/Profiling/atlasprofiling_16_0137.
            html>`_ .
            Note that in step 3 of configuring user permissions, the content in the msprof_data_collection.sh
            script needs to be replaced with `msprof_data_collection.sh
            <https://atomgit.com/mindspore/mindspore/blob/master/docs/api/api_python/mindspore/script/
            msprof_data_collection.sh>`_.

            Final deliverables by `MindStudio Insight
            <https://www.hiascend.com/developer/download/community/result?module=pt+sto+cann>`_
            tool visualizes the analysis results.
            For detailed analysis, please refer to `host-side CPU data analysis
            <https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/T&ITools/Profiling/
            atlasprofiling_16_0106.html>`_, `host-side MEM data analysis
            <https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/T&ITools/Profiling/
            atlasprofiling_16_0107.html>`_, `host-side DISK data analysis
            <https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/T&ITools/Profiling/
            atlasprofiling_16_0108.html>`_, `host-side NETWORK data analysis
            <https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/T&ITools/Profiling/
            atlasprofiling_16_0109.html>`_, `host-side OSRT data analysis
            <https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/T&ITools/Profiling/
            atlasprofiling_16_0110.html>`_.

            - HostSystem.CPU: Collect the CPU utilization at the process level.
            - HostSystem.MEM: Collect the memory utilization at the process level.
            - HostSystem.DISK: Collect the disk I/O utilization at the process level.
            - HostSystem.NETWORK: Collect the network I/O utilization at the system level.
            - HostSystem.OSRT: Collect system call stack data at the system level.

    Raises:
        RuntimeError: When the version of CANN does not match the version of MindSpore,
            MindSpore cannot parse the generated ascend_job_id directory structure.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import nn, context
        >>> import mindspore.dataset as ds
        >>> from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics, ExportType
        >>>
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.fc = nn.Dense(2,2)
        ...     def construct(self, x):
        ...         return self.fc(x)
        >>>
        >>> def generator():
        ...     for i in range(2):
        ...         yield np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32)
        >>>
        >>> def train(net):
        ...     optimizer = nn.Momentum(net.trainable_params(), 1, 0.9)
        ...     loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        ...     data = ds.GeneratorDataset(generator, ["data", "label"])
        ...     model = mindspore.train.Model(net, loss, optimizer)
        ...     model.train(1, data)
        >>>
        >>> if __name__ == '__main__':
        ...     # If the device_target is GPU, set the device_target to "GPU"
        ...     context.set_context(mode=mindspore.GRAPH_MODE)
        ...     mindspore.set_device("Ascend")
        ...
        ...     # Init Profiler
        ...     experimental_config = mindspore.profiler._ExperimentalConfig(
        ...                                 profiler_level=ProfilerLevel.Level0,
        ...                                 aic_metrics=AicoreMetrics.AiCoreNone,
        ...                                 l2_cache=False,
        ...                                 mstx=False,
        ...                                 data_simplification=False,
        ...                                 export_type=[ExportType.Text])
        ...     steps = 10
        ...     net = Net()
        ...     # Note that the Profiler should be initialized before model.train
        ...     with mindspore.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        ...                                     schedule=mindspore.profiler.schedule(wait=0, warmup=0, active=1,
        ...                                           repeat=1, skip_first=0),
        ...                                     on_trace_ready=mindspore.profiler.tensorboard_trace_handler("./data"),
        ...                                     profile_memory=False,
        ...                                     experimental_config=experimental_config) as prof:
        ...
        ...         # Train Model
        ...         for step in range(steps):
        ...             train(net)
        ...             prof.step()
    """

    def __init__(self,
                 profiler_level: ProfilerLevel = ProfilerLevel.Level0,
                 aic_metrics: AicoreMetrics = AicoreMetrics.AiCoreNone,
                 l2_cache: bool = False,
                 mstx: bool = False,
                 data_simplification: bool = True,
                 export_type: list = None,
                 mstx_domain_include: list = None,
                 mstx_domain_exclude: list = None,
                 sys_io: bool = False,
                 sys_interconnection: bool = False,
                 host_sys: list = None
                 ):
        self._profiler_level = profiler_level
        self._aic_metrics = aic_metrics
        self._l2_cache = l2_cache
        self._mstx = mstx
        self._data_simplification = data_simplification
        self._export_type = export_type
        self._mstx_domain_include = mstx_domain_include
        self._mstx_domain_exclude = mstx_domain_exclude
        self._sys_io = sys_io
        self._sys_interconnection = sys_interconnection
        self._host_sys = host_sys

    @property
    def profiler_level(self) -> ProfilerLevel:
        """Get profiler_level."""
        return self._profiler_level

    @property
    def aic_metrics(self) -> AicoreMetrics:
        """Get aic_metrics."""
        return self._aic_metrics

    @property
    def l2_cache(self) -> bool:
        """Get l2_cache."""
        return self._l2_cache

    @property
    def mstx(self) -> bool:
        """Get mstx."""
        return self._mstx

    @property
    def data_simplification(self) -> bool:
        """Get data_simplification."""
        return self._data_simplification

    @property
    def export_type(self) -> list:
        """Get export_type."""
        return self._export_type

    @property
    def mstx_domain_include(self) -> list:
        return self._mstx_domain_include

    @property
    def mstx_domain_exclude(self) -> list:
        return self._mstx_domain_exclude

    @property
    def sys_io(self) -> bool:
        return self._sys_io

    @property
    def sys_interconnection(self) -> bool:
        return self._sys_interconnection

    @property
    def host_sys(self) -> list:
        """Get host_sys."""
        return self._host_sys

    # Setters
    @profiler_level.setter
    def profiler_level(self, value: ProfilerLevel):
        self._profiler_level = value

    @aic_metrics.setter
    def aic_metrics(self, value: AicoreMetrics):
        self._aic_metrics = value

    @l2_cache.setter
    def l2_cache(self, value: bool):
        self._l2_cache = value

    @mstx.setter
    def mstx(self, value: bool):
        self._mstx = value

    @data_simplification.setter
    def data_simplification(self, value: bool):
        self._data_simplification = value

    @export_type.setter
    def export_type(self, value: list):
        self._export_type = value

    @mstx_domain_include.setter
    def mstx_domain_include(self, value: list):
        self._mstx_domain_include = value

    @mstx_domain_exclude.setter
    def mstx_domain_exclude(self, value: list):
        self._mstx_domain_exclude = value

    @sys_io.setter
    def sys_io(self, value: bool):
        self._sys_io = value

    @sys_interconnection.setter
    def sys_interconnection(self, value: bool):
        self._sys_interconnection = value

    @host_sys.setter
    def host_sys(self, value: list):
        self._host_sys = value
