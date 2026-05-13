# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Cell_wrapper."""
from __future__ import absolute_import
from __future__ import division

__all__ = ['PipelineCell', 'Pipeline', 'MicroBatchInterleaved', 'GradAccumulation']

from mindspore import nn
from mindspore.ops import operations as P
from mindspore.nn.cell import Cell
from mindspore.nn.wrap.cell_wrapper import _MicroBatch
from mindspore import log as logger


class PipelineCell(Cell):
    """
    Slice MiniBatch into finer-grained MicroBatch for use in pipeline-parallel training,
    and specify the segment info.

    Note:
        micro_size must be greater or equal to pipeline stages.

    Args:
        network (Cell): The target network to wrap.
        micro_size (int): MicroBatch size.
        stage_config (dict, optional): The stage configuration for each cell's execution in pipeline parallel.
        segment_config (dict, optional): The segment configuration for each cell's execution in pipeline parallel.
            Default ``None``.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore.nn as nn
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://atomgit.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> net = nn.PipelineCell(net, 4, stage_config={"cell_name_0": 0, "cell_name_1": 1})
    """
    def __init__(self, network, micro_size, stage_config=None, segment_config=None):
        super().__init__(auto_prefix=False)
        self.network = network
        self.micro_inputs = nn.CellList()
        self.micro_size = micro_size
        self.add_list = []
        if not isinstance(network, Cell):
            raise TypeError("For 'PipelineCell', the argument 'network' must cell type, "
                            "but got the type : {}.".format(type(network)))
        if not isinstance(micro_size, int):
            raise TypeError("For 'PipelineCell', the argument 'micro_size' must be integer, "
                            "but got the type : {}.".format(type(micro_size)))
        if micro_size <= 0:
            raise ValueError("For 'PipelineCell', the argument 'micro_size' must be large than 0, "
                             "but got {}.".format(micro_size))
        for i in range(micro_size):
            micro_input = _MicroBatch(micro_size)
            self.micro_inputs.append(micro_input)
            self.add = P.Add().add_prim_attr("pipeline_end", i)
            self.add_list.append(self.add)
        self._get_attr_from_cell(network)

        # prase stage_config
        config_dict = {}
        if stage_config is not None:
            for cell_name, stage_num in stage_config.items():
                config_cell_name = cell_name
                config_stage_num = stage_num
                config_dict[config_cell_name] = config_stage_num

        # set cell.stage_config
            for cell_name, cell in self.network.cells_and_names():
                for config_cell_name, config_stage_num in config_dict.copy().items():
                    if not cell_name or not config_cell_name:
                        continue
                    if cell_name == config_cell_name:
                        setattr(cell, "pipeline_stage", config_stage_num)
                        del config_dict[config_cell_name]

            for config_cell_name, config_stage_num in config_dict.copy().items():
                if str(network) == config_cell_name:
                    setattr(network, "pipeline_stage", config_stage_num)
                    del config_dict[config_cell_name]

            # if there are any config elements left, print them
            if config_dict:
                for config_cell_name, config_stage_num in config_dict.items():
                    logger.error("pipeline_cell stage_config set pipeline_stage fail!")
                    logger.warning("config cell name:" + str(config_cell_name) +
                                   " config stage num:" + str(config_stage_num))
                logger.warning("network:" + str(self.network))
                logger.warning("cell name available:")
                for cell_name, _ in self.network.cells_and_names():
                    logger.warning(cell_name)
                raise KeyError("For 'PipelineCell', the argument 'stage_config' : {} is not "
                               "found in 'network' : {}".format(config_dict, network))
        if segment_config is None:
            return
        self._config_segment(segment_config)


    def _config_segment(self, segment_config):
        """
        Config segment num for cell.
        """
        config_dict = segment_config.copy()

        for cell_name, cell in self.network.cells_and_names():
            if cell_name in segment_config:
                setattr(cell, "pipeline_segment", segment_config[cell_name])
                del config_dict[cell_name]
        if str(self.network) in segment_config:
            setattr(self.network, "pipeline_segment", segment_config[str(self.network)])
            del config_dict[str(self.network)]
        # if there are any config elements left, print them
        if config_dict:
            for config_cell_name, config_segment_num in config_dict.items():
                logger.error("pipeline_cell segment_config set pipeline_segment fail!")
                logger.warning("config cell name:" + str(config_cell_name) +
                               " config segment num:" + str(config_segment_num))
            logger.warning("network:" + str(self.network))
            logger.warning("cell name available:")
            for cell_name, _ in self.network.cells_and_names():
                logger.warning(cell_name)
            raise KeyError("For 'PipelineCell', the argument 'segment_config' : {} is not "
                           "found in 'network' : {}".format(config_dict, self.network))


    def construct(self, *args, **kwargs):
        ret = None
        for i in range(self.micro_size):
            micro_input = self.micro_inputs[i](i, *args, **kwargs)
            output = self.network(*micro_input)
            if ret is not None:
                ret = self.add_list[i](ret, output)
            else:
                ret = output
        return ret


class Pipeline(PipelineCell):
    """
    Specify the number of micro_batch for pipeline parallelism and the division rules for stage,
    and specify the segment info.

    Note:
        micro_size must be greater than or equal to pipeline stages.

    Args:
        network (Cell): The target network to wrap.
        micro_size (int): MicroBatch size.
        stage_config (dict, optional): Stage configuration for cell's execution in pipeline parallel. Default ``None``.
        segment_config (dict, optional): The segment configuration for each cell's execution in pipeline parallel.
            Default ``None``.

    Raises:
        TypeError: The type of `network` is not cell.
        TypeError: If the type of `micro_size` is not int.
        ValueError: When `micro_size` is 0 or negative value.
        KeyError: `dict` cell name matching exception,
            there are remaining configuration items after traversing all `cell` under the current `network`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore.parallel.nn import Pipeline
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://atomgit.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> net = Pipeline(net, 4, stage_config={"cell_name_0": 0, "cell_name_1": 1})
    """


class MicroBatchInterleaved(Cell):
    """
    Implement the static graph parallel multi-copy splitting function to enable concurrent computation
    and communication.
    Application scenario: When there is model parallelism in semi-automatic mode
    and network, if the first slice data is calculating forward, the second slice data will execute the
    communication operators at the same time, to achieve the performance acceleration of communication and computing
    concurrency.

    Args:
        network (Cell): The target network to wrap.
        interleave_num (int, optional): split num of batch size. Default: ``2`` .

    Inputs:
        tuple[Tensor]. It's the same as the input of the `network` .

    Outputs:
        The wrapped input. The output of the input `network` should be a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore.nn as nn
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://atomgit.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> net = nn.MicroBatchInterleaved(net, 2)
    """
    def __init__(self, network, interleave_num=2):
        super().__init__(auto_prefix=False)
        if not isinstance(interleave_num, int):
            raise TypeError("For 'MicroBatchInterleaved', the argument 'interleave_num' must be integer, "
                            "but got the type : {}.".format(type(interleave_num)))
        if interleave_num <= 0:
            raise ValueError("For 'MicroBatchInterleaved', the argument 'interleave_num' must be large than 0, "
                             "but got {}.".format(interleave_num))
        self.network = network
        self.interleave_num = interleave_num
        self.interleave_inputs = nn.CellList()
        self.add = P.Add().add_prim_attr("micro_interleaved_add_flag", True)
        for _ in range(interleave_num):
            interleave_data = _MicroBatch(interleave_num)
            interleave_data.strided_slice.add_prim_attr("strided_slice_flag", True)
            interleave_data.strided_slice.add_prim_attr("interleave_num", interleave_num)
            self.interleave_inputs.append(interleave_data)
        self._get_attr_from_cell(network)

    def construct(self, *args, **kwargs):
        output = 0.0
        for i in range(self.interleave_num):
            interleave_input = self.interleave_inputs[i](i, *args, **kwargs)
            output = self.add(output, self.network(*interleave_input))
        return output


class GradAccumulation(Cell):
    """
    Implementation of parallel gradient accumulation for static graphs.

    Args:
        network (Cell): The target network to wrap.
        micro_size (int): MicroBatch size.

    Raises:
        TypeError: The type of `network` is not cell.
        TypeError: If the type of `micro_size` is not int.
        ValueError: When `micro_size` is 0 or a negative value.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore.parallel.nn import GradAccumulation
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://atomgit.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> net = GradAccumulation(net, 4)
    """
    def __init__(self, network, micro_size):
        super().__init__(auto_prefix=False)
        self.network = network
        self.micro_inputs = nn.CellList()
        self.micro_size = micro_size
        self.add_list = []
        if not isinstance(network, Cell):
            raise TypeError("For 'GradAccumulation', the argument 'network' must cell type, "
                            "but got the type : {}.".format(type(network)))
        if not isinstance(micro_size, int):
            raise TypeError("For 'GradAccumulation', the argument 'micro_size' must be integer, "
                            "but got the type : {}.".format(type(micro_size)))
        if micro_size <= 0:
            raise ValueError("For 'GradAccumulation', the argument 'micro_size' must be large than 0, "
                             "but got {}.".format(micro_size))
        for i in range(micro_size):
            micro_input = _MicroBatch(micro_size)
            micro_input.strided_slice.add_prim_attr("grad_accu_num", micro_size)
            self.micro_inputs.append(micro_input)
            self.add = P.Add().add_prim_attr("forward_end", i)
            self.add_list.append(self.add)
        self._get_attr_from_cell(network)

    def construct(self, *args, **kwargs):
        ret = None
        for i in range(self.micro_size):
            micro_input = self.micro_inputs[i](i, *args, **kwargs)
            output = self.network(*micro_input)
            if ret is not None:
                ret = self.add_list[i](ret, output)
            else:
                ret = output
        return ret
