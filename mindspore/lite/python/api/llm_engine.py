# Copyright 2023 Huawei Technologies Co., Ltd
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
"""
LLMEngin interface
"""
__all__ = ['LLMReq', 'LLMEngineStatus', 'LLMRole', 'LLMEngine', 'LLMStatusCode', 'LLMException']

import os
import sys
import threading
from enum import Enum
from typing import Union, List, Tuple, Dict
from mindspore_lite._checkparam import check_isinstance, check_uint32_number_range, check_uint64_number_range
from mindspore_lite.tensor import Tensor
from mindspore_lite.lib._c_lite_wrapper import LLMEngine_, LLMReq_, LLMRole_, StatusCode, LLMClusterInfo_
from mindspore_lite.model import set_env


class LLMReq:
    """
    LLMEngine request, used to represent a multi round inference task.
    """

    def __init__(self, prompt_cluster_id: int, req_id: int, prompt_length: int):
        check_uint64_number_range("prompt_cluster_id", prompt_cluster_id)
        check_uint64_number_range("req_id", req_id)
        check_uint64_number_range("prompt_length", prompt_length)
        self.llm_request_ = LLMReq_()
        self.llm_request_.prompt_cluster_id = prompt_cluster_id
        self.llm_request_.req_id = req_id
        self.llm_request_.prompt_length = prompt_length

    _llm_req_id = 0
    _llm_req_id_lock = threading.Lock()

    @staticmethod
    def next_req_id():
        with LLMReq._llm_req_id_lock:
            new_req_id = LLMReq._llm_req_id
            LLMReq._llm_req_id += 1
        return new_req_id

    @property
    def req_id(self):
        """Get request id of this inference task"""
        return self.llm_request_.req_id

    @req_id.setter
    def req_id(self, req_id: int):
        """Set request id of this inference task"""
        check_uint64_number_range("req_id", req_id)
        self.llm_request_.req_id = req_id

    @property
    def prompt_length(self):
        """Set prompt length of this inference task"""
        return self.llm_request_.prompt_length

    @prompt_length.setter
    def prompt_length(self, prompt_length: int):
        """Get prompt length of this inference task"""
        check_uint64_number_range("prompt_length", prompt_length)
        self.llm_request_.prompt_length = prompt_length

    @property
    def prompt_cluster_id(self):
        """Get prompt cluster id of this inference task in LLMEngine"""
        return self.llm_request_.prompt_cluster_id

    @prompt_cluster_id.setter
    def prompt_cluster_id(self, prompt_cluster_id: int):
        """Set prompt cluster id of this inference task in LLMEngine"""
        check_uint64_number_range("prompt_cluster_id", prompt_cluster_id)
        self.llm_request_.prompt_cluster_id = prompt_cluster_id

    @property
    def decoder_cluster_id(self):
        """Get decoder cluster id of this inference task in LLMEngine"""
        return self.llm_request_.decoder_cluster_id

    @decoder_cluster_id.setter
    def decoder_cluster_id(self, decoder_cluster_id: int):
        """Set decoder cluster id of this inference task in LLMEngine"""
        check_uint64_number_range("decoder_cluster_id", decoder_cluster_id)
        self.llm_request_.decoder_cluster_id = decoder_cluster_id

    @property
    def prefix_id(self):
        """Get decoder prefix id of this inference task in LLMEngine"""
        return self.llm_request_.prefix_id

    @prefix_id.setter
    def prefix_id(self, prefix_id: int):
        """Set decoder prefix id of this inference task in LLMEngine"""
        check_uint64_number_range("prefix_id", prefix_id)
        self.llm_request_.prefix_id = prefix_id

    @property
    def sequence_length(self):
        """Get decoder sequence length of this inference task in LLMEngine"""
        return self.llm_request_.sequence_length

    @sequence_length.setter
    def sequence_length(self, sequence_length: int):
        """Set decoder sequence length of this inference task in LLMEngine"""
        check_uint64_number_range("sequence_length", sequence_length)
        self.llm_request_.sequence_length = sequence_length


class LLMEngineStatus:
    """
    LLMEngine Status, which can be got from LLEngine.fetch_status.
    """

    def __init__(self, status):
        self.status_ = status

    @property
    def empty_max_prompt_kv(self):
        """Get empty count of prompt KV cache of this LLMEngine object"""
        return self.status_.empty_max_prompt_kv

    @property
    def num_free_blocks(self):
        """Get number of free blocks PagedAttention"""
        return self.status_.num_free_blocks

    @property
    def num_total_blocks(self):
        """Get number of total blocks PagedAttention"""
        return self.status_.num_total_blocks

    @property
    def block_size(self):
        """Get block size of PagedAttention"""
        return self.status_.block_size


class LLMStatusCode(Enum):
    """
    LLM Error Code
    """
    LLM_SUCCESS = StatusCode.kSuccess
    LLM_WAIT_PROC_TIMEOUT = StatusCode.kLiteLLMWaitProcessTimeOut
    LLM_KV_CACHE_NOT_EXIST = StatusCode.kLiteLLMKVCacheNotExist
    LLM_REPEAT_REQUEST = StatusCode.kLiteLLMRepeatRequest
    LLM_REQUEST_ALREADY_COMPLETED = StatusCode.kLiteLLMRequestAlreadyCompleted
    LLM_PARAM_INVALID = StatusCode.kLiteParamInvalid
    LLM_ENGINE_FINALIZED = StatusCode.kLiteLLMEngineFinalized
    LLM_NOT_YET_LINK = StatusCode.kLiteLLMNotYetLink
    LLM_ALREADY_LINK = StatusCode.kLiteLLMAlreadyLink
    LLM_LINK_FAILED = StatusCode.kLiteLLMLinkFailed
    LLM_UNLINK_FAILED = StatusCode.kLiteLLMUnlinkFailed
    LLM_NOTIFY_PROMPT_UNLINK_FAILED = StatusCode.kLiteLLMNofiryPromptUnlinkFailed
    LLM_CLUSTER_NUM_EXCEED_LIMIT = StatusCode.kLiteLLMClusterNumExceedLimit
    LLM_PROCESSING_LINK = StatusCode.kLiteLLMProcessingLink
    LLM_DEVICE_OUT_OF_MEMORY = StatusCode.kLiteLLMOutOfMemory
    LLM_PREFIX_ALREADY_EXIST = StatusCode.kLiteLLMPrefixAlreadyExist
    LLM_PREFIX_NOT_EXIST = StatusCode.kLiteLLMPrefixNotExist
    LLM_SEQ_LEN_OVER_LIMIT = StatusCode.kLiteLLMSeqLenOverLimit
    LLM_NO_FREE_BLOCK = StatusCode.kLiteLLMNoFreeBlock
    LLM_BLOCKS_OUT_OF_MEMORY = StatusCode.kLiteLLMBlockOutOfMemory


class LLMRole(Enum):
    """
    Role of LLMEngine. When LLMEngine accelerates inference performance through KVCache, the generation process includes
    one full inference and n incremental inference, involving both full and incremental models. When the full and
    incremental models are deployed on different nodes, the role of the node where the full models are located is
    ``Prompt``, and the role of the node where the incremental models are located is ``Decoder``.
    """
    Prompt = 0
    Decoder = 1


class LLMException(RuntimeError):
    """
    Base Error class for LLM
    """
    def __init__(self, *args: object):
        super().__init__(*args)
        self._status_code = LLMStatusCode.LLM_SUCCESS

    @property
    def statusCode(self):
        """
        LLMException status code property
        """
        return self._status_code

    def StatusCode(self):
        """
        get LLMException status code
        """
        return self._status_code



class LLMKVCacheNotExist(LLMException):
    """
    Key & Value cache does not exist in Prompt cluster specified by parameter LLMReq.prompt_cluster_id, and the
    LLM request may have been released in Prompt cluster by calling method LLMEngine.complete_request.
    Raised in LLMEngine.predict.
    """
    def __init__(self, *args: object):
        super().__init__(*args)
        self._status_code = LLMStatusCode.LLM_KV_CACHE_NOT_EXIST



class LLMWaitProcessTimeOut(LLMException):
    """
    Request waiting for processing timed out. Raised in LLMEngine.predict.
    """
    def __init__(self, *args: object):
        super().__init__(*args)
        self._status_code = LLMStatusCode.LLM_WAIT_PROC_TIMEOUT


class LLMRepeatRequest(LLMException):
    """
    Request repeated . Raised in LLMEngine.predict.
    """
    def __init__(self, *args: object):
        super().__init__(*args)
        self._status_code = LLMStatusCode.LLM_REPEAT_REQUEST


class LLMRequestAlreadyCompleted(LLMException):
    """
    Request has already completed. Raised in LLMEngine.predict.
    """
    def __init__(self, *args: object):
        super().__init__(*args)
        self._status_code = LLMStatusCode.LLM_REQUEST_ALREADY_COMPLETED


class LLMEngineFinalized(LLMException):
    """
    LLMEngine has finalized. Raised in LLMEngine.predict.
    """
    def __init__(self, *args: object):
        super().__init__(*args)
        self._status_code = LLMStatusCode.LLM_ENGINE_FINALIZED


class LLMParamInvalid(LLMException):
    """
    Parameters invalid. Raised in LLMEngine.predict.
    """
    def __init__(self, *args: object):
        super().__init__(*args)
        self._status_code = LLMStatusCode.LLM_PARAM_INVALID



class LLMNotYetLink(LLMException):
    """
    Decoder cluster has no link with prompt.
    """
    def __init__(self, *args: object):
        super().__init__(*args)
        self._status_code = LLMStatusCode.LLM_NOT_YET_LINK



class LLMOutOfMemory(LLMException):
    """
    Device out of memory.
    """
    def __init__(self, *args: object):
        super().__init__(*args)
        self._status_code = LLMStatusCode.LLM_DEVICE_OUT_OF_MEMORY



class LLMPrefixAlreadyExist(LLMException):
    """
    Prefix has already existed.
    """
    def __init__(self, *args: object):
        super().__init__(*args)
        self._status_code = LLMStatusCode.LLM_PREFIX_ALREADY_EXIST



class LLMPrefixNotExist(LLMException):
    """
    Prefix does not exist.
    """
    def __init__(self, *args: object):
        super().__init__(*args)
        self._status_code = LLMStatusCode.LLM_PREFIX_NOT_EXIST



class LLMSeqLenOverLimit(LLMException):
    """
    Sequence length exceed limit.
    """
    def __init__(self, *args: object):
        super().__init__(*args)
        self._status_code = LLMStatusCode.LLM_SEQ_LEN_OVER_LIMIT



class LLMNoFreeBlocks(LLMException):
    """
    No free block.
    """
    def __init__(self, *args: object):
        super().__init__(*args)
        self._status_code = LLMStatusCode.LLM_NO_FREE_BLOCK



class LLMBlockOutOfMemory(LLMException):
    """
    Block is out of memory.
    """
    def __init__(self, *args: object):
        super().__init__(*args)
        self._status_code = LLMStatusCode.LLM_BLOCKS_OUT_OF_MEMORY



class LLMClusterInfo:
    """
    The `LLMClusterInfo` class defines a MindSpore Lite's LLMEngine cluster, used to link and unlink clusters.

    Args:
        remote_role (LLMRole): Role of remote LLMEngine object.
        remote_cluster_id (int): Cluster id of remote LLMEngine object.

    Raises:
        TypeError: `remote_role` is not a LLMRole.
        TypeError: `remote_cluster_id` is not an int.

    Examples:
        >>> import mindspore_lite as mslite
        >>> remote_cluster_id = 1
        >>> cluster = mslite.LLMClusterInfo(mslite.LLMRole.Prompt, remote_cluster_id)
        >>> cluster.append_local_ip_info(("192.168.1.1", 2222))
        >>> cluster.append_remote_ip_info(("192.168.2.1", 2222))
        >>> local_cluster_id = 0
        >>> llm_engine = mslite.LLMEngine(mslite.LLMRole.Decoder, local_cluster_id)
        >>> # ... llm_engine.init
        >>> llm_engine.link_clusters([cluster])
    """
    def __init__(self, remote_role: LLMRole, remote_cluster_id: int):
        check_uint64_number_range("remote_cluster_id", remote_cluster_id)
        check_isinstance("remote_role", remote_role, LLMRole)
        self.llm_cluster_ = LLMClusterInfo_()
        self.llm_cluster_.remote_cluster_id = remote_cluster_id
        remote_role_type_int = 0 if remote_role == LLMRole.Prompt else 1  # 0: Prompt, 1: Decoder
        self.llm_cluster_.remote_role_type = remote_role_type_int

    @property
    def remote_role(self):
        """Get remote role of this LLMClusterInfo object"""
        remote_role_type_int = self.llm_cluster_.remote_role
        return LLMRole.Prompt if remote_role_type_int == 0 else LLMRole.Decoder  # 0: Prompt, 1: Decoder

    @remote_role.setter
    def remote_role(self, remote_role):
        """Set remote role of this LLMClusterInfo object"""
        check_isinstance("remote_role", remote_role, LLMRole)
        remote_role_type_int = 0 if remote_role == LLMRole.Prompt else 1  # 0: Prompt, 1: Decoder
        self.llm_cluster_.remote_role_type = remote_role_type_int

    @property
    def remote_cluster_id(self):
        """Get remote cluster id of this LLMClusterInfo object"""
        return self.llm_cluster_.remote_cluster_id

    @remote_cluster_id.setter
    def remote_cluster_id(self, remote_cluster_id):
        """Set remote cluster id of this LLMClusterInfo object"""
        check_uint64_number_range("remote_cluster_id", remote_cluster_id)
        self.llm_cluster_.remote_cluster_id = remote_cluster_id

    def append_local_ip_info(self, address):
        """
        Append local ip info.

        Args:
            address: ip address, in format ('xxx.xxx.xxx.xxx', xxx) or (xxx, xxx).

        Raises:
            TypeError: `address` format or type is invalid.
            ValueError: `address` value is invalid.

        Examples:
            >>> import mindspore_lite as mslite
            >>> cluster_id = 1
            >>> cluster = mslite.LLMClusterInfo(mslite.LLMRole.Prompt, 0)
            >>> cluster.append_local_ip_info(("192.168.1.1", 2222))
        """
        ip, port = LLMClusterInfo._trans_address(address)
        self.llm_cluster_.append_local_ip_info(ip, port)

    @property
    def local_ip_infos(self):
        """Get all local ip infos of this LLMClusterInfo object"""
        return tuple((ip, port) for ip, port in self.llm_cluster_.get_local_ip_infos())

    def append_remote_ip_info(self, address):
        """
        Append remote ip info.

        Args:
            address: ip address, in format ('xxx.xxx.xxx.xxx', xxx) or (xxx, xxx).

        Raises:
            TypeError: `address` format or type is invalid.
            ValueError: `address` value is invalid.

        Examples:
            >>> import mindspore_lite as mslite
            >>> cluster_id = 1
            >>> cluster = mslite.LLMClusterInfo(mslite.LLMRole.Prompt, 0)
            >>> cluster.append_remote_ip_info(("192.168.1.1", 2222))
        """
        ip, port = LLMClusterInfo._trans_address(address)
        self.llm_cluster_.append_remote_ip_info(ip, port)

    @property
    def remote_ip_infos(self):
        """Get all remote ip infos of this LLMClusterInfo object"""
        return tuple((ip, port) for ip, port in self.llm_cluster_.get_remote_ip_infos())

    @staticmethod
    def _trans_address(address):
        """Transfer address from str format 'xxx.xxx.xxx.xxx' to int"""
        if not isinstance(address, tuple):
            raise TypeError(f"address must be in format of ('xxx.xxx.xxx.xxx', xxx) or (xxx, xxx), but got {address}")
        if len(address) != 2:
            raise TypeError(f"address must be in format of ('xxx.xxx.xxx.xxx', xxx) or (xxx, xxx), but got {address}")
        ip, port = address
        if not isinstance(ip, (str, int)) or not isinstance(port, int):
            raise TypeError(f"address must be in format of ('xxx.xxx.xxx.xxx', xxx) or (xxx, xxx), but got {address}")
        if isinstance(ip, int) and (ip < 0 or ip > pow(2, 32) - 1):
            raise ValueError(f"address ip should in range [0,{pow(2, 32) - 1}], but got {ip}")
        if port < 0 or port > 65535:
            raise ValueError(f"address port should in range [0,65535], but got {port}")
        if isinstance(ip, str):
            try:
                if "." not in ip:  # format ("[0-9]+", xxx)
                    ip = int(ip)
                    return ip, port
            except ValueError:
                raise ValueError(
                    f"address must be in format of ('xxx.xxx.xxx.xxx', xxx) or (xxx, xxx), but got {address}")
            try:
                import socket
                ip = socket.inet_aton(ip)
                ip = int.from_bytes(ip, byteorder=sys.byteorder)
            except OSError:
                raise ValueError(
                    f"address must be in format of ('xxx.xxx.xxx.xxx', xxx) or (xxx, xxx), but got {address}")
        return ip, port


def _handle_llm_status(status, func_name, other_info):
    """Handle LLM error code"""
    status_code = status.StatusCode()
    if status_code != StatusCode.kSuccess:
        if not isinstance(other_info, str):
            other_info = other_info()
        error_code_map = {
            StatusCode.kLiteLLMWaitProcessTimeOut:
                LLMWaitProcessTimeOut(f"{func_name} failed: Waiting for processing timeout, {other_info}"),
            StatusCode.kLiteLLMKVCacheNotExist:
                LLMKVCacheNotExist(f"{func_name} failed: KV Cache not exist, {other_info}."),
            StatusCode.kLiteLLMRepeatRequest: LLMRepeatRequest(f"{func_name} failed: Repeat request, {other_info}."),
            StatusCode.kLiteLLMRequestAlreadyCompleted:
                LLMRequestAlreadyCompleted(f"{func_name} failed: Request has already completed, {other_info}."),
            StatusCode.kLiteLLMEngineFinalized:
                LLMEngineFinalized(f"{func_name} failed: LLMEngine has finalized, {other_info}."),
            StatusCode.kLiteParamInvalid: LLMParamInvalid(f"{func_name} failed: Parameters invalid, {other_info}."),
            StatusCode.kLiteLLMNotYetLink:
                LLMNotYetLink(f"{func_name} failed: Decoder cluster is no link with prompt, {other_info}."),
            StatusCode.kLiteLLMOutOfMemory: LLMOutOfMemory(f"{func_name} failed: Device out of memory, {other_info}."),
            StatusCode.kLiteLLMPrefixAlreadyExist:
                LLMPrefixAlreadyExist(f"{func_name} failed: Prefix has already existed, {other_info}."),
            StatusCode.kLiteLLMPrefixNotExist:
                LLMPrefixNotExist(f"{func_name} failed: Prefix does not exist, {other_info}."),
            StatusCode.kLiteLLMSeqLenOverLimit:
                LLMSeqLenOverLimit(f"{func_name} failed: Sequence length exceed limit, {other_info}."),
            StatusCode.kLiteLLMNoFreeBlock: LLMNoFreeBlocks(f"{func_name} failed: No free block, {other_info}."),
            StatusCode.kLiteLLMBlockOutOfMemory:
                LLMBlockOutOfMemory(f"{func_name} failed: NBlock is out of memory, {other_info}."),
        }
        if status_code in error_code_map:
            raise error_code_map[status_code]
        raise RuntimeError(f"{func_name} failed, {other_info}.")


def _llm_req_str(llm_req):
    return "{" + f"llm_req: {llm_req.req_id}, prompt_cluster_id: {llm_req.prompt_cluster_id}, " \
                 f"decoder_cluster_id: {llm_req.decoder_cluster_id}, prefix_id: {llm_req.prefix_id}, " \
                 f"prompt_length: {llm_req.prompt_length}" + "}"


class LLMModel:
    """
    The `LLMModel` class defines one model of MindSpore Lite's LLMEngine, used to schedule and execute inference
    request. LLMModel object should be created from LLMEngine.add_model.

    Examples:
        >>> import mindspore_lite as mslite
        >>> cluster_id = 1
        >>> llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, cluster_id)
        >>> model_paths = [os.path.join(model_dir, f"device_${rank}") for rank in range(4)]
        >>> options = {}
        >>> llm_model = llm_engine.add_mode(model_paths, options)  # return LLMModel object
        >>> llm_engine.init()
        >>> llm_req = mslite.LLMReq(llm_engine.cluster_id, mslite.LLMReq.next_req_id(), prompt_length=1024)
        >>> inputs = [mslite.Tensor(np_input) for np_input in np_inputs]
        >>> outputs = llm_model.predit(llm_req, inputs)
        >>> for output in outputs:
        >>>    print(f"output is {output.get_data_to_numpy()}")
        >>> llm_engine.complete(llm_req)
    """
    def __init__(self, model_obj, batch_mode):
        self.model_ = model_obj  # inited by LLMEngine
        self.batch_mode_ = batch_mode
        self.inited_ = False

    def predict(self, llm_req: Union[LLMReq, List[LLMReq], Tuple[LLMReq]], inputs: Union[Tuple[Tensor], List[Tensor]]):
        """
        Schedule and execute inference request.

        Args:
            llm_req (Union[LLMReq, list[LLMReq], Tuple[LLMReq]]): Request of LLMEngine.
            inputs (Union[Tuple[Tensor], List[Tensor]]): A list that includes all input Tensors in order.

        Returns:
            list[Tensor], the output Tensor list of the model.

        Raises:
            TypeError: `llm_req` is not a LLMReq.
            TypeError: `inputs` is not a list.
            TypeError: `inputs` is a list, but the elements are not Tensor.
            RuntimeError: schedule and execute inference request failed.
            RuntimeError: this LLMEngine object has not been inited.
            LLMKVCacheNotExist: Key & Value cache does not exist in Prompt cluster specified by
                `llm_req.prompt_cluster_id`, and the LLM request may have been released in Prompt cluster
                by calling method LLMEngine.complete_request.
            LLMWaitProcessTimeOut: Request waiting for processing timed out.
            LLMRepeatRequest: Repeat request.
            LLMRequestAlreadyCompleted: Request has already completed.
            LLMEngineFinalized: LLMEngine has finalized.
            LLMParamInvalid: Parameters invalid.
        """
        if not self.inited_:
            raise RuntimeError(f"LLMEngine is not inited or init failed")
        if not isinstance(inputs, (tuple, list)):
            raise TypeError(f"inputs must be list/tuple of Tensor, but got {type(inputs)}.")
        if not isinstance(llm_req, (list, tuple, LLMReq)):
            raise TypeError(f"llm_req must be instance of LLMReq or list/tuple of LLMReq, but got {type(llm_req)}.")
        if self.batch_mode_ == "manual":
            if not isinstance(llm_req, (list, tuple)):
                raise TypeError(f"llm_req must be list/tuple of LLMReq when batch_mode is \"manual\","
                                f" but got {type(llm_req)}.")
            for i, item in enumerate(llm_req):
                if not isinstance(item, LLMReq):
                    raise TypeError(f"llm_req element must be LLMReq when batch_mode is \"manual\","
                                    f" but got {type(item)} at index {i}.")
        else:
            if not isinstance(llm_req, LLMReq):
                raise TypeError(f"llm_req must be LLMReq when batch_mode is \"auto\", but got {type(llm_req)}.")

        _inputs = []
        for i, element in enumerate(inputs):
            if not isinstance(element, Tensor):
                raise TypeError(f"inputs element must be Tensor, but got {type(element)} at index {i}.")
            # pylint: disable=protected-access
            _inputs.append(element._tensor)
        # pylint: disable=protected-access
        if self.batch_mode_ == "manual":
            llm_req_list = [item.llm_request_ for item in llm_req]
            outputs, status = self.model_.predict_batch(llm_req_list, _inputs)
        else:
            outputs, status = self.model_.predict(llm_req.llm_request_, _inputs)

        def get_info():
            if isinstance(llm_req, LLMReq):
                req_infos = _llm_req_str(llm_req)
            else:
                req_infos = [_llm_req_str(llm) for llm in llm_req]

            input_infos = [(item.shape, item.dtype) for item in inputs]
            info = f"llm_req {req_infos}, inputs {input_infos}"
            return info

        _handle_llm_status(status, "predict", get_info)
        if not outputs:
            raise RuntimeError(f"predict failed, {get_info()}.")
        predict_outputs = [Tensor(output) for output in outputs]
        return predict_outputs

    def pull_kv(self, llm_req: LLMReq):
        """
        For Decoder LLMEngine, fetch KVCache from Prompt LLMEngine specified by llm_req.prompt_cluster and
        llm_req.req_id.

        Args:
            llm_req (LLMReq): Request of LLMEngine.

        Raises:
            TypeError: `llm_req` is not a LLMReq.
            RuntimeError: this LLMEngine object has not been inited.
            RuntimeError: Failed to pull KVCache.
            LLMKVCacheNotExist: Key & Value cache does not exist in Prompt cluster specified by
                `llm_req.prompt_cluster_id`, and the LLM request may have been released in Prompt cluster
                by calling method LLMEngine.complete_request.
            LLMParamInvalid: Parameters invalid.
        """
        if not self.inited_:
            raise RuntimeError(f"LLMEngine is not inited or init failed")
        if self.batch_mode_ != "manual":
            raise RuntimeError(f"LLMEngine.pull_kv is only support when batch_mode is \"manual\"")
        check_isinstance("llm_req", llm_req, LLMReq)
        # pylint: disable=protected-access
        status = self.model_.pull_kv(llm_req.llm_request_)
        _handle_llm_status(status, "pull_kv", "llm_req " + _llm_req_str(llm_req))

    def merge_kv(self, llm_req: LLMReq, batch_index: int, batch_id: int = 0):
        """
        For Decoder LLMEngine, merge KVCache of LLMReq specified by `llm_req.req_id` into `batch_index` slot.
        Args:
            llm_req (LLMReq): Request of LLMEngine.
            batch_index (int): Request batch index.
            batch_id (int): Request pipline index for ping pong pipeline.

        Raises:
            TypeError: `llm_req` is not a LLMReq.
            RuntimeError: this LLMEngine object has not been inited.
            RuntimeError: Failed to merge KVCache.
            LLMParamInvalid: Parameters invalid.
        """
        if not self.inited_:
            raise RuntimeError(f"LLMEngine is not inited or init failed")
        if self.batch_mode_ != "manual":
            raise RuntimeError(f"LLMEngine.merge_kv is only support when batch_mode is \"manual\"")
        check_isinstance("llm_req", llm_req, LLMReq)
        check_uint32_number_range("batch_index", batch_index)
        check_uint32_number_range("batch_id", batch_id)
        # pylint: disable=protected-access
        status = self.model_.merge_kv(llm_req.llm_request_, batch_index, batch_id)
        _handle_llm_status(status, "merge_kv", "llm_req " + _llm_req_str(llm_req))

    def preload_prompt_prefix(self, llm_req: LLMReq, inputs: Union[Tuple[Tensor], List[Tensor]]):
        """
        Preload prompt inference common prefix.

        Args:
            llm_req (LLMReq): Request of LLMEngine.
            inputs (Union[Tuple[Tensor], List[Tensor]]): A list that includes all input Tensors in order.

        Raises:
            TypeError: `llm_req` is not a LLMReq.
            TypeError: `inputs` is not a list.
            TypeError: `inputs` is a list, but the elements are not Tensor.
            RuntimeError: preload prompt prefix inference request failed.
            RuntimeError: this LLMEngine object has not been inited.
            LLMParamInvalid: Parameters invalid.
        """
        if not self.inited_:
            raise RuntimeError(f"LLMEngine is not inited or init failed")
        if not isinstance(inputs, (tuple, list)):
            raise TypeError(f"inputs must be list/tuple of Tensor, but got {type(inputs)}.")
        check_isinstance("llm_req", llm_req, LLMReq)
        _inputs = []
        for i, element in enumerate(inputs):
            if not isinstance(element, Tensor):
                raise TypeError(f"inputs element must be Tensor, but got {type(element)} at index {i}.")
            # pylint: disable=protected-access
            _inputs.append(element._tensor)
        # pylint: disable=protected-access
        ret = self.model_.preload_prompt_prefix(llm_req.llm_request_, _inputs)
        _handle_llm_status(ret, "preload_prompt_prefix", "llm_req " + _llm_req_str(llm_req))

    def release_prompt_prefix(self, llm_req: LLMReq):
        """
        Release the memory space used by prompt inference common prefix.

        Args:
            llm_req (LLMReq): Request of LLMEngine.

        Raises:
            TypeError: `llm_req` is not a LLMReq.
            RuntimeError: this LLMEngine object has not been inited.
            LLMParamInvalid: Parameters invalid.
        """
        if not self.inited_:
            raise RuntimeError(f"LLMEngine is not inited or init failed")
        check_isinstance("llm_req", llm_req, LLMReq)
        # pylint: disable=protected-access
        ret = self.model_.release_prompt_prefix(llm_req.llm_request_)
        _handle_llm_status(ret, "release_prompt_prefix", "llm_req " + _llm_req_str(llm_req))

    def get_inputs(self) -> List[Tensor]:
        """
        Get inputs of this LLMModel.

        Returns:
            Tuple[Tensor], the input Tensor list of the model.

        Examples:
            >>> import mindspore_lite as mslite
            >>> cluster_id = 1
            >>> llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, cluster_id)
            >>> model_paths = [os.path.join(model_dir, f"device_${rank}") for rank in range(4)]
            >>> options = {}
            >>> llm_model = llm_engine.add_mode(model_paths, options)  # return LLMModel object
            >>> inputs = llm_model.get_inputs()
            >>> for i in range(len(inputs)):
            ...     print(f"Input name {inputs[i].name}, dtype {inputs[i].dtype}, shape: {inputs[i].shape}")
        """
        if not self.model_:
            raise RuntimeError(f"LLMModel is invalid, please return LLMModel from LLMEngine.add_model.")
        inputs = []
        for _tensor in self.model_.get_inputs():
            inputs.append(Tensor(_tensor))
        return inputs


class LLMEngine:
    """
    The `LLMEngine` class defines a MindSpore Lite's LLMEngine, used to load and manage Large Language Mode,
    and schedule and execute inference request.

    Args:
        role (LLMRole): Role of this LLMEngine object.
        cluster_id (int): Cluster id of this LLMEngine object.
        batch_mode (str): Controls whether the request batching is "auto" formed by the framework or "manual"ly
            by the user. Option is "auto" or "manual", default "auto".

    Raises:
        TypeError: `role` is not a LLMRole.
        TypeError: `cluster_id` is not an int.

    Examples:
        >>> import mindspore_lite as mslite
        >>> cluster_id = 1
        >>> llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, cluster_id)
        >>> model_paths = [os.path.join(model_dir, f"device_${rank}") for rank in range(4)]
        >>> options = {}
        >>> llm_model = llm_engine.add_mode(model_paths, options)  # return LLMModel object
        >>> llm_engine.init()
        >>> llm_req = mslite.LLMReq(llm_engine.cluster_id, mslite.LLMReq.next_req_id(), prompt_length=1024)
        >>> inputs = [mslite.Tensor(np_input) for np_input in np_inputs]
        >>> outputs = llm_model.predit(llm_req, inputs)
        >>> for output in outputs:
        >>>    print(f"output is {output.get_data_to_numpy()}")
        >>> llm_engine.complete(llm_req)
    """

    def __init__(self, role: LLMRole, cluster_id: int, batch_mode="auto"):
        check_isinstance("role", role, LLMRole)
        check_uint64_number_range("cluster_id", cluster_id)
        check_isinstance("batch_mode", batch_mode, str)
        if batch_mode != "auto" and batch_mode != "manual":
            raise ValueError(f"batch_mode should be str \"auto\" or \"manual\", but got {batch_mode}")
        self.role_ = role
        self.cluster_id_ = cluster_id
        self.batch_mode_ = batch_mode
        self.models_ = []
        self.inited_ = False
        role_inner = LLMRole_.Prompt if self.role == LLMRole.Prompt else LLMRole_.Decoder
        self.engine_ = LLMEngine_(role_inner, self.cluster_id, self.batch_mode)

    @property
    def cluster_id(self):
        """Get cluster id set to this LLMEngine object"""
        return self.cluster_id_

    @property
    def role(self):
        """Get LLM role set to this LLMEngine object"""
        return self.role_

    @property
    def batch_mode(self):
        """Get batch mode of this LLMEngine object"""
        return self.batch_mode_

    def add_model(self, model_paths: Union[Tuple[str], List[str]], options: Dict[str, str],
                  postprocess_model_path=None) -> LLMModel:
        """
        Add model to LLMEngine.

        Args:
            model_paths (Union[Tuple[str], List[str]]): List or tuple of model path.
            options (Dict[str, str]): Other init options of this LLMEngine object.
            postprocess_model_path (Union[str, None]): Postprocess model path, default None.

        Raises:
            TypeError: `model_paths` is not a list and tuple.
            TypeError: `model_paths` is a list or tuple, but the elements are not str.
            TypeError: `options` is not a dict.
            RuntimeError: add model failed.
        """
        if self.inited_:
            raise RuntimeError(f"Cannot add model for LLMEngine: LLMEngine has been inited")
        if not isinstance(model_paths, (list, tuple)):
            raise TypeError(f"model_paths must be tuple/list of str, but got {type(model_paths)}.")
        for i, model_path in enumerate(model_paths):
            if not isinstance(model_path, str):
                raise TypeError(f"model_paths element must be str, but got {type(model_path)} at index {i}.")
            if not os.path.exists(model_path):
                raise RuntimeError(f"model_paths {model_path} at index {i} does not exist!")
        check_isinstance("options", options, dict)
        for key, value in options.items():
            if not isinstance(key, str):
                raise TypeError(f"options key must be str, but got {type(key)}.")
            if not isinstance(value, str):
                raise TypeError(f"options value must be str, but got {type(value)}.")
        if postprocess_model_path is not None:
            if not isinstance(postprocess_model_path, str):
                raise TypeError(
                    f"postprocess_model_path must be None or str, but got {type(postprocess_model_path)}.")
            if not os.path.exists(postprocess_model_path):
                raise RuntimeError(f"postprocess_model_path {postprocess_model_path} does not"
                                   f" exist!")
        else:
            postprocess_model_path = ""

        ret, llm_model_inner = self.engine_.add_model(model_paths, options, postprocess_model_path)
        status_code = ret.StatusCode()
        if status_code == StatusCode.kLiteParamInvalid:
            raise LLMParamInvalid("Parameters invalid")
        if not ret.IsOk():
            role_str = 'Prompt' if self.role == LLMRole.Prompt else 'Decoder'
            raise RuntimeError(
                f"Failed to add_model, model paths {model_paths}, options {options}, postprocess path"
                f" {postprocess_model_path}, role {role_str}, cluster id {self.cluster_id}")
        llm_model = LLMModel(llm_model_inner, self.batch_mode_)
        self.models_.append(llm_model)
        return llm_model

    @set_env
    def init(self, options: Dict[str, str]):
        """
        Init LLMEngine.

        Args:
            options (Dict[str, str]): init options of this LLMEngine object.

        Raises:
            TypeError: `options` is not a dict.
            RuntimeError: init LLMEngine failed.
        """
        if self.inited_:
            raise RuntimeError(f"LLMEngine has been inited")
        if not self.models_:
            raise RuntimeError(f"At least one group of models need to be added through LLMEngine.add_model before call"
                               f" LLMEngine.init.")
        check_isinstance("options", options, dict)
        for key, value in options.items():
            if not isinstance(key, str):
                raise TypeError(f"options key must be str, but got {type(key)}.")
            if not isinstance(value, str):
                raise TypeError(f"options value must be str, but got {type(value)}.")
        ret = self.engine_.init(options)
        status_code = ret.StatusCode()
        if status_code == StatusCode.kLiteParamInvalid:
            raise LLMParamInvalid("Parameters invalid")
        if not ret.IsOk():
            role_str = 'Prompt' if self.role == LLMRole.Prompt else 'Decoder'
            raise RuntimeError(f"Failed to init LLMEngine, role {role_str}, cluster id {self.cluster_id},"
                               f" options {options}")
        self.inited_ = True
        for model in self.models_:
            model.inited_ = True

    def complete_request(self, llm_req: LLMReq):
        """
        Complete inference request.

        Args:
            llm_req (LLMReq): Request of LLMEngine.

        Raises:
            TypeError: `llm_req` is not a LLMReq.
            RuntimeError: this LLMEngine object has not been inited.
        """
        if not self.inited_:
            raise RuntimeError(f"LLMEngine is not inited or init failed")
        check_isinstance("llm_req", llm_req, LLMReq)
        ret = self.engine_.complete_request(llm_req.llm_request_)
        _handle_llm_status(ret, "complete_request", "llm_req " + _llm_req_str(llm_req))

    def finalize(self):
        """
        Finalize LLMEngine.
        """
        if not self.inited_:
            print(f"LLMEngine is not inited or init failed", flush=True)
            return
        self.engine_.finalize()

    def fetch_status(self):
        """
        Get LLMEngine status.

        Returns:
            LLMEngineStatus, LLMEngine status.

        Raises:
            RuntimeError: this LLMEngine object has not been inited.
        """
        if not self.inited_:
            raise RuntimeError(f"LLMEngine is not inited or init failed")
        status = self.engine_.fetch_status()
        return LLMEngineStatus(status)

    def link_clusters(self, clusters: Union[List[LLMClusterInfo], Tuple[LLMClusterInfo]], timeout=-1):
        """
        Link clusters.

        Args:
            clusters (Union[List[LLMClusterInfo], Tuple[LLMClusterInfo]]): clusters.
            timeout (int): timeout in seconds.

        Raises:
            TypeError: `clusters` is not list/tuple of LLMClusterInfo.
            RuntimeError: LLMEngine is not inited or init failed.

        Returns:
            Status, tuple[Status], Whether all clusters link normally, and the link status of each cluster.

        Examples:
            >>> import mindspore_lite as mslite
            >>> cluster_id = 1
            >>> llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, cluster_id)
            >>> model_paths = [os.path.join(model_dir, f"device_${rank}") for rank in range(4)]
            >>> options = {}
            >>> llm_engine.init(model_paths, options)
            >>> cluster = mslite.LLMClusterInfo(mslite.LLMRole.Prompt, 0)
            >>> cluster.append_local_ip_info(("192.168.1.1", 2222))
            >>> cluster.append_remote_ip_info(("192.168.2.1", 2222))
            >>> cluster2 = mslite.LLMClusterInfo(mslite.LLMRole.Prompt, 1)
            >>> cluster2.append_local_ip_info(("192.168.3.1", 2222))
            >>> cluster2.append_remote_ip_info(("192.168.4.2", 2222))
            >>> ret, rets = llm_engine.link_clusters((cluster, cluster2))
            >>> if not ret.IsOk():
            >>>    for ret_item in rets:
            >>>        if not ret_item.IsOk():
            >>>            # do something
        """
        if not self.inited_:
            raise RuntimeError(f"LLMEngine is not inited or init failed")
        if not isinstance(clusters, (tuple, list)):
            raise TypeError(f"clusters must be list/tuple of LLMClusterInfo, but got {type(clusters)}.")
        check_isinstance("timeout", timeout, int)
        for i, element in enumerate(clusters):
            if not isinstance(element, LLMClusterInfo):
                raise TypeError(f"clusters element must be LLMClusterInfo, but got {type(element)} at index {i}.")
        clusters_inners = [item.llm_cluster_ for item in clusters]
        ret, rets = self.engine_.link_clusters(clusters_inners, timeout)
        if not rets:
            _handle_llm_status(ret, "link_clusters", "")
        return ret, rets

    def unlink_clusters(self, clusters: Union[List[LLMClusterInfo], Tuple[LLMClusterInfo]], timeout=-1):
        """
        Unlink clusters.

        Args:
            clusters (Union[List[LLMClusterInfo], Tuple[LLMClusterInfo]]): clusters.
            timeout (int): LLMEngine is not inited or init failed.

        Raises:
            TypeError: `clusters` is not list/tuple of LLMClusterInfo.
            RuntimeError: Some error occurred.

        Returns:
            Status, tuple[Status], Whether all clusters unlink normally, and the unlink status of each cluster.

        Examples:
            >>> import mindspore_lite as mslite
            >>> cluster_id = 1
            >>> llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, cluster_id)
            >>> model_paths = [os.path.join(model_dir, f"device_${rank}") for rank in range(4)]
            >>> options = {}
            >>> llm_engine.init(model_paths, options)
            >>> cluster = mslite.LLMClusterInfo(mslite.LLMRole.Prompt, 0)
            >>> cluster.append_local_ip_info(("192.168.1.1", 2222))
            >>> cluster.append_remote_ip_info(("192.168.2.1", 2222))
            >>> cluster2 = mslite.LLMClusterInfo(mslite.LLMRole.Prompt, 1)
            >>> cluster2.append_local_ip_info(("192.168.3.1", 2222))
            >>> cluster2.append_remote_ip_info(("192.168.4.2", 2222))
            >>> ret, rets = llm_engine.unlink_clusters((cluster, cluster2))
            >>> if not ret.IsOk():
            >>>    for ret_item in rets:
            >>>        if not ret_item.IsOk():
            >>>            # do something
        """
        if not self.inited_:
            raise RuntimeError(f"LLMEngine is not inited or init failed")
        if not isinstance(clusters, (tuple, list)):
            raise TypeError(f"clusters must be list/tuple of LLMClusterInfo, but got {type(clusters)}.")
        check_isinstance("timeout", timeout, int)
        for i, element in enumerate(clusters):
            if not isinstance(element, LLMClusterInfo):
                raise TypeError(f"clusters element must be LLMClusterInfo, but got {type(element)} at index {i}.")
        clusters_inners = [item.llm_cluster_ for item in clusters]
        ret, rets = self.engine_.unlink_clusters(clusters_inners, timeout)
        if not rets:
            _handle_llm_status(ret, "unlink_clusters", "")
            raise RuntimeError(f"Failed to call unlink_clusters")
        return ret, rets
