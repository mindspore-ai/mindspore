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
import os
import threading
from enum import Enum
from typing import Union, List, Tuple, Dict
from mindspore_lite._checkparam import check_isinstance
from mindspore_lite.tensor import Tensor
from mindspore_lite.lib._c_lite_wrapper import LLMEngine_, LLMReq_, LLMRole_, StatusCode, LLMClusterInfo_

__all__ = ['LLMReq', 'LLMEngineStatus', 'LLMRole', 'LLMEngine']


class LLMReq:
    """
    LLMEngine request, used to represent a multi round inference task.
    """

    def __init__(self, prompt_cluster_id: int, req_id: int, prompt_length: int):
        check_isinstance("prompt_cluster_id", prompt_cluster_id, int)
        check_isinstance("req_id", req_id, int)
        check_isinstance("prompt_length", prompt_length, int)
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
        check_isinstance("req_id", req_id, int)
        self.llm_request_.req_id = req_id

    @property
    def prompt_length(self):
        """Set prompt length of this inference task"""
        return self.llm_request_.prompt_length

    @prompt_length.setter
    def prompt_length(self, prompt_length: int):
        """Get prompt length of this inference task"""
        check_isinstance("prompt_length", prompt_length, int)
        self.llm_request_.prompt_length = prompt_length

    @property
    def prompt_cluster_id(self):
        """Get prompt cluster id of this inference task in LLMEngine"""
        return self.llm_request_.prompt_cluster_id

    @prompt_cluster_id.setter
    def prompt_cluster_id(self, prompt_cluster_id: int):
        """Set prompt cluster id of this inference task in LLMEngine"""
        check_isinstance("prompt_cluster_id", prompt_cluster_id, int)
        self.llm_request_.prompt_cluster_id = prompt_cluster_id

    @property
    def decoder_cluster_id(self):
        """Get decoder cluster id of this inference task in LLMEngine"""
        return self.llm_request_.decoder_cluster_id

    @decoder_cluster_id.setter
    def decoder_cluster_id(self, decoder_cluster_id: int):
        """Set decoder cluster id of this inference task in LLMEngine"""
        check_isinstance("decoder_cluster_id", decoder_cluster_id, int)
        self.llm_request_.decoder_cluster_id = decoder_cluster_id

    @property
    def prefix_id(self):
        """Get decoder prefix id of this inference task in LLMEngine"""
        return self.llm_request_.prefix_id

    @prefix_id.setter
    def prefix_id(self, prefix_id: int):
        """Set decoder prefix id of this inference task in LLMEngine"""
        check_isinstance("prefix_id", prefix_id, int)
        self.llm_request_.prefix_id = prefix_id


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


class LLMRole(Enum):
    """
    Role of LLMEngine. When LLMEngine accelerates inference performance through KVCache, the generation process includes
    one full inference and n incremental inference, involving both full and incremental models. When the full and
    incremental models are deployed on different nodes, the role of the node where the full models are located is
    ``Prompt``, and the role of the node where the incremental models are located is ``Decoder``.
    """
    Prompt = 0
    Decoder = 1


class LLMKVCacheNotExist(RuntimeError):
    """
    Key & Value cache does not exist in Prompt cluster specified by parameter LLMReq.prompt_cluster_id, and the
    LLM request may have been released in Prompt cluster by calling method LLMEngine.complete_request.
    Raised in LLMEngine.predict.
    """


class LLMWaitProcessTimeOut(RuntimeError):
    """
    Request waiting for processing timed out. Raised in LLMEngine.predict.
    """


class LLMRepeatRequest(RuntimeError):
    """
    Request repeated . Raised in LLMEngine.predict.
    """


class LLMRequestAlreadyCompleted(RuntimeError):
    """
    Request has already completed. Raised in LLMEngine.predict.
    """


class LLMEngineFinalized(RuntimeError):
    """
    LLMEngine has finalized. Raised in LLMEngine.predict.
    """


class LLMParamInvalid(RuntimeError):
    """
    Parameters invalid. Raised in LLMEngine.predict.
    """


class LLMClusterInfo:
    """
    The `LLMClusterInfo` class defines a MindSpore Lite's LLMEngine cluster, used to link and unlink clusters.

    Args:
        remote_role (LLMRole): Role of remote LLMEngine object.
        remote_cluster_id (int): Cluster id of remote LLMEngine object.

    Raises:
        TypeError: `role` is not a LLMRole.
        TypeError: `cluster_id` is not an int.

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
        check_isinstance("remote_cluster_id", remote_cluster_id, int)
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
        check_isinstance("remote_role_type", remote_role, LLMRole)
        remote_role_type_int = 0 if remote_role == LLMRole.Prompt else 1  # 0: Prompt, 1: Decoder
        self.llm_cluster_.remote_role_type = remote_role_type_int

    @property
    def remote_cluster_id(self):
        """Get remote cluster id of this LLMClusterInfo object"""
        return self.llm_cluster_.remote_cluster_id

    @remote_cluster_id.setter
    def remote_cluster_id(self, remote_cluster_id):
        """Set remote cluster id of this LLMClusterInfo object"""
        check_isinstance("remote_cluster_id", remote_cluster_id, int)
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
                import socket
                ip = socket.inet_aton(ip)
                ip = int.from_bytes(ip, byteorder='big')
            except OSError:
                raise ValueError(
                    f"address must be in format of ('xxx.xxx.xxx.xxx', xxx) or (xxx, xxx), but got {address}")
        return ip, port


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
        >>> llm_engine.init(model_paths, options)
        >>> llm_req = mslite.LLMReq(llm_engine.cluster_id, mslite.LLMReq.next_req_id(), prompt_length=1024)
        >>> inputs = [mslite.Tensor(np_input) for np_input in np_inputs]
        >>> outputs = llm_req.predit(inputs)
        >>> for output in outputs:
        >>>    print(f"output is {output.get_data_to_numpy()}")
        >>> llm_req.complete()
    """

    def __init__(self, role: LLMRole, cluster_id: int, batch_mode="auto"):
        check_isinstance("role", role, LLMRole)
        check_isinstance("cluster_id", cluster_id, int)
        check_isinstance("batch_mode", batch_mode, str)
        if batch_mode != "auto" and batch_mode != "manual":
            raise RuntimeError(f"batch_mode should be str \"auto\" or \"manual\", but got {batch_mode}")
        if role != LLMRole.Decoder and batch_mode != "auto":
            raise RuntimeError(f"batch_mode should be \"auto\" when role is not Decoder, but got {batch_mode}")
        self.role_ = role
        self.cluster_id_ = cluster_id
        self.engine_ = None
        self.batch_mode_ = batch_mode

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

    def init(self, model_paths: Union[Tuple[str], List[str]], options: Dict[str, str],
             postprocess_model_path=None):
        """
        Init LLMEngine.

        Args:
            model_paths (Union[Tuple[str], List[str]]): List or tuple of model path.
            options (Dict[str, str]): Other init options of this LLMEngine object.
            postprocess_model_path (Union[str, None]): Postprocess model path, default None.

        Raises:
            TypeError: `model_paths` is not a list and tuple.
            TypeError: `model_paths` is a list or tuple, but the elements are not str.
            TypeError: `options` is not a dict.
            RuntimeError: init LLMEngine failed.
        """
        if not isinstance(model_paths, (list, tuple)):
            raise TypeError(f"model_paths must be tuple/list of str, but got item {type(model_paths)}.")
        for i, model_path in enumerate(model_paths):
            if not isinstance(model_path, str):
                raise TypeError(f"model_paths element must be str, but got {type(model_path)} at index {i}.")
            if not os.path.exists(model_path):
                raise RuntimeError(f"Failed to init LLMEngine, model path {model_path} at index {i} does not exist!")
        check_isinstance("options", options, dict)
        if postprocess_model_path is not None:
            if not isinstance(postprocess_model_path, str):
                raise TypeError(
                    f"postprocess_model_path must be None or str, but got item {type(postprocess_model_path)}.")
        else:
            postprocess_model_path = ""

        self.engine_ = LLMEngine_()
        role_inner = LLMRole_.Prompt if self.role == LLMRole.Prompt else LLMRole_.Decoder
        ret = self.engine_.init(model_paths, role_inner, self.cluster_id, options, self.batch_mode,
                                postprocess_model_path)
        status_code = ret.StatusCode()
        if status_code == StatusCode.kLiteParamInvalid:
            raise LLMParamInvalid("Parameters invalid")
        if not ret.IsOk():
            role_str = 'Prompt' if self.role == LLMRole.Prompt else 'Decoder'
            raise RuntimeError(f"Failed to init LLMEngine, model paths {model_paths}, role {role_str},"
                               f" cluster id {self.cluster_id}, options {options}")

    def finalize(self):
        """
        Finalize LLMEngine.
        """
        if not self.engine_:
            print(f"LLMEngine is not inited or init failed", flush=True)
            return
        self.engine_.finalize()

    @staticmethod
    def _handle_llm_status(status, func_name, other_info):
        """Handle LLM error code"""
        status_code = status.StatusCode()
        if status_code == StatusCode.kLiteLLMWaitProcessTimeOut:
            raise LLMWaitProcessTimeOut(f"{func_name} failed: Waiting for processing timeout, {other_info}")
        if status_code == StatusCode.kLiteLLMKVCacheNotExist:
            raise LLMKVCacheNotExist(f"{func_name} failed: KV Cache not exist, {other_info}.")
        if status_code == StatusCode.kLiteLLMRepeatRequest:
            raise LLMRepeatRequest(f"{func_name} failed: Repeat request, {other_info}.")
        if status_code == StatusCode.kLiteLLMRequestAlreadyCompleted:
            raise LLMRequestAlreadyCompleted(f"{func_name} failed: Request has already completed, {other_info}.")
        if status_code == StatusCode.kLiteLLMEngineFinalized:
            raise LLMEngineFinalized(f"{func_name} failed: LLMEngine has finalized, {other_info}.")
        if status_code == StatusCode.kLiteParamInvalid:
            raise LLMParamInvalid(f"{func_name} failed: Parameters invalid, {other_info}.")
        if status_code != StatusCode.kSuccess:
            raise RuntimeError(f"{func_name} failed, {other_info}.")

    @staticmethod
    def _llm_req_str(llm_req):
        return "{" + f"llm_req: {llm_req.req_id}, prompt_cluster_id: {llm_req.prompt_cluster_id}, " \
                     f"decoder_cluster_id: {llm_req.decoder_cluster_id}, prefix_id: {llm_req.prefix_id}, " \
                     f"prompt_length: {llm_req.prompt_length}" + "}"

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
        if not self.engine_:
            raise RuntimeError(f"LLMEngine is not inited or init failed")
        if not isinstance(inputs, (tuple, list)):
            raise TypeError(f"inputs must be list/tuple of Tensor, but got {type(inputs)}.")
        if not isinstance(llm_req, (list, tuple, LLMReq)):
            raise TypeError(f"llm_req must be instance of LLMReq or list/tuple of LLMReq, but got {type(llm_req)}.")
        if self.batch_mode == "manual":
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
        if self.batch_mode == "manual":
            llm_req_list = [item.llm_request_ for item in llm_req]
            outputs, status = self.engine_.predict_batch(llm_req_list, _inputs)
        else:
            outputs, status = self.engine_.predict(llm_req.llm_request_, _inputs)

        if isinstance(llm_req, LLMReq):
            req_infos = LLMEngine._llm_req_str(llm_req)
        else:
            req_infos = [LLMEngine._llm_req_str(llm) for llm in llm_req]

        input_infos = [(item.shape, item.dtype) for item in inputs]
        info = f"llm_req {req_infos}, inputs {input_infos}"
        LLMEngine._handle_llm_status(status, "predict", info)
        if not outputs:
            raise RuntimeError(f"predict failed, {info}.")
        predict_outputs = [Tensor(output) for output in outputs]
        return predict_outputs

    def complete_request(self, llm_req: LLMReq):
        """
        Complete inference request.

        Args:
            llm_req (LLMReq): Request of LLMEngine.

        Raises:
            TypeError: `llm_req` is not a LLMReq.
            RuntimeError: this LLMEngine object has not been inited.
        """
        if not self.engine_:
            raise RuntimeError(f"LLMEngine is not inited or init failed")
        check_isinstance("llm_req", llm_req, LLMReq)
        self.engine_.complete_request(llm_req.llm_request_)

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
        if not self.engine_:
            raise RuntimeError(f"LLMEngine is not inited or init failed")
        if self.batch_mode != "manual":
            raise RuntimeError(f"LLMEngine.pull_kv is only support when batch_mode is \"manual\"")
        check_isinstance("llm_req", llm_req, LLMReq)
        status = self.engine_.pull_kv(llm_req.llm_request_)
        LLMEngine._handle_llm_status(status, "pull_kv", "llm_req " + LLMEngine._llm_req_str(llm_req))

    def merge_kv(self, llm_req: LLMReq, batch_index: int):
        """
        For Decoder LLMEngine, merge KVCache of LLMReq specified by `llm_req.req_id` into `batch_index` slot.
        Args:
            llm_req (LLMReq): Request of LLMEngine.

        Raises:
            TypeError: `llm_req` is not a LLMReq.
            RuntimeError: this LLMEngine object has not been inited.
            RuntimeError: Failed to merge KVCache.
            LLMParamInvalid: Parameters invalid.
        """
        if not self.engine_:
            raise RuntimeError(f"LLMEngine is not inited or init failed")
        if self.batch_mode != "manual":
            raise RuntimeError(f"LLMEngine.pull_kv is only support when batch_mode is \"manual\"")
        check_isinstance("llm_req", llm_req, LLMReq)
        check_isinstance("batch_index", batch_index, int)
        status = self.engine_.merge_kv(llm_req.llm_request_, batch_index)
        LLMEngine._handle_llm_status(status, "merge_kv", "llm_req " + LLMEngine._llm_req_str(llm_req))

    def fetch_status(self):
        """
        Get LLMEngine status.

        Returns:
            LLMEngineStatus, LLMEngine status.

        Raises:
            RuntimeError: this LLMEngine object has not been inited.
        """
        if not self.engine_:
            raise RuntimeError(f"LLMEngine is not inited or init failed")
        status = self.engine_.fetch_status()
        return LLMEngineStatus(status)

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
        if not self.engine_:
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
        ret = self.engine_.preload_prompt_prefix(llm_req.llm_request_, _inputs)
        LLMEngine._handle_llm_status(ret, "preload_prompt_prefix", "llm_req " + LLMEngine._llm_req_str(llm_req))

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
        if not self.engine_:
            raise RuntimeError(f"LLMEngine is not inited or init failed")
        check_isinstance("llm_req", llm_req, LLMReq)
        ret = self.engine_.release_prompt_prefix(llm_req.llm_request_)
        LLMEngine._handle_llm_status(ret, "release_prompt_prefix", "llm_req " + LLMEngine._llm_req_str(llm_req))

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
        if not self.engine_:
            raise RuntimeError(f"LLMEngine is not inited or init failed")
        if not isinstance(clusters, (tuple, list)):
            raise TypeError(f"clusters must be list/tuple of LLMClusterInfo, but got {type(clusters)}.")
        for i, element in enumerate(clusters):
            if not isinstance(element, LLMClusterInfo):
                raise TypeError(f"clusters element must be LLMClusterInfo, but got {type(element)} at index {i}.")
        clusters_inners = [item.llm_cluster_ for item in clusters]
        ret, rets = self.engine_.link_clusters(clusters_inners, timeout)
        status_code = ret.StatusCode()
        if status_code == StatusCode.kLiteParamInvalid:
            raise LLMParamInvalid("Parameters invalid")
        if not rets:
            raise RuntimeError(f"Failed to call link_clusters")
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
        if not self.engine_:
            raise RuntimeError(f"LLMEngine is not inited or init failed")
        if not isinstance(clusters, (tuple, list)):
            raise TypeError(f"clusters must be list/tuple of LLMClusterInfo, but got {type(clusters)}.")
        for i, element in enumerate(clusters):
            if not isinstance(element, LLMClusterInfo):
                raise TypeError(f"clusters element must be LLMClusterInfo, but got {type(element)} at index {i}.")
        clusters_inners = [item.llm_cluster_ for item in clusters]
        ret, rets = self.engine_.unlink_clusters(clusters_inners, timeout)
        status_code = ret.StatusCode()
        if status_code == StatusCode.kLiteParamInvalid:
            raise LLMParamInvalid("Parameters invalid")
        if not rets:
            raise RuntimeError(f"Failed to call unlink_clusters")
        return ret, rets
