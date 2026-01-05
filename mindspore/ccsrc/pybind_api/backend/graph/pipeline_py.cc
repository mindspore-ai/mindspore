/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pybind_api/backend/graph/pipeline_py.h"

#include <memory>
#include <map>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <unordered_map>
#include <functional>

#include "pybind_api/pybind_patch.h"
#include "pybind11/pybind11.h"

#include "ir/anf.h"
#include "ir/tensor.h"
#include "utils/ms_context.h"
#include "utils/compile_config.h"
#include "utils/ir_dump/anf_ir_dump.h"
#include "utils/config_manager.h"
#include "load_mindir/load_model.h"
#include "include/cluster/init.h"
#include "include/utils/tensor_py.h"
#include "include/utils/comm_manager.h"
#include "include/utils/python_adapter.h"
#include "include/frontend/jit/ps/pipeline_interface.h"
#include "include/backend/backend_manager/backend_manager.h"
#include "runtime/hardware_abstract/utils.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"

namespace mindspore {
namespace pipeline {
void ResetOpId() { mindspore::id_generator::reset_id(); }

void ResetOpIdWithOffset() { mindspore::id_generator::reset_id_with_offset(); }

void InitHccl() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->set_param<bool>(MS_CTX_ENABLE_HCCL, true);
#ifdef WITH_BACKEND
  auto backend = ms_context->backend_policy();
  if (backend == "ge") {
    if (!mindspore::distributed::Initialize()) {
      MS_LOG(EXCEPTION) << "InitHccl failed.";
    }
    InitPipeline();
    return;
  }
#endif
  mindspore::python_adapter::set_python_env_flag(true);
  std::string device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (ms_context->backend_policy() == "ms" && device_name == kAscendDevice) {
    if (!mindspore::distributed::Initialize()) {
      MS_LOG(EXCEPTION) << "InitHccl failed.";
    }
  }
}

void InitHccl(std::optional<std::string> url, int64_t timeout, uint32_t world_size, uint32_t node_id,
              distributed::cluster::TCPStoreClientPtr store) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->set_param<bool>(MS_CTX_ENABLE_HCCL, true);
#ifdef WITH_BACKEND
  auto backend = ms_context->backend_policy();
  if (backend == "ge") {
    if (!mindspore::distributed::Initialize(url, timeout, world_size, node_id, store)) {
      MS_LOG(EXCEPTION) << "InitHccl failed.";
    }
    InitPipeline();
    return;
  }
#endif
  mindspore::python_adapter::set_python_env_flag(true);
  std::string device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (ms_context->backend_policy() == "ms" && device_name == kAscendDevice) {
    if (!mindspore::distributed::Initialize(url, timeout, world_size, node_id, store)) {
      MS_LOG(EXCEPTION) << "InitHccl failed.";
    }
  }
}

void FinalizeHccl() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
#ifdef WITH_BACKEND
  auto backend = ms_context->backend_policy();
  if (backend == "ge") {
    return;
  }
#endif
  device::DeviceContextManager::GetInstance().ClearDeviceContexts();
  device::DeviceContextManager::GetInstance().UnloadPlugin();
}

uint32_t GetHcclRankId() {
  uint32_t rank_id = 0;
  bool ret = CommManager::GetInstance().GetRankID("", &rank_id);
  if (!ret) {
    MS_LOG(ERROR) << "Get rank id failed, return rank id " << rank_id << " as default.";
  }
  return rank_id;
}

uint32_t GetHcclRankSize() {
  uint32_t rank_size = 0;
  bool ret = CommManager::GetInstance().GetRankSize("", &rank_size);
  if (!ret) {
    MS_LOG(ERROR) << "Get rank size failed, return rank size " << rank_size << " as default.";
  }
  return rank_size;
}

void BindDeviceCtx() { device::DeviceContextManager::GetInstance().BindDeviceCtx(); }

bool InitExecDatasetVm(const std::string &queue_name, int64_t size, int64_t batch_size,
                       const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                       const std::vector<int64_t> &input_indexes, bool need_run) {
  PROF_START(InitExecDatasetVm);
  MS_LOG(INFO) << "Start InitDataSet Entry";
  mindspore::python_adapter::set_python_env_flag(true);
  ShapeVector int_input_indexes;
  (void)std::transform(input_indexes.begin(), input_indexes.end(), std::back_inserter(int_input_indexes),
                       [](int64_t item) { return static_cast<int64_t>(item); });
  std::vector<ShapeVector> int_shapes;
  (void)std::transform(shapes.begin(), shapes.end(), std::back_inserter(int_shapes),
                       [](const std::vector<int64_t> &item) {
                         ShapeVector vector_item;
                         (void)std::transform(item.begin(), item.end(), std::back_inserter(vector_item),
                                              [](int64_t inner_item) { return static_cast<int64_t>(inner_item); });
                         return vector_item;
                       });
  auto p_init = std::make_shared<Primitive>("InitDataSetQueue");
  p_init->set_attr("queue_name", MakeValue(queue_name));
  p_init->set_attr("size", MakeValue(static_cast<int64_t>(size)));
  p_init->set_attr("batch_size", MakeValue(static_cast<int64_t>(batch_size)));
  p_init->set_attr("types", MakeValue(types));
  p_init->set_attr("shapes", MakeValue(int_shapes));
  p_init->set_attr("input_indexes", MakeValue(int_input_indexes));

  const std::vector<std::string> empty_str_list;
  p_init->set_attr("input_names", MakeValue(empty_str_list));
  p_init->set_attr("output_names", MakeValue(empty_str_list));

  FuncGraphPtr func_graph = std::make_shared<FuncGraph>();
  auto app_init = std::make_shared<CNode>(AnfNodeWeakPtrList({NewValueNode(p_init)}), func_graph);
  func_graph->set_output(app_init);
  auto manager = MakeManager();
  manager->AddFuncGraph(func_graph);

  // AbstractNone indicates there is no output for this apply node.
  auto abstract_none = std::make_shared<abstract::AbstractNone>();
  app_init->set_abstract(abstract_none);
  // Before the graph compiling, need reset the iter num.
  ConfigManager::GetInstance().ResetIterNum();

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  MS_LOG(DEBUG) << "Enable mindRT.";
  context_ptr->set_param<bool>(MS_CTX_ENABLE_MINDRT, true);
  context_ptr->Refresh();

  VectorRef args;
  if (need_run) {
    VectorRef outputs;
    const auto &backend_jit_config = backend::BackendJitConfig::ParseBackendJitConfig();
    auto backend_ret =
      backend::BackendManager::GetInstance().Build(func_graph, backend_jit_config, backend_jit_config.backend);
    backend::BackendManager::GetInstance().Run(backend_ret.first, backend_ret.second, args, &outputs);
  }
  ConfigManager::GetInstance().set_iter_num(queue_name, size);
  PROF_END(InitExecDatasetVm);
  return true;
}

bool InitExecDataset(const std::string &queue_name, int64_t iter_num, int64_t batch_size,
                     const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                     const std::vector<int64_t> &input_indexes, const std::string &, bool need_run) {
  if (UseSimulationApi()) {
    return true;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string name = ms_context->backend_policy();
#ifdef WITH_BACKEND
  if (ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    if (ms_context->get_param<uint32_t>(MS_CTX_TSD_REF) <= 0) {
      InitPipeline();
    }
  }
#endif

  if (name == "ms" || name == "vm" || name == "ge") {
#ifdef WITH_BACKEND
    if (iter_num == -1) {
      iter_num = INT32_MAX;
    }
    bool status = InitExecDatasetVm(queue_name, iter_num, batch_size, types, shapes, input_indexes, need_run);
    return status;
#endif
  }
  return name == "ge" ? true : false;
}

py::bytes PyEncrypt(char *plain_data, size_t plain_len, char *key, size_t key_len, const std::string &enc_mode) {
  size_t encrypt_len;
  auto encrypt_data = mindspore::Encrypt(&encrypt_len, reinterpret_cast<Byte *>(plain_data), plain_len,
                                         reinterpret_cast<Byte *>(key), key_len, enc_mode);
  if (encrypt_data == nullptr) {
    MS_EXCEPTION(ValueError) << "Encrypt failed";
  }
  auto py_encrypt_data = py::bytes(reinterpret_cast<char *>(encrypt_data.get()), encrypt_len);
  return py_encrypt_data;
}

py::bytes PyDecrypt(const std::string &encrypt_data_path, char *key, size_t key_len, const std::string &dec_mode) {
  size_t decrypt_len;
  auto decrypt_data =
    mindspore::Decrypt(&decrypt_len, encrypt_data_path, reinterpret_cast<Byte *>(key), key_len, dec_mode);
  if (decrypt_data == nullptr) {
    MS_LOG(ERROR) << "Decrypt failed";
    return py::none();
  }
  auto py_decrypt_data = py::bytes(reinterpret_cast<char *>(decrypt_data.get()), decrypt_len);
  return py_decrypt_data;
}

py::bytes PyDecryptData(char *model_data, size_t data_size, char *key, size_t key_len, const std::string &dec_mode) {
  size_t decrypt_len;
  auto decrypt_data = mindspore::Decrypt(&decrypt_len, reinterpret_cast<Byte *>(model_data), data_size,
                                         reinterpret_cast<Byte *>(key), key_len, dec_mode);
  if (decrypt_data == nullptr) {
    MS_LOG(ERROR) << "Decrypt failed";
    return py::none();
  }
  auto py_decrypt_data = py::bytes(reinterpret_cast<char *>(decrypt_data.get()), decrypt_len);
  return py_decrypt_data;
}

bool PyIsCipherFile(const std::string &file_path) { return mindspore::IsCipherFile(file_path); }

void SwapCache(const py::object &host_, const py::object &device_, const py::object &block_mapping_,
               const bool &is_device_to_host) {
  tensor::TensorPtr block_mapping = tensor::ConvertToTensor(block_mapping_);
  MS_EXCEPTION_IF_NULL(block_mapping);
  auto block_mapping_shape = block_mapping->shape();
  const size_t num_two = 2;
  if (block_mapping_shape.size() != num_two) {
    MS_LOG_EXCEPTION << "The shape size of Cache input mapping tensor should be 2, but got: "
                     << block_mapping_shape.size();
  }
  if (block_mapping_shape[kIndex1] != num_two) {
    MS_LOG_EXCEPTION << "The second dim of CacheKernel input mapping tensor should be 2, but got: "
                     << block_mapping_shape[0];
  }

  tensor::TensorPtr device = tensor::ConvertToTensor(device_);
  MS_EXCEPTION_IF_NULL(device);
  auto in_shape = device->shape();
  tensor::TensorPtr host = tensor::ConvertToTensor(host_);
  MS_EXCEPTION_IF_NULL(host);
  auto type_byte = GetTypeByte(TypeIdToType(host->data_type()));
  size_t block_size_in_bytes = LongToSize(
    std::accumulate(in_shape.begin() + kIndex1, in_shape.end(), SizeToLong(type_byte), std::multiplies<int64_t>()));

  uint8_t *host_ptr = reinterpret_cast<uint8_t *>(host->data_c());
  MS_EXCEPTION_IF_NULL(host_ptr);
  auto device_addr = std::dynamic_pointer_cast<device::DeviceAddress>(device->device_address());
  MS_EXCEPTION_IF_NULL(device_addr);
  uint8_t *device_ptr = reinterpret_cast<uint8_t *>(const_cast<void *>(device_addr->GetPtr()));
  MS_EXCEPTION_IF_NULL(device_ptr);

  auto block_mapping_data = reinterpret_cast<int64_t *>(block_mapping->data_c());
  device::DeviceContextKey host_key = {device_addr->GetDeviceType(), device_addr->device_id()};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);

  host_context->device_res_manager_->SyncAllStreams();
  for (size_t i = 0; i < LongToSize(block_mapping_shape[0]); i++) {
    int64_t src_block_num = block_mapping_data[num_two * i];
    int64_t dst_block_num = block_mapping_data[num_two * i + kIndex1];
    size_t src_block_offset = LongToSize(src_block_num) * block_size_in_bytes;
    size_t dst_block_offset = LongToSize(dst_block_num) * block_size_in_bytes;
    if (is_device_to_host) {
      host_context->device_res_manager_->Copy(host_ptr + dst_block_offset, device_ptr + src_block_offset,
                                              block_size_in_bytes, device::CopyType::kD2H, device_addr->stream_id());
    } else {
      host_context->device_res_manager_->Copy(device_ptr + dst_block_offset, host_ptr + src_block_offset,
                                              block_size_in_bytes, device::CopyType::kH2D, device_addr->stream_id());
    }
  }
}
}  // namespace pipeline
}  // namespace mindspore
