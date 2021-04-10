/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ps/ps_cache/ascend/ascend_ps_cache.h"
#include <google/protobuf/text_format.h>
#include <string>
#include <vector>
#include <memory>
#include "ps/ps_cache/ps_cache_factory.h"
#include "runtime/device/ascend/ascend_memory_pool.h"
#include "backend/kernel_compiler/aicpu/aicpu_kernel_mod.h"
#include "utils/ms_context.h"
#include "proto/tensor.pb.h"
#include "proto/tensor_shape.pb.h"
#include "proto/attr.pb.h"
#include "proto/node_def.pb.h"

using mindspore::kernel::Address;
using AddressPtr = std::shared_ptr<Address>;
using AddressPtrList = std::vector<AddressPtr>;

namespace mindspore {
namespace ps {
namespace ascend {
MS_REG_PS_CACHE(kAscendDevice, AscendPsCache);
namespace {
bool SetProtoInputs(const std::vector<std::vector<size_t>> &data_shape, const std::vector<TypeId> &data_type,
                    mindspore::NodeDef *proto) {
  MS_ERROR_IF_NULL(proto);
  if (data_shape.size() != data_type.size()) {
    MS_LOG(ERROR) << "The size of data shape is not equal to the size of data type.";
    return false;
  }
  for (size_t input_index = 0; input_index < data_shape.size(); input_index++) {
    ::mindspore::Tensor *proto_inputs = proto->add_inputs();
    MS_ERROR_IF_NULL(proto_inputs);
    auto input_shape = data_shape[input_index];
    mindspore::TensorShape *tensorShape = proto_inputs->mutable_tensor_shape();
    MS_ERROR_IF_NULL(tensorShape);
    for (auto item : input_shape) {
      mindspore::TensorShape_Dim *dim = tensorShape->add_dim();
      MS_ERROR_IF_NULL(dim);
      dim->set_size((::google::protobuf::int64)item);
    }
    auto input_type = kernel::AicpuOpUtil::MsTypeToProtoType(data_type[input_index]);
    proto_inputs->set_tensor_type(input_type);
    proto_inputs->set_mem_device("HBM");
  }
  return true;
}

bool SetProtoOutputs(const std::vector<std::vector<size_t>> &data_shape, const std::vector<TypeId> &data_type,
                     mindspore::NodeDef *proto) {
  MS_ERROR_IF_NULL(proto);
  if (data_shape.size() != data_type.size()) {
    MS_LOG(ERROR) << "The size of data shape is not equal to the size of data type.";
    return false;
  }
  for (size_t output_index = 0; output_index < data_shape.size(); output_index++) {
    ::mindspore::Tensor *proto_outputs = proto->add_outputs();
    MS_ERROR_IF_NULL(proto_outputs);
    auto output_shape = data_shape[output_index];
    mindspore::TensorShape *tensorShape = proto_outputs->mutable_tensor_shape();
    MS_ERROR_IF_NULL(tensorShape);
    for (auto item : output_shape) {
      mindspore::TensorShape_Dim *dim = tensorShape->add_dim();
      MS_ERROR_IF_NULL(dim);
      dim->set_size((::google::protobuf::int64)item);
    }
    auto output_type = kernel::AicpuOpUtil::MsTypeToProtoType(data_type[output_index]);
    proto_outputs->set_tensor_type(output_type);
    proto_outputs->set_mem_device("HBM");
  }
  return true;
}

bool SetNodedefProto(const std::shared_ptr<KernelNodeInfo> &op_info,
                     const std::shared_ptr<kernel::AicpuOpKernelMod> &kernel_mod_ptr) {
  MS_ERROR_IF_NULL(op_info);
  MS_ERROR_IF_NULL(kernel_mod_ptr);
  mindspore::NodeDef proto;
  proto.set_op(op_info->op_name_);
  RETURN_IF_FALSE(SetProtoInputs(op_info->input_data_shape_, op_info->input_data_type_, &proto));
  RETURN_IF_FALSE(SetProtoOutputs(op_info->output_data_shape_, op_info->output_data_type_, &proto));
  std::string nodeDefStr;
  if (!proto.SerializeToString(&nodeDefStr)) {
    MS_LOG(ERROR) << "Serialize nodeDef to string failed.";
    return false;
  }
  MS_LOG(DEBUG) << "Set node def proto, node name:" << op_info->op_name_;
  kernel_mod_ptr->SetNodeDef(nodeDefStr);
  return true;
}
}  // namespace

bool AscendPsCache::InitDevice(uint32_t device_id, const void *context) {
  MS_ERROR_IF_NULL(context);
  auto ret = rtSetDevice(device_id);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rtSetDevice, ret[" << ret << "]";
    return false;
  }
  auto rt_context = const_cast<rtContext_t>(context);
  ret = rtCtxSetCurrent(rt_context);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rtCtxSetCurrent, ret[" << ret << "]";
    return false;
  }
  ret = rtStreamCreate(&stream_, 0);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rtStreamCreate, ret[" << ret << "]";
    return false;
  }
  return true;
}

void *AscendPsCache::MallocMemory(size_t size) {
  return device::ascend::AscendMemoryPool::GetInstance().AllocTensorMem(size);
}

bool AscendPsCache::MallocConstantMemory(size_t cache_vocab_size) {
  offset_addr_ = reinterpret_cast<int *>(device::ascend::AscendMemoryPool::GetInstance().AllocTensorMem(sizeof(int)));
  MS_ERROR_IF_NULL(offset_addr_);
  rtMemset(offset_addr_, sizeof(int), 0, sizeof(int));
  cache_vocab_size_addr_ =
    reinterpret_cast<int *>(device::ascend::AscendMemoryPool::GetInstance().AllocTensorMem(sizeof(int)));
  MS_ERROR_IF_NULL(cache_vocab_size_addr_);
  int copy_value = SizeToInt(cache_vocab_size);
  if (!CopyHostMemToDevice(cache_vocab_size_addr_, &copy_value, sizeof(int))) {
    return false;
  }
  return SynchronizeStream();
}

bool AscendPsCache::RecordEvent() {
  event_.reset(new rtEvent_t());
  auto ret = rtEventCreate(&(*event_));
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Create event failed";
    return false;
  }
  ret = rtEventRecord(*event_, stream_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Record event failed";
    return false;
  }
  return true;
}

bool AscendPsCache::SynchronizeEvent() {
  auto ret = rtEventSynchronize(*event_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "tEventSynchronize failed";
    return false;
  }
  ret = rtEventDestroy(*event_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "rtEventDestroy failed";
    return false;
  }
  return true;
}

bool AscendPsCache::SynchronizeStream() {
  auto ret = rtStreamSynchronize(stream_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "rtStreamSynchronize failed";
    return false;
  }
  return true;
}

bool AscendPsCache::CopyHostMemToDevice(void *dst, const void *src, size_t size) {
  MS_ERROR_IF_NULL(dst);
  MS_ERROR_IF_NULL(src);
  auto ret = rtMemcpyAsync(dst, size, src, size, RT_MEMCPY_HOST_TO_DEVICE, stream_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "rtMemcpyAsync failed, the error num is:" << ret;
    return false;
  }
  return true;
}

bool AscendPsCache::CopyDeviceMemToHost(void *dst, const void *src, size_t size) {
  MS_ERROR_IF_NULL(dst);
  MS_ERROR_IF_NULL(src);
  auto ret = rtMemcpyAsync(dst, size, src, size, RT_MEMCPY_DEVICE_TO_HOST, stream_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "rtMemcpyAsync failed, the error num is:" << ret;
    return false;
  }
  return true;
}

bool AscendPsCache::HashSwapOut(void *hash_table_addr, void *swap_out_value_addr, void *swap_out_index_addr,
                                size_t cache_vocab_size, size_t embedding_size, size_t swap_out_size) {
  MS_ERROR_IF_NULL(hash_table_addr);
  MS_ERROR_IF_NULL(swap_out_value_addr);
  MS_ERROR_IF_NULL(swap_out_index_addr);
  auto hash_swap_out_mod = std::make_shared<kernel::AicpuOpKernelMod>();
  MS_ERROR_IF_NULL(hash_swap_out_mod);
  hash_swap_out_mod->SetNodeName(kEmbeddingLookupOpName);
  std::vector<std::vector<size_t>> input_shape;
  std::vector<std::vector<size_t>> output_shape;
  std::vector<TypeId> input_type = {TypeId::kNumberTypeFloat32, TypeId::kNumberTypeInt32, TypeId::kNumberTypeInt32};
  std::vector<TypeId> output_type = {TypeId::kNumberTypeFloat32};
  input_shape.push_back({cache_vocab_size, embedding_size});
  input_shape.push_back({swap_out_size});
  input_shape.push_back({1});
  output_shape.push_back({swap_out_size, embedding_size});
  auto op_info =
    std::make_shared<KernelNodeInfo>(kEmbeddingLookupOpName, input_shape, input_type, output_shape, output_type);
  RETURN_IF_FALSE(SetNodedefProto(op_info, hash_swap_out_mod));

  AddressPtrList kernel_inputs;
  AddressPtrList kernel_outputs = {
    std::make_shared<Address>(swap_out_value_addr, swap_out_size * embedding_size * sizeof(float))};
  AddressPtrList kernel_workspaces;
  kernel_inputs.push_back(
    std::make_shared<Address>(hash_table_addr, cache_vocab_size * embedding_size * sizeof(float)));
  kernel_inputs.push_back(std::make_shared<Address>(swap_out_index_addr, swap_out_size * sizeof(int)));
  kernel_inputs.push_back(std::make_shared<Address>(offset_addr_, sizeof(int)));
  auto ret = hash_swap_out_mod->Launch(kernel_inputs, kernel_workspaces, kernel_outputs, stream_);
  if (!ret) {
    MS_LOG(ERROR) << "Hash swap out launch failed.";
    return false;
  }
  return true;
}

bool AscendPsCache::HashSwapIn(void *hash_table_addr, void *swap_in_value_addr, void *swap_in_index_addr,
                               size_t cache_vocab_size, size_t embedding_size, size_t swap_in_size) {
  MS_ERROR_IF_NULL(hash_table_addr);
  MS_ERROR_IF_NULL(swap_in_value_addr);
  MS_ERROR_IF_NULL(swap_in_index_addr);
  auto hash_swap_in_mod = std::make_shared<kernel::AicpuOpKernelMod>();
  MS_ERROR_IF_NULL(hash_swap_in_mod);
  hash_swap_in_mod->SetNodeName(kernel::kUpdateCache);
  std::vector<std::vector<size_t>> input_shape;
  std::vector<std::vector<size_t>> output_shape;
  std::vector<TypeId> input_type = {TypeId::kNumberTypeFloat32, TypeId::kNumberTypeInt32, TypeId::kNumberTypeFloat32,
                                    TypeId::kNumberTypeInt32};
  std::vector<TypeId> output_type = {TypeId::kNumberTypeInt32};
  input_shape.push_back({cache_vocab_size, embedding_size});
  input_shape.push_back({swap_in_size});
  input_shape.push_back({swap_in_size, embedding_size});
  input_shape.push_back({1});
  output_shape.push_back({1});
  auto op_info =
    std::make_shared<KernelNodeInfo>(kernel::kUpdateCache, input_shape, input_type, output_shape, output_type);
  SetNodedefProto(op_info, hash_swap_in_mod);

  AddressPtrList kernel_inputs;
  AddressPtrList kernel_outputs;
  AddressPtrList kernel_workspaces;
  kernel_inputs.push_back(
    std::make_shared<Address>(hash_table_addr, cache_vocab_size * embedding_size * sizeof(float)));
  kernel_inputs.push_back(std::make_shared<Address>(swap_in_index_addr, swap_in_size * sizeof(int)));
  kernel_inputs.push_back(std::make_shared<Address>(swap_in_value_addr, swap_in_size * embedding_size * sizeof(float)));
  kernel_inputs.push_back(std::make_shared<Address>(cache_vocab_size_addr_, sizeof(int)));
  // The output of updateCache kernel is required but not useful, so any address can be assigned.
  kernel_outputs.push_back(std::make_shared<Address>(offset_addr_, sizeof(int)));
  auto ret = hash_swap_in_mod->Launch(kernel_inputs, kernel_workspaces, kernel_outputs, stream_);
  if (!ret) {
    MS_LOG(ERROR) << "Hash swap in launch failed.";
    return false;
  }
  return true;
}
}  // namespace ascend
}  // namespace ps
}  // namespace mindspore
