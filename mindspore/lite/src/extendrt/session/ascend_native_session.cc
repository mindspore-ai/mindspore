/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "extendrt/session/ascend_native_session.h"
#include <vector>
#include <algorithm>
#include <string>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include "extendrt/utils/tensor_utils.h"
#include "extendrt/session/factory.h"
#include "extendrt/utils/tensor_default_impl.h"
#include "extendrt/delegate/ascend_native/delegate.h"
#include "src/common/log_adapter.h"
#include "src/litert/cxx_api/converters.h"
#include "ir/graph_utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/utils.h"
#include "src/train/opt_allocator.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"

namespace mindspore {
Status AscendNativeSession::MoveDataFromHostToDevice(void *sd, bool s_fp16, void *dd, bool d_fp16, size_t elem_num) {
  if (s_fp16) {
    if (d_fp16) {
      ascend_native::CopyHostFp16ToDeviceFp16(sd, &dd, elem_num, ascend_native_stream_);
    } else {
      ascend_native::CopyHostFp16ToDeviceFp32(sd, &dd, elem_num, ascend_native_stream_);
    }
  } else {
    if (d_fp16) {
      ascend_native::CopyHostFp32ToDeviceFp16(sd, &dd, elem_num, ascend_native_stream_);
    } else {
      ascend_native::CopyHostFp32ToDeviceFp32(sd, &dd, elem_num, ascend_native_stream_);
    }
  }
  return kSuccess;
}

Status AscendNativeSession::MoveDataFromDeviceToHost(void *sd, bool s_fp16, void *dd, bool d_fp16, size_t elem_num) {
  if (sd == nullptr) {
    MS_LOG(ERROR) << "source pointer is null";
    return kLiteNullptr;
  }
  if (dd == nullptr) {
    MS_LOG(ERROR) << "destination pointer is null";
    return kLiteNullptr;
  }
  if (s_fp16) {
    if (d_fp16) {
      ascend_native::CopyDeviceFp16ToHostFp16(sd, dd, elem_num, ascend_native_stream_);
    } else {
      ascend_native::CopyDeviceFp16ToHostFp32(sd, dd, elem_num, ascend_native_stream_);
    }
  } else {
    if (d_fp16) {
      ascend_native::CopyDeviceFp32ToHostFp16(sd, dd, elem_num, ascend_native_stream_);
    } else {
      ascend_native::CopyDeviceFp32ToHostFp32(sd, dd, elem_num, ascend_native_stream_);
    }
  }
  return kSuccess;
}

void *AscendNativeSession::MallocDevice(size_t size) {
  MS_CHECK_GT(size, 0, nullptr);
  auto device_data = ascend_native::MallocDevice(size, ascend_native_stream_);
  if (device_data == nullptr) {
    MS_LOG(ERROR) << "fail to allocate " << size << " bytes of device data";
  }
  return device_data;
}

void AscendNativeSession::FreeDevice(void *ptr) { ascend_native::FreeDevice(ptr, ascend_native_stream_); }

void AscendNativeSession::InitializeTensorRefrenceCnt() {
  for (auto &kernel : kernels_) {
    for (auto tensor : kernel->in_tensors()) {
      if (tensor->category() == lite::VAR || tensor->category() == lite::GRAPH_INPUT) {
        auto ref_count = tensor->init_ref_count();
        tensor->set_init_ref_count(ref_count + 1);
      }
    }
  }
}

Status AscendNativeSession::AllocTensors() {
  OptAllocator allocator;
  std::unordered_map<kernel::InferTensor *, int> ref_count;
  std::unordered_map<kernel::InferTensor *, size_t> offset_map;
  auto device_elem_size = context_->IsNpuFloat16Enabled() ? C2NUM : C4NUM;
  for (auto &kernel : kernels_) {
    // malloc Graph inputs
    for (auto &tensor : kernel->in_tensors()) {
      // TBD - when is allowed to free input ??
      if (tensor->category() == lite::GRAPH_INPUT) {
        size_t elem_num = tensor->ElementsNum();
        if (offset_map.find(tensor) == offset_map.end()) {
          size_t offset = allocator.Malloc(device_elem_size * elem_num);
          offset_map[tensor] = offset;
          ref_count[tensor] = tensor->init_ref_count();
        }
      }
    }
    // malloc output tensors
    for (auto &tensor : kernel->out_tensors()) {
      size_t elem_num = tensor->ElementsNum();
      size_t offset = allocator.Malloc(device_elem_size * elem_num);
      offset_map[tensor] = offset;
      ref_count[tensor] = tensor->init_ref_count();
    }
    // free according to reference counter
    for (auto &tensor : kernel->in_tensors()) {
      if (tensor->category() == lite::Category::VAR) {
        int count = ref_count[tensor] - 1;
        ref_count[tensor] = count;
        if (count == 0) {
          allocator.Free(offset_map[tensor]);
        }
      }
    }
  }
  // Set Tensor data
  mem_size_ = allocator.total_size();
  if (mem_size_ > 0) {
    memory_base_addr_ = malloc(mem_size_);
    if (memory_base_addr_ == nullptr) {
      MS_LOG(EXCEPTION) << "Allocation of " << mem_size_ << "B on device failed";
      return kMDOutOfMemory;
    }
    for (auto &kernel : kernels_) {
      // allocate graph inputs
      for (auto &tensor : kernel->in_tensors()) {
        if (tensor->category() == lite::Category::GRAPH_INPUT) {
          auto it = offset_map.find(tensor);
          if (it != offset_map.end()) {
            tensor->set_data(reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(memory_base_addr_) + it->second));
          }
        }
      }
      // allocate activation
      for (auto &tensor : kernel->out_tensors()) {
        auto it = offset_map.find(tensor);
        if (it != offset_map.end()) {
          tensor->set_data(reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(memory_base_addr_) + it->second));
        }
      }
    }
  }
  return kSuccess;
}

Status AscendNativeSession::AllocateGraphTensors() {
  if (memory_base_addr_ == nullptr) {
    InitializeTensorRefrenceCnt();
  } else {
    free(memory_base_addr_);
    memory_base_addr_ = nullptr;
  }
  return AllocTensors();
}

std::shared_ptr<AscendDeviceInfo> AscendNativeSession::GetDeviceInfo(const std::shared_ptr<Context> &context) {
  auto device_list = context->MutableDeviceInfo();
  auto ascend_info_iter = std::find_if(
    device_list.begin(), device_list.end(), [&](std::shared_ptr<mindspore::DeviceInfoContext> &device_info) {
      return (device_info && device_info->GetDeviceType() == kAscend && device_info->GetProvider() == "Ascend_native");
    });
  if (ascend_info_iter == device_list.end()) {
    MS_LOG(ERROR) << "AscendDeviceInfo is not set. If using distributed inference, make sure device_id "
                     "and rank_id are set in AscendDeviceInfo";
    return nullptr;
  }
  auto device_info = *(ascend_info_iter);
  return device_info->Cast<mindspore::AscendDeviceInfo>();
}

Status AscendNativeSession::Init(const std::shared_ptr<Context> &context, const ConfigInfos &config_info) {
  MS_LOG(INFO) << "AscendNativeSession::Init";
  context_ = ContextUtils::Convert(context.get());
  auto ascend_info = GetDeviceInfo(context);
  if (ascend_info != nullptr) {
    std::string rank_table_file = "";
    uint32_t device_id = ascend_info->GetDeviceID();
    int rank_id = static_cast<int>(ascend_info->GetRankID());
    std::string s_rank_id = std::to_string(rank_id);
    bool ret = hccl::HcclAdapter::GetInstance().InitHccl(device_id, s_rank_id, rank_table_file, hccl::HcclMode::kGraph);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "HcclAdapter::initHccl failed";
    }
  }
  return kSuccess;
}

Status AscendNativeSession::FindGraphInputs(const std::vector<AnfNodePtr> &node_list,
                                            const std::vector<AnfNodePtr> &graph_inputs,
                                            const std::vector<std::shared_ptr<kernel::BaseKernel>> &kernels) {
  if (graph_inputs.empty()) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::Schedule get graph inputs node failed";
    return kLiteError;
  }

  size_t found_input_node = 0;
  this->inputs_.resize(graph_inputs.size());
  int kernel_id = 0;
  std::unordered_set<AnfNodePtr> input_hash;
  for (size_t ni = 0; ni < node_list.size(); ni++) {
    auto &node = node_list[ni];
    if (!node->isa<CNode>() || !AnfUtils::IsRealKernel(node)) {
      continue;
    }
    auto cnode = utils::cast<CNodePtr>(node);
    for (size_t i = 1; i < cnode->inputs().size(); i++) {
      for (size_t j = 0; j < graph_inputs.size(); j++) {
        if (cnode->input(i) == graph_inputs[j]) {
          this->inputs_[j] = kernels[kernel_id]->in_tensors().at(i - 1);
          this->inputs_[j]->set_category(lite::GRAPH_INPUT);
          if (input_hash.find(cnode->input(i)) == input_hash.end()) {
            input_hash.insert(cnode->input(i));
            found_input_node++;
            break;
          }
        }
      }
    }
    kernel_id++;
  }
  if (found_input_node != graph_inputs.size()) {
    MS_LOG(ERROR) << "Can not find corresponding anfnode for all funcgraph inputs.";
    return kLiteError;
  }
  return kSuccess;
}

Status AscendNativeSession::FindGraphOutputs(const std::vector<AnfNodePtr> &node_list, const AnfNodePtr &graph_output,
                                             const std::vector<std::shared_ptr<kernel::BaseKernel>> &kernels) {
  if (graph_output == nullptr) {
    MS_LOG(ERROR) << "get graph output node failed.";
    return kLiteError;
  }
  const PrimitiveSet prims{prim::kPrimTupleGetItem, prim::kPrimListGetItem, prim::kPrimArrayGetItem,
                           prim::kPrimMakeTuple};
  auto cnode = utils::cast<CNodePtr>(graph_output);
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "ascend_native delegate not support empty subgraph now.";
    return kLiteError;
  }
  auto prim_vnode = cnode->input(0);
  if (IsOneOfPrimitive(prim_vnode, prims)) {
    MS_LOG(ERROR) << "ascend_native delegate not support maketuple and tuple-get-item operator now.";
    return kLiteError;
  }
  int kernel_id = 0;
  for (size_t ni = 0; ni < node_list.size(); ni++) {
    auto &node = node_list[ni];
    if (!node->isa<CNode>() || !AnfUtils::IsRealKernel(node)) {
      continue;
    }
    if (node == graph_output) {
      for (auto &output : kernels_[kernel_id]->out_tensors()) {  // TBD do kernel hash map
        this->outputs_.emplace_back(output);
        output->set_category(lite::GRAPH_OUTPUT);
      }
      break;
    }
    kernel_id++;
  }
  return kSuccess;
}

Status AscendNativeSession::CompileGraph(FuncGraphPtr func_graph, const void *data, size_t size, uint32_t *graph_id) {
  MS_LOG(INFO) << "AscendNativeSession::CompileGraph";
  if (delegate_ == nullptr) {
    MS_LOG(ERROR) << "ascend_native delegate not inited";
    return kLiteNullptr;
  }

  delegate_->set_ascend_native_ctx(context_);
  ascend_native_stream_ = ascend_native::CreateStream();
  // call delegate replace nodes make the delegate replace the graph nodes
  delegate_->ReplaceNodes(func_graph);
  auto nodes = TopoSort(func_graph->get_return());
  // for all the nodes in the graph, call the delegate isDelegateNode and CreateKernel interface to create kernels
  for (auto &node : nodes) {
    if (!node->isa<CNode>() || !AnfUtils::IsRealKernel(node)) {
      continue;
    }

    auto kernel = delegate_->CreateKernel(node);
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "delegate create kernel failed.";
      return kLiteError;
    }
    kernels_.emplace_back(kernel);
  }
  if (kernels_.empty()) {
    MS_LOG(ERROR) << "delegate not support empty subgraph now.";
    return kLiteError;
  }

  auto findio_ret = FindGraphInputs(nodes, func_graph->get_inputs(), kernels_);
  if (findio_ret != kSuccess) {
    MS_LOG(ERROR) << "Search graph input tensors failed.";
    return findio_ret;
  }
  findio_ret = FindGraphOutputs(nodes, func_graph->output(), kernels_);
  if (findio_ret != kSuccess) {
    MS_LOG(ERROR) << "Search graph output tensors failed.";
    return findio_ret;
  }
  if (AllocateGraphTensors() != kSuccess) {
    MS_LOG(ERROR) << "kernel graph allocation failed ";
    return kLiteError;
  }
  // call kernel prepare
  for (auto &kernel : kernels_) {
    auto ret = kernel->Prepare();
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "kernel prepare failed with " << ret;
      return kLiteError;
    }
  }
  return kSuccess;
}

void AscendNativeSession::ResetTensorData(const std::vector<void *> &old_data,
                                          const std::vector<lite::Tensor *> &tensors) {
  for (size_t j = 0; j < old_data.size(); j++) {
    tensors.at(j)->set_data(old_data.at(j));
  }
}

Status AscendNativeSession::RefDataFromOuter(const std::vector<tensor::Tensor> &outer_tensors) {
  const std::vector<infer::abstract::Tensor *> &inner_tensors = this->inputs_;
  if (outer_tensors.size() != inner_tensors.size()) {
    MS_LOG(EXCEPTION) << "user input size " << outer_tensors.size() << " is not equal to graph input size "
                      << inner_tensors.size();
  }
  std::vector<void *> old_data;

  for (size_t i = 0; i < outer_tensors.size(); i++) {
    auto &user_input = outer_tensors.at(i);
    auto input = inner_tensors.at(i);
    if (user_input.data_type() != input->data_type()) {
      ResetTensorData(old_data, inner_tensors);
      MS_LOG(ERROR) << "Tensor " << user_input.id() << " has a different data type from input" << input->tensor_name()
                    << ".";
      return kLiteError;
    }
    if (user_input.data_c() == nullptr) {
      ResetTensorData(old_data, inner_tensors);
      MS_LOG(ERROR) << "Tensor " << user_input.id() << " has no data.";
      return kLiteError;
    }
    old_data.push_back(input->data());
    if (input->data_type() == kObjectTypeString) {
      MS_LOG(ERROR) << "Not support string type tensor now!";
      return kLiteError;
    }
    if (user_input.data_c() != input->data()) {
      if (input->Size() != user_input.Size()) {
        ResetTensorData(old_data, inner_tensors);
        MS_LOG(ERROR) << "Tensor " << user_input.id() << " has wrong data size.";
        return kLiteError;
      }
      input->set_data(user_input.data_c(), false);
    }
  }
  return kSuccess;
}

std::vector<mindspore::tensor::Tensor> AscendNativeSession::LiteTensorToTensor() {
  const std::vector<infer::abstract::Tensor *> &inner_tensors = this->outputs_;
  std::vector<mindspore::tensor::Tensor> tensors;
  for (auto inner_tensor : inner_tensors) {
    if (inner_tensor == nullptr) {
      MS_LOG(ERROR) << "Input inner_tensors has nullptr.";
      return std::vector<mindspore::tensor::Tensor>{};
    }
    auto type_id = inner_tensor->data_type();
    auto shape = inner_tensor->shape();
    auto data = inner_tensor->MutableData();
    auto data_size = inner_tensor->Size();
    auto ref_tensor_data = std::make_shared<TensorRefData>(data, inner_tensor->ElementsNum(), data_size, shape.size());
    std::vector<int64_t> shape64;
    std::transform(shape.begin(), shape.end(), std::back_inserter(shape64),
                   [](int dim) { return static_cast<int64_t>(dim); });
    mindspore::tensor::Tensor tensor(type_id, shape64, ref_tensor_data);
    tensors.emplace_back(std::move(tensor));
  }
  return tensors;
}

Status AscendNativeSession::RunGraph(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
                                     std::vector<tensor::Tensor> *outputs, const MSKernelCallBack &before,
                                     const MSKernelCallBack &after) {
  MS_LOG(INFO) << "AscendNativeSession::RunGraph";

  // get inputs and outputs tensors, set the data ptr for inputs
  auto ret = RefDataFromOuter(inputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Sync tensor data from use tensor failed: " << ret;
    return ret;
  }
  // call kernel run interface one by one
  for (auto &kernel : kernels_) {
    auto exec_ret = kernel->Run();
    if (exec_ret != kSuccess) {
      MS_LOG(ERROR) << "kernel Run failed with " << exec_ret;
      return kLiteError;
    }
  }
  // synchronize all tasks are finished
  *outputs = LiteTensorToTensor();
  if (outputs->size() != this->outputs_.size()) {
    MS_LOG(ERROR) << "Convert output tensors failed";
    return kLiteNullptr;
  }
  ascend_native::SyncDevice(ascend_native_stream_);
  return kSuccess;
}

Status AscendNativeSession::RunGraph(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
                                     std::vector<tensor::Tensor> *outputs) {
  return RunGraph(graph_id, inputs, outputs, nullptr, nullptr);
}

Status AscendNativeSession::Resize(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
                                   const std::vector<std::vector<int64_t>> &new_shapes) {
  MS_LOG(EXCEPTION) << "AscendNativeSession::Resize not implemented";
}

std::vector<MutableTensorImplPtr> AscendNativeSession::GetOutputs(uint32_t graph_id) {
  std::vector<MutableTensorImplPtr> result;
  std::transform(this->outputs_.begin(), this->outputs_.end(), std::back_inserter(result),
                 [](infer::abstract::Tensor *output) { return std::make_shared<LiteTensorImpl>(output); });
  return result;
}
std::vector<MutableTensorImplPtr> AscendNativeSession::GetInputs(uint32_t graph_id) {
  std::vector<MutableTensorImplPtr> result;
  std::transform(this->inputs_.begin(), this->inputs_.end(), std::back_inserter(result),
                 [](infer::abstract::Tensor *input) { return std::make_shared<LiteTensorImpl>(input); });
  return result;
}
std::vector<std::string> AscendNativeSession::GetOutputNames(uint32_t graph_id) {
  std::vector<std::string> output_names;

  auto lite_outputs = this->outputs_;
  std::transform(lite_outputs.begin(), lite_outputs.end(), std::back_inserter(output_names),
                 [](infer::abstract::Tensor *tensor) { return tensor->tensor_name(); });
  return output_names;
}
std::vector<std::string> AscendNativeSession::GetInputNames(uint32_t graph_id) {
  std::vector<std::string> input_names;
  auto lite_inputs = this->inputs_;
  std::transform(lite_inputs.begin(), lite_inputs.end(), std::back_inserter(input_names),
                 [](infer::abstract::Tensor *tensor) { return tensor->tensor_name(); });
  return input_names;
}
MutableTensorImplPtr AscendNativeSession::GetOutputByTensorName(uint32_t graph_id, const std::string &tensorName) {
  auto lite_outputs = this->outputs_;
  auto it = std::find_if(lite_outputs.begin(), lite_outputs.end(), [tensorName](infer::abstract::Tensor *tensor) {
    if (tensor->tensor_name() == tensorName) {
      return true;
    }
    return false;
  });
  if (it != lite_outputs.end()) {
    return std::make_shared<LiteTensorImpl>(*it);
  }
  return nullptr;
}
MutableTensorImplPtr AscendNativeSession::GetInputByTensorName(uint32_t graph_id, const std::string &tensorName) {
  auto lite_inputs = this->inputs_;
  auto it = std::find_if(lite_inputs.begin(), lite_inputs.end(), [tensorName](infer::abstract::Tensor *tensor) {
    if (tensor->tensor_name() == tensorName) {
      return true;
    }
    return false;
  });
  if (it != lite_inputs.end()) {
    return std::make_shared<LiteTensorImpl>(*it);
  }
  return nullptr;
}

static std::shared_ptr<InferSession> AscendNativeSessionCreator(const std::shared_ptr<Context> &ctx,
                                                                const ConfigInfos &config_infos) {
  auto &device_contexts = ctx->MutableDeviceInfo();
  if (device_contexts.empty()) {
    return nullptr;
  }

  auto provider = device_contexts.at(0)->GetProvider();
  auto delegate = std::make_shared<mindspore::AscendNativeDelegate>();
  if (delegate == nullptr) {
    return nullptr;
  }
  auto session = std::make_shared<AscendNativeSession>(delegate);
  constexpr auto kAscendProviderAscendNative = "ascend_native";

  if (provider == kAscendProviderAscendNative) {
    session->Init(ctx);
  }
  return session;
}

REG_SESSION(kAscendNativeSession, AscendNativeSessionCreator);
}  // namespace mindspore
