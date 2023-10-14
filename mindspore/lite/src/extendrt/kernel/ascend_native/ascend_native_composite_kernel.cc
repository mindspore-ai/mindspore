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

#include "extendrt/kernel/ascend_native/ascend_native_composite_kernel.h"
#include <algorithm>
#include <string>
#include "extendrt/delegate/ascend_native/ascend_native_kernel_registry.h"
#include "extendrt/utils/func_graph_utils.h"
#include "extendrt/delegate/ascend_native/ops/ascend_native_composite.h"
#include "ops/primitive_c.h"
#include "src/train/opt_allocator.h"

#define DPRN() std::cout
namespace mindspore::kernel {
using mindspore::ops::AscendNativeComposite;

int AscendNativeCompositeKernel::InferShape() {
  for (auto &kernel : kernels_) {
    auto ret = kernel->InferShape();
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "kernel InferShape failed for " << kernel->get_name();
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

static inline BaseOperatorPtr CreateOperatorByCNode(const CNodePtr &cnode) {
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  auto kernel_name = prim->name();
  // Create PrimtiveC from map and create BaseOperator.
  mindspore::ops::PrimitiveCPtr primc_ptr = nullptr;
  static auto primc_fns = mindspore::ops::OpPrimCRegister::GetInstance().GetPrimCMap();
  if (primc_fns.find(kernel_name) != primc_fns.end()) {
    primc_ptr = primc_fns[kernel_name]();
    (void)primc_ptr->SetAttrs(prim->attrs());
  }
  MS_EXCEPTION_IF_NULL(primc_ptr);

  static auto operator_fns = mindspore::ops::OperatorRegister::GetInstance().GetOperatorMap();
  if (operator_fns.find(kernel_name) == operator_fns.end()) {
    MS_LOG(EXCEPTION) << "Cannot create BaseOperator for " << kernel_name;
  }
  auto base_operator = operator_fns[kernel_name](primc_ptr);
  MS_EXCEPTION_IF_NULL(base_operator);
  return base_operator;
}

std::shared_ptr<AscendNativeBaseKernel> AscendNativeCompositeKernel::CreateKernel(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    MS_LOG(ERROR) << "AscendNativeCompositeKernel::CreateKernel not a cnode";
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "AscendNativeCompositeKernel::CreateKernel cnode is nullptr";
    return nullptr;
  }
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  // step II - Prepare kernel attributes
  std::vector<kernel::InferTensor *> input_tensors;
  CreateInputKernelTensors(cnode, &input_tensors);
  std::vector<kernel::InferTensor *> output_tensors;
  CreateOutputKernelTensors(cnode, &output_tensors);
  kernel::InferPrimitive primitive;
  primitive.base_operator = CreateOperatorByCNode(cnode);
  primitive.cnode = cnode;
  auto kernel_name = cnode->fullname_with_scope();
  auto node_type = primitive.base_operator->name();
  // step III - Create Ascend native Kernel
  auto &plugin_factory = kernel::AscendNativeRegistrationFactory::Get();
  // TODO(nizzan) :: remove stub patch
  if (!plugin_factory.HasKey(node_type)) node_type = "AscendNativeStub";
  if (plugin_factory.HasKey(node_type)) {
    kernel::AscendNativeBaseKernel *ascend_native_op =
      plugin_factory.GetCreator(node_type)(input_tensors, output_tensors, primitive, context_, stream_, node_type);
    if (ascend_native_op == nullptr) {
      return nullptr;
    }
    auto ker = std::shared_ptr<kernel::AscendNativeBaseKernel>(ascend_native_op);
    if (ker == nullptr) {
      MS_LOG(ERROR) << "Kernel is nullptr";
      return nullptr;
    }

    if (!ker->IsWeightInputHanledInner()) {
      auto in_tensors = ker->in_tensors();
      for (auto &t : in_tensors) {
        MS_EXCEPTION_IF_NULL(t);
        if (t->IsConst() && (t->data() == nullptr)) {
          MS_LOG(ERROR) << "no data to tensor " << t->tensor_name();
          return nullptr;
        }
        if (t->IsConst() && t->device_data() == nullptr) {
          bool t_is_float = (t->data_type() == kNumberTypeFloat || t->data_type() == kNumberTypeFloat32);
          if (t_is_float) {
            void *device_ptr = nullptr;
            ascend_native::CopyHostFp32ToDeviceFp16(t->data(), &device_ptr, t->ElementsNum(),
                                                    const_cast<void *>(stream_));
            t->set_device_data(device_ptr);
          } else {
            t->set_device_data(ascend_native::MallocCopy(t->data(), t->Size(), const_cast<void *>(stream_)));
          }
        }
      }
    }
    // TODO(nizzan) :: remove if
    if (node_type == "AscendNativeStub") {
      ker->set_name(primitive.base_operator->name());
    }
    return ker;
  } else {
    MS_LOG(WARNING) << "Unsupported op type for ascend native. kernel name:" << kernel_name << " type:" << node_type;
    return nullptr;
  }
}

int AscendNativeCompositeKernel::GetIdxFromString(std::string str) {
  auto idx_str = str.rfind("_");
  std::string sub = str.substr(idx_str + 1);
  return std::stoi(sub);
}

static inline kernel::InferTensor *anfTensorToTensorInfo(const common::KernelWithIndex &tensor_id) {
  auto prev_node = tensor_id.first;
  auto tensor_val = FuncGraphUtils::GetConstNodeValue(prev_node);

  constexpr auto tensorrt_format = mindspore::Format::NCHW;
  auto name = FuncGraphUtils::GetTensorName(tensor_id);
  auto shape = FuncGraphUtils::GetTensorShape(tensor_id);
  auto data_type = FuncGraphUtils::GetTensorDataType(tensor_id);
  auto format = tensorrt_format;
  const void *data = nullptr;
  size_t data_len = 0;
  if (tensor_val) {
    data = tensor_val->data_c();
    data_len = tensor_val->Size();
    shape = tensor_val->shape_c();
  }
  std::vector<int> t_shape;
  t_shape.resize(shape.size());
  std::transform(shape.begin(), shape.end(), t_shape.begin(), [](int64_t x) { return static_cast<int>(x); });

  auto t = kernel::InferTensor::CreateTensor(name, static_cast<TypeId>(data_type), t_shape, data, data_len);
  if (t == nullptr) {
    MS_LOG(EXCEPTION) << "Cannot CreateTensor for " << name;
  }
  t->set_format(format);
  return t;
}

void AscendNativeCompositeKernel::CreateInputKernelTensors(const CNodePtr &cnode,
                                                           std::vector<kernel::InferTensor *> *input_tensors) {
  input_tensors->clear();
  auto graph_inputs = func_graph_->get_inputs();
  auto input_nodes = FuncGraphUtils::GetNodeInputs(cnode);
  auto cnode_inputs = this->primitive_.cnode->inputs();
  for (auto &tensor_id : input_nodes) {
    bool found_tensor = false;
    for (size_t j = 0; j < graph_inputs.size(); j++) {
      if (tensor_id.first == graph_inputs[j]) {
        int idx = GetIdxFromString(tensor_id.first->fullname_with_scope());
        input_tensors->push_back(in_tensors_[idx]);
        allocated_tensors_.insert(in_tensors_[idx]);
        auto it = std::find_if(kernel_list_.begin(), kernel_list_.end(),
                               [&tensor_id](const KernelWithIndexAndTensor &k) { return k.kernel_index == tensor_id; });
        if (it == kernel_list_.end()) {
          kernel_list_.push_back(KernelWithIndexAndTensor(tensor_id, in_tensors_[idx]));
        }
        found_tensor = true;
        break;
      }
    }
    if (!found_tensor) {
      for (size_t j = 1; j < cnode_inputs.size(); j++) {
        if (tensor_id.first == cnode_inputs[j]) {
          input_tensors->push_back(in_tensors_[j - 1]);
          allocated_tensors_.insert(in_tensors_[j - 1]);
          auto it =
            std::find_if(kernel_list_.begin(), kernel_list_.end(),
                         [&tensor_id](const KernelWithIndexAndTensor &k) { return k.kernel_index == tensor_id; });
          if (it == kernel_list_.end()) {
            kernel_list_.push_back(KernelWithIndexAndTensor(tensor_id, in_tensors_[j - 1]));
          }
          found_tensor = true;
          break;
        }
      }
      if (!found_tensor) {
        auto it = std::find_if(kernel_list_.begin(), kernel_list_.end(),
                               [&tensor_id](const KernelWithIndexAndTensor &k) { return k.kernel_index == tensor_id; });
        // tensor already created - use the same tensor
        if (it != kernel_list_.end()) {
          input_tensors->push_back(it->tensor_info);
        } else {
          auto tensor_info = anfTensorToTensorInfo(tensor_id);
          if (tensor_info == nullptr) {
            MS_LOG(ERROR) << "failed to get tensor info";
            return;
          }
          input_tensors->push_back(tensor_info);
          kernel_list_.push_back(KernelWithIndexAndTensor(tensor_id, tensor_info));
        }
      }
    }
  }
}

void AscendNativeCompositeKernel::CreateOutputKernelTensors(const CNodePtr &cnode,
                                                            std::vector<kernel::InferTensor *> *output_tensors) {
  output_tensors->clear();
  auto output_num = mindspore::AnfUtils::GetOutputTensorNum(cnode);
  bool output_found = false;
  for (size_t output_idx = 0; output_idx < output_num; ++output_idx) {
    mindspore::common::KernelWithIndex tensor_id = {cnode, output_idx};
    auto it = std::find_if(kernel_list_.begin(), kernel_list_.end(),
                           [&tensor_id](const KernelWithIndexAndTensor &k) { return k.kernel_index == tensor_id; });
    if (it != kernel_list_.end()) {
      output_tensors->push_back(it->tensor_info);
      output_found = true;
    } else {
      auto graph_output = func_graph_->output();
      if (IsPrimitiveCNode(graph_output, prim::kPrimMakeTuple)) {
        auto outc = graph_output->cast<CNodePtr>();
        for (size_t i = 1; i < outc->inputs().size(); i++) {
          if (IsPrimitiveCNode(outc->input(i), prim::kPrimTupleGetItem)) {
            auto get_item = outc->input(i)->cast<CNodePtr>();
            auto tuple_idx = common::AnfAlgo::GetTupleGetItemOutIndex(get_item);
            if ((get_item->input(SECOND_INPUT) == cnode) && (tuple_idx == output_idx)) {
              out_tensors_[i - 1]->set_device_data(
                ascend_native::MallocDevice(out_tensors_[i - 1]->Size(), const_cast<void *>(stream_)));
              out_tensors_[i - 1]->ResetRefCount();
              allocated_tensors_.insert(out_tensors_[i - 1]);
              output_tensors->push_back(out_tensors_[i - 1]);
              output_found = true;
            }
          } else if (outc->input(i) == cnode) {
            out_tensors_[i - 1]->set_device_data(
              ascend_native::MallocDevice(out_tensors_[i - 1]->Size(), const_cast<void *>(stream_)));
            out_tensors_[i - 1]->ResetRefCount();
            allocated_tensors_.insert(out_tensors_[i - 1]);
            output_tensors->push_back(out_tensors_[i - 1]);
            output_found = true;
          }
        }
      } else {
        if (graph_output == cnode) {
          out_tensors_[0]->set_device_data(
            ascend_native::MallocDevice(out_tensors_[0]->Size(), const_cast<void *>(stream_)));
          out_tensors_[0]->ResetRefCount();
          allocated_tensors_.insert(out_tensors_[0]);
          output_tensors->push_back(out_tensors_[0]);
          output_found = true;
        }
      }
    }
    if (!output_found) {
      auto tensor_info = anfTensorToTensorInfo(tensor_id);
      output_tensors->push_back(tensor_info);
      kernel_list_.push_back(KernelWithIndexAndTensor(tensor_id, tensor_info));
    }
  }
}

int AscendNativeCompositeKernel::AllocTensors() {
  OptAllocator allocator;
  std::unordered_map<kernel::InferTensor *, int> ref_count;
  offset_map_.clear();
  for (auto &kernel : kernels_) {
    // malloc output tensors
    for (auto &tensor : kernel->out_tensors()) {
      if ((allocated_tensors_.find(tensor) == allocated_tensors_.end())) {
        if (offset_map_.find(tensor) == offset_map_.end()) {
          size_t tensor_size = tensor->Size();
          size_t offset = allocator.Malloc(tensor_size);
          offset_map_[tensor] = offset;
          ref_count[tensor] = tensor->init_ref_count();
        }
      }
    }
    // free according to reference counter
    for (auto &tensor : kernel->in_tensors()) {
      if ((tensor->category() == lite::Category::VAR) &&
          ((allocated_tensors_.find(tensor) == allocated_tensors_.end()))) {
        int count = ref_count[tensor] - 1;
        ref_count[tensor] = count;
        if (count == 0) {
          allocator.Free(offset_map_[tensor]);
        }
      }
    }
  }
  // Set Tensor data
  device_mem_size_ = allocator.total_size();
  return ReAllocTensors();
}

int AscendNativeCompositeKernel::ReAllocTensors() {
  if (device_memory_base_addr_ != nullptr) {
    return lite::RET_OK;
  }
  if (device_mem_size_ > 0) {
    device_memory_base_addr_ = ascend_native::MallocDevice(device_mem_size_, const_cast<void *>(stream_));
    if (device_memory_base_addr_ == nullptr) {
      MS_LOG(EXCEPTION) << "Allocation of " << device_mem_size_ << "B on device failed";
      return kMDOutOfMemory;
    }
    for (auto &it : offset_map_) {
      auto &tensor = it.first;
      tensor->set_device_data(
        reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(device_memory_base_addr_) + it.second));
    }
  }
  return lite::RET_OK;
}

void AscendNativeCompositeKernel::FreeDevice() {
  ascend_native::FreeDevice(device_memory_base_addr_, const_cast<void *>(stream_));
  device_memory_base_addr_ = nullptr;
  for (auto &it : offset_map_) {
    auto &tensor = it.first;
    tensor->set_device_data(nullptr);
  }
}

void AscendNativeCompositeKernel::InitializeTensorRefrenceCnt() {
  for (auto &kernel : kernels_) {
    for (auto tensor : kernel->in_tensors()) {
      if (tensor->category() == lite::VAR || tensor->category() == lite::GRAPH_INPUT) {
        auto ref_count = tensor->init_ref_count();
        tensor->set_init_ref_count(ref_count + 1);
      }
    }
  }
}

int AscendNativeCompositeKernel::AllocateGraphTensors() {
  if (device_memory_base_addr_ == nullptr) {
    InitializeTensorRefrenceCnt();
  } else {
    FreeDevice();
  }
  return AllocTensors();
}

int AscendNativeCompositeKernel::AllocateGraphWorkspace(size_t ws_size) {
  if (get_workspace() != nullptr) return lite::RET_OK;
  void *ws_ptr = nullptr;
  if (ws_size > 0) {
    if (ws_size > max_ws_size_) {
      MS_LOG(ERROR) << "kernel ws is too big " << ws_size;
      return kLiteError;
    }
    // alloc ws on device space
    ws_ptr = ascend_native::MallocDevice(ws_size, const_cast<void *>(stream_));
    if (ws_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Allocation of " << ws_size << "B on device failed";
      return kMDOutOfMemory;
    }
    set_workspace(ws_ptr);
    set_workspace_size(ws_size);
    for (auto &kernel : kernels_) {
      kernel->set_workspace(ws_ptr);
    }
  }
  return lite::RET_OK;
}

int AscendNativeCompositeKernel::Prepare() {
  auto nodes = TopoSort(func_graph_->get_return());
  for (auto &node : nodes) {
    if (!node->isa<CNode>() || !AnfUtils::IsRealKernel(node)) {
      continue;
    }
    auto kernel = CreateKernel(node);
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "composite create kernel failed.";
      return lite::RET_ERROR;
    }
    kernels_.emplace_back(kernel);
  }
  if (kernels_.empty()) {
    MS_LOG(ERROR) << "composite does not support empty subgraph now.";
    return lite::RET_ERROR;
  }
  // call kernel prepare
  size_t ws_size = 0;
  for (auto &kernel : kernels_) {
    auto ret = kernel->Prepare();
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "composite kernel prepare failed with " << ret;
      return lite::RET_ERROR;
    }
    size_t k_ws_size = kernel->get_workspace_size();
    if (k_ws_size > ws_size) ws_size = k_ws_size;
  }
  if (AllocateGraphWorkspace(ws_size) != lite::RET_OK) {
    MS_LOG(ERROR) << "kernel workspace allocation failed ";
    return lite::RET_ERROR;
  }
  if (AllocateGraphTensors() != lite::RET_OK) {
    MS_LOG(ERROR) << "kernel graph allocation failed ";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

int AscendNativeCompositeKernel::Run() {
  MS_LOG(INFO) << "AscendNativeCompositeKernel::Execute Begin";
  // call kernel run interface one by one
  for (auto &kernel : kernels_) {
    auto ret = kernel->PreProcess();
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "kernel preprocess failed with " << ret << " for " << kernel->get_name();
      return lite::RET_ERROR;
    }
    ret = kernel->Run();
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "kernel run failed with " << ret << " for " << kernel->get_name();
      return lite::RET_ERROR;
    }
    // synchronize all tasks are finished
    ascend_native::SyncDevice(const_cast<void *>(stream_));
    ret = kernel->PostProcess();
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "kernel postprocess failed with " << ret << " for " << kernel->get_name();
      return lite::RET_ERROR;
    }
  }
  MS_LOG(INFO) << "AscendNativeCompositeKernel::Execute End";
  return lite::RET_OK;
}

int AscendNativeCompositeKernel::PostProcess() {
  // Free device data
  FreeDevice();
  ascend_native::FreeDevice(get_workspace(), const_cast<void *>(stream_));
  set_workspace(nullptr);
  // Decrement inputs ref count
  for (size_t i = 0; i < in_tensors_.size(); i++) {
    auto ref = in_tensors_[i]->ref_count();
    in_tensors_[i]->set_ref_count(--ref);
    if ((ref <= 0) && (in_tensors_[i]->category() == lite::VAR)) {
      ascend_native::FreeDevice(in_tensors_[i]->device_data(), const_cast<void *>(stream_));
      in_tensors_[i]->set_device_data(nullptr);
    }
  }
  return lite::RET_OK;
}

int AscendNativeCompositeKernel::PreProcess() {
  for (auto &tensor : out_tensors()) {
    if (tensor->device_data() == nullptr) {
      auto data_ptr = ascend_native::MallocDevice(tensor->Size(), const_cast<void *>(stream_));
      if (data_ptr == nullptr) {
        MS_LOG(ERROR) << "Cannot allocate device memory size:" << tensor->Size();
        return lite::RET_NULL_PTR;
      }
      tensor->set_device_data(data_ptr);
    }
    tensor->ResetRefCount();
  }
  auto ws_size = get_workspace_size();
  ReAllocTensors();
  if (AllocateGraphWorkspace(ws_size) != lite::RET_OK) {
    MS_LOG(ERROR) << "kernel workspace allocation failed ";
    return kLiteError;
  }
  if (InferShape() != lite::RET_OK) {
    MS_LOG(ERROR) << "InferShape AscendNativeCompositeKernel failed ";
    return kLiteError;
  }
  return lite::RET_OK;
}

int AscendNativeCompositeKernel::ReSize() {
  size_t ws_size = 0;
  for (auto &kernel : kernels_) {
    size_t k_ws_size = kernel->get_workspace_size();
    if (k_ws_size > ws_size) ws_size = k_ws_size;
    auto ret = kernel->ReSize();
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "kernel" << kernel->get_name() << " ReSize failed ";
      return lite::RET_ERROR;
    }
  }
  auto ret = AllocateGraphWorkspace(ws_size);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "kernel workspace allocation failed ";
    return lite::RET_ERROR;
  }
  ret = AllocateGraphTensors();
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "kernel graph allocation failed ";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}
REGISTER_ASCEND_NATIVE_CREATOR(ops::kNameAscendNativeComposite, AscendNativeCompositeKernel)
}  // namespace mindspore::kernel
