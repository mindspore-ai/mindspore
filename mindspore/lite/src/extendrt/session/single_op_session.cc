/**
 * Copyright 2019-2023  uawei Technologies Co., Ltd
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

#include <algorithm>
#include <functional>
#include <string>
#include <vector>
#include <map>
#include <utility>

#include "src/extendrt/session/single_op_session.h"
#include "src/extendrt/infer_device_address.h"

#include "plugin/factory/ms_factory.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "kernel/common_utils.h"
#include "plugin/device/cpu/kernel/cpu_kernel_mod.h"
#include "src/extendrt/utils/kernel_build_utils.h"
#include "src/extendrt/kernel/ascend/plugin/ascend_kernel_plugin.h"
#include "src/common/common.h"
#include "mindspore/core/ops/custom.h"
#include "extendrt/session/factory.h"
#include "extendrt/utils/runtime_utils.h"
#include "extendrt/utils/tensor_default_impl.h"
#include "extendrt/utils/func_graph_utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "extendrt/utils/tensor_utils.h"
#include "mindspore/lite/src/common/common.h"

namespace mindspore {
const size_t tensor_max_size = 0x1000000;

Status SingleOpInferSession::AscendInit(const std::shared_ptr<Context> &context) {
  auto device_list = context->MutableDeviceInfo();
  for (const auto &device_info : device_list) {
    if (device_info == nullptr) {
      MS_LOG(ERROR) << "Device info get from Context cannot be nullptr";
      return kLiteError;
    }
    if (device_info->GetDeviceType() == DeviceType::kAscend) {
      if (!kernel::AscendKernelPlugin::Register()) {
        MS_LOG(ERROR) << "Failed to register Ascend plugin";
        return kLiteError;
      }
      bool is_registered = kernel::AscendAllocatorPlugin::GetInstance().Register();
      if (!is_registered) {
        MS_LOG(ERROR) << "AscendAllocatorPlugin failed to register, cannot do acl memory operations";
        return kLiteError;
      }
      auto ascend_device_info = device_info->Cast<mindspore::AscendDeviceInfo>();
      if (ascend_device_info == nullptr) {
        MS_LOG(ERROR) << "Failed to cast device info to AscendDeviceInfo";
        return kLiteError;
      }
      device_id_ = ascend_device_info->GetDeviceID();
      return kSuccess;
    }
  }
  MS_LOG(DEBUG) << "There is no ascend device info, no need to register ascend plugin.";
  return kSuccess;
}

Status SingleOpInferSession::Init(const std::shared_ptr<Context> &context, const ConfigInfos &config_info) {
  MS_LOG(INFO) << "SingleOpInferSession::Init";
  if (context == nullptr) {
    MS_LOG(ERROR) << "Input argument context cannot be nullptr";
    return kLiteError;
  }
  if (AscendInit(context) != kSuccess) {
    MS_LOG(ERROR) << "Init ascend failed.";
    return kLiteError;
  }
  config_infos_ = config_info;
  return kSuccess;
}

void SingleOpInferSession::SetCustomAscendOpAttrs(const kernel::BaseOperatorPtr &op) {
  if (config_infos_.find(lite::kAscendContextSection) == config_infos_.end() &&
      config_infos_.find("inner_common") == config_infos_.end()) {
    MS_LOG(DEBUG) << "There is no ascend context info in config infos.";
    return;
  }
  // set custom op attrs
  auto custom_op = std::dynamic_pointer_cast<ops::Custom>(op);
  if (custom_op == nullptr) {
    MS_LOG(ERROR) << "Cast Custom op failed, can't set custom attrs.";
    return;
  }
  auto dst_prim = custom_op->GetPrim();
  if (dst_prim == nullptr) {
    MS_LOG(ERROR) << "Get prim from custom op failed.";
    return;
  }
  auto share_mem = config_infos_["inner_common"];
  if (share_mem.find("inner_calc_workspace_size") != share_mem.end()) {
    auto value = share_mem["inner_calc_workspace_size"];
    is_multi_model_sharing_mem_prepare_ = value == "true" ? true : false;
    dst_prim->AddAttr("inner_calc_workspace_size", MakeValue(is_multi_model_sharing_mem_prepare_));
    MS_LOG(INFO) << "inner_calc_workspace_size: " << is_multi_model_sharing_mem_prepare_;
  }
  if (share_mem.find("inner_sharing_workspace") != share_mem.end()) {
    auto value = share_mem["inner_sharing_workspace"];
    bool is_inner_sharing_workspace = value == "true" ? true : false;
    dst_prim->AddAttr("inner_sharing_workspace", MakeValue(is_inner_sharing_workspace));
    MS_LOG(INFO) << "is_inner_sharing_workspace: " << is_inner_sharing_workspace;
  }
  if (share_mem.find("inner_model_path") != share_mem.end()) {
    auto model_path = share_mem["inner_model_path"];
    dst_prim->AddAttr("inner_model_path", MakeValue(model_path));
    MS_LOG(INFO) << "inner_model_path: " << model_path;
  }
  if (share_mem.find("inner_workspace") != share_mem.end()) {
    dst_prim->AddAttr("inner_workspace", MakeValue(true));
  }
  if (share_mem.find("inner_weightspace") != share_mem.end()) {
    dst_prim->AddAttr("inner_weightspace", MakeValue(true));
  }
  if (share_mem.find("inner_weightspace_workspace") != share_mem.end()) {
    dst_prim->AddAttr("inner_weightspace_workspace", MakeValue(true));
  }
  auto ascend_context = config_infos_[lite::kAscendContextSection];
  std::string profiling_path;
  if (ascend_context.find(lite::kProfilingPathKey) != ascend_context.end()) {
    profiling_path = ascend_context[lite::kProfilingPathKey];
    dst_prim->AddAttr(lite::kProfilingPathKey, MakeValue(profiling_path));
  }
  if (ascend_context.find(lite::kDumpPathKey) != ascend_context.end()) {
    if (!profiling_path.empty()) {
      MS_LOG(ERROR) << "Profiling and dump can't be set at the same time.";
      return;
    }
    auto dump_path = ascend_context[lite::kDumpPathKey];
    dst_prim->AddAttr(lite::kDumpPathKey, MakeValue(dump_path));
  }
}

std::tuple<kernel::KernelModPtr, LiteKernelArgs> SingleOpInferSession::BuildCustomAscendKernelImpl(
  const CNodePtr &cnode) {
  auto kernel_name = lite::kNameCustomAscend;
  std::shared_ptr<kernel::KernelMod> kernel_mod = kernel::Factory<kernel::KernelMod>::Instance().Create(kernel_name);
  if (kernel_mod == nullptr) {
    MS_LOG(ERROR) << "Kernel mod is nullptr, kernel name: " << kernel_name;
    return std::make_tuple(nullptr, LiteKernelArgs{});
  }
  MS_LOG(INFO) << "SingleOpInferSession::Kernels " << kernel_name;
  kernel_mod->SetDevicedId(device_id_);

  auto make_kernel_tensor = [](TypeId type_id, const ShapeVector &shape) {
    auto kernel_tensor = new (std::nothrow) kernel::KernelTensor();
    if (kernel_tensor == nullptr) {
      return kernel_tensor;
    }
    kernel_tensor->SetType(std::make_shared<TensorType>(TypeIdToType(type_id)));
    kernel_tensor->SetShape(std::make_shared<abstract::TensorShape>(shape));
    return kernel_tensor;
  };

  LiteKernelArgs args;
  BaseOperatorPtr op;
  if (!FuncGraphUtils::GetCNodeOperator(cnode, &op)) {
    MS_LOG(ERROR) << "Failed to create operator for cnode " << cnode->fullname_with_scope();
    return std::make_tuple(nullptr, LiteKernelArgs{});
  }
  std::vector<tensor::TensorPtr> tensor_cache;
  std::map<AnfWithOutIndex, kernel::KernelTensor *> kernel_tensor_map;
  std::vector<AnfWithOutIndex> inputs;
  std::vector<AnfWithOutIndex> outputs;
  FuncGraphUtils::GetCNodeInputsOutputs(cnode, &inputs, &outputs);
  for (size_t i = 0; i < inputs.size(); i++) {
    auto &input = inputs[i];
    auto data_type = FuncGraphUtils::GetTensorDataType(input);
    auto shape = FuncGraphUtils::GetTensorShape(input);
    auto kernel_tensor = make_kernel_tensor(static_cast<TypeId>(data_type), shape);
    auto tensor_data = FuncGraphUtils::GetConstNodeValue(input.first);
    if (tensor_data) {
      tensor_cache.push_back(tensor_data);
      kernel_tensor->SetData(std::make_shared<kernel::Address>(tensor_data->data_c(), tensor_data->Size()));
    }
    args.inputs.push_back(kernel_tensor);
    kernel_tensor_map[input] = kernel_tensor;
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    auto &output = outputs[i];
    auto kernel_tensor = new (std::nothrow) kernel::KernelTensor();
    auto it = kernel_tensor_map.find(output);
    if (it != kernel_tensor_map.end()) {  // use input as output
      kernel_tensor = it->second;
    } else {
      auto data_type = FuncGraphUtils::GetTensorDataType(output);
      auto shape = FuncGraphUtils::GetTensorShape(output);
      kernel_tensor = make_kernel_tensor(static_cast<TypeId>(data_type), shape);
    }
    args.outputs.push_back(kernel_tensor);
  }
  SetCustomAscendOpAttrs(op);
  auto ret = kernel_mod->Init(op->GetPrim(), args.inputs, args.outputs);
  MS_LOG(INFO) << "SingleOpInferSession::Kernels ret " << ret;
  if (!ret) {
    MS_LOG(ERROR) << "kernel init failed " << kernel_name;
    return std::make_tuple(nullptr, LiteKernelArgs{});
  }
  if (is_multi_model_sharing_mem_prepare_) {
    MS_LOG(INFO) << "is multi model sharing mem prepare";
    return std::make_tuple(nullptr, LiteKernelArgs{});
  }
  if (args.inputs.size() > 0) {
    args.inputs.pop_back();
  }
  return std::make_tuple(kernel_mod, args);
}

Status SingleOpInferSession::BuildCustomAscendKernel(const CNodePtr &cnode) {
  kernel::KernelModPtr kernel_mod;
  LiteKernelArgs args;
  std::tie(kernel_mod, args) = BuildCustomAscendKernelImpl(cnode);
  if (is_multi_model_sharing_mem_prepare_) {
    MS_LOG(INFO) << "using ascend workspace sharing.";
    return kSuccess;
  }
  if (kernel_mod == nullptr) {
    MS_LOG(ERROR) << "Build ascend kernel failed for node: " << cnode->fullname_with_scope();
    return kLiteError;
  }
  kernel_mod_ = kernel_mod;
  kernel_args_ = args;
  return kSuccess;
}

Status SingleOpInferSession::InitInputOutputInfos(const FuncGraphPtr &graph) {
  std::vector<AnfWithOutIndex> input_tensors;
  std::vector<AnfWithOutIndex> output_tensors;
  FuncGraphUtils::GetFuncGraphInputs(graph, &input_tensors);
  FuncGraphUtils::GetFuncGraphOutputs(graph, &output_tensors);
  if (kernel_args_.inputs.size() != input_tensors.size()) {
    MS_LOG(ERROR) << "Graph inputs size " << input_tensors.size() << " != custom inputs size "
                  << kernel_args_.inputs.size();
    return kCoreFailed;
  }
  if (kernel_args_.outputs.size() != output_tensors.size()) {
    MS_LOG(ERROR) << "Graph outputs size " << output_tensors.size() << " != custom inputs size "
                  << kernel_args_.outputs.size();
    return kCoreFailed;
  }
  for (size_t i = 0; i < input_tensors.size(); i++) {
    auto &tensor = input_tensors[i];
    auto &kernel_tensor = kernel_args_.inputs[i];
    MS_CHECK_TRUE_RET(kernel_tensor != nullptr, kLiteNullptr);
    auto tensor_name = FuncGraphUtils::GetTensorName(tensor);
    auto data_type = static_cast<DataType>(kernel_tensor->dtype_id());
    auto shape = kernel_tensor->GetShapeVector();
    // when input shape is NOT dynamic, the sizes are known and memory can be pre-alloced (thus set is_acl_host to true)
    bool is_acl_host = IsDynamicShape(shape) ? false : true;
    auto input = std::make_shared<TensorDefaultImpl>(tensor_name, data_type, shape, is_acl_host);
    MS_CHECK_TRUE_RET(input != nullptr, kLiteNullptr);
    inputs_.push_back(input);
    input_names_.push_back(FuncGraphUtils::GetTensorName(tensor));
    (void)malloced_data_size_.insert(std::make_pair(input, input->DataSize()));
  }
  for (size_t i = 0; i < output_tensors.size(); i++) {
    auto &tensor = output_tensors[i];
    auto &kernel_tensor = kernel_args_.outputs[i];
    auto tensor_name = FuncGraphUtils::GetTensorName(tensor);
    auto data_type = static_cast<DataType>(kernel_tensor->dtype_id());
    auto shape = kernel_tensor->GetShapeVector();
    if (dyn_outshape_.size() < output_tensors.size()) {
      dyn_outshape_.push_back(false);
    }
    if (IsDynamicShape(shape)) {
      dyn_outshape_[i] = true;
      MS_LOG(INFO) << "output " << i << " shape is dynamic: " << shape;
    }
    outputs_.push_back(std::make_shared<TensorDefaultImpl>(tensor_name, data_type, shape));
    output_names_.push_back(FuncGraphUtils::GetTensorName(tensor));
  }
  return kSuccess;
}

Status SingleOpInferSession::CompileGraph(FuncGraphPtr graph, const void *data, size_t size, uint32_t *) {
  MS_LOG(INFO) << "SingleOpInferSession::CompileGraph";
  MS_CHECK_TRUE_RET(graph != nullptr, kLiteNullptr);
  auto nodes = graph->TopoSort(graph->get_return());
  if (nodes.empty()) {
    MS_LOG(ERROR) << "There are no nodes in the graph";
    return kLiteNullptr;
  }
  size_t cnode_count = 0;
  for (const auto &node : nodes) {
    auto cnode = node->cast<CNodePtr>();
    if (!cnode || !AnfUtils::IsRealKernel(cnode)) {
      continue;
    }
    std::string kernel_name = common::AnfAlgo::GetCNodeName(cnode);
    if (kernel_name != lite::kNameCustomAscend) {
      MS_LOG(ERROR) << "Only support " << lite::kNameCustomAscend << ", but got " << kernel_name << ", node "
                    << cnode->fullname_with_scope();
      return kLiteError;
    }
    cnode_count += 1;
    if (cnode_count > 1) {
      MS_LOG(ERROR) << "Only support one " << lite::kNameCustomAscend << " node, but got " << kernel_name << ", node "
                    << cnode->fullname_with_scope();
      return kLiteError;
    }
    auto ret = BuildCustomAscendKernel(cnode);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Failed to Build custom ascend kernel";
      return ret;
    }
  }
  if (is_multi_model_sharing_mem_prepare_) {
    MS_LOG(INFO) << "is multi model sharing mem prepare";
    return kSuccess;
  }
  auto ret = InitInputOutputInfos(graph);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Failed to init graph input and output infos";
    return ret;
  }
  return kSuccess;
}

Status SingleOpInferSession::RunGraph(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
                                      std::vector<tensor::Tensor> *outputs, const MSKernelCallBack &before,
                                      const MSKernelCallBack &after) {
  return RunGraph(graph_id, inputs, outputs);
}

void SingleOpInferSession::SetBackOutputIfDynamic(std::vector<tensor::Tensor> *outputs) {
  for (size_t i = 0; i < kernel_args_.outputs.size(); ++i) {
    if (dyn_outshape_[i]) {
      MS_CHECK_TRUE_RET_VOID(kernel_args_.outputs[i] != nullptr);
      ShapeVector shape = kernel_args_.outputs[i]->GetShapeVector();
      (*outputs)[i].set_shape(shape);
      kernel::AddressPtr host_addr = kernel_args_.outputs[i]->GetHostData();
      kernel::AddressPtr device_addr = kernel_args_.outputs[i]->GetData();
      if (device_addr != nullptr) {
        TypeId out_type = kernel_args_.outputs[i]->dtype_id();
        (*outputs)[i] = tensor::Tensor(out_type, shape, nullptr, device_addr->size);
        (*outputs)[i].set_device_address(std::make_shared<LiteDeviceAddress>(device_addr->addr, device_addr->size));
      } else if (host_addr != nullptr) {
        TypeId out_type = kernel_args_.outputs[i]->dtype_id();
        auto type_size = abstract::TypeIdSize(out_type);
        MS_CHECK_TRUE_RET_VOID(type_size != 0);
        auto elem_num = kernel_args_.outputs[i]->size() / type_size;
        auto acl_mem_deleter = [](uint8_t *data_buf_ptr) {
          kernel::AscendAllocatorPlugin::GetInstance().FreeHost(static_cast<void *>(data_buf_ptr));
        };
        auto ref_tensor_data =
          std::make_shared<TensorRefData>(host_addr->addr, elem_num, host_addr->size, shape.size(), acl_mem_deleter);
        (*outputs)[i] = tensor::Tensor(out_type, shape, ref_tensor_data);
        MS_LOG(DEBUG) << "resetting kernel tensor shape to 0 for the next prediction";
        kernel_args_.outputs[i]->SetShapeVector({0});
      }
    }
  }
}

Status SingleOpInferSession::InitInputOutputData(const std::vector<tensor::Tensor> &inputs,
                                                 std::vector<tensor::Tensor> *outputs) {
  if (inputs.size() != kernel_args_.inputs.size()) {
    MS_LOG(ERROR) << "Given inputs size " << inputs.size() << " != graph inputs size " << kernel_args_.inputs.size();
    return kLiteError;
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    auto &input = inputs[i];
    auto &kernel_input = kernel_args_.inputs[i];
    MS_CHECK_TRUE_RET(kernel_input != nullptr, kLiteError);
    if (input.Size() != kernel_input->size()) {
      MS_LOG(ERROR) << "Byte size of input " << i << " != the size expected, given size " << input.Size()
                    << ", expected size " << kernel_input->size()
                    << ", input shape: " << kernel_input->GetShapeVector();
      return kLiteError;
    }
    auto input_device_address = input.device_address();
    if (input_device_address != nullptr && input_device_address->GetMutablePtr() != nullptr) {
      auto device_ptr = input_device_address->GetMutablePtr();
      kernel_args_.inputs[i]->SetData(std::make_shared<kernel::Address>(device_ptr, input.Size()));
      kernel_args_.inputs[i]->SetHostData(nullptr);
    } else {
      kernel_args_.inputs[i]->SetHostData(std::make_shared<kernel::Address>(input.data_c(), input.Size()));
      kernel_args_.inputs[i]->SetData(nullptr);
    }
    kernel_args_.inputs[i]->set_device_id(input.device_info().device_id_);
  }
  if (outputs->empty()) {
    std::transform(kernel_args_.outputs.begin(), kernel_args_.outputs.end(), std::back_inserter(*outputs),
                   [](auto &item) { return tensor::Tensor(item->dtype_id(), item->GetShapeVector()); });
  }
  if (outputs->size() != kernel_args_.outputs.size()) {
    MS_LOG(ERROR) << "Given outputs size " << outputs->size() << " != graph inputs size "
                  << kernel_args_.outputs.size();
    return kLiteError;
  }
  for (size_t i = 0; i < outputs->size(); i++) {
    auto &output = (*outputs)[i];
    auto &kernel_output = kernel_args_.outputs[i];
    if (!dyn_outshape_[i] && output.Size() != kernel_output->size()) {
      MS_LOG(ERROR) << "Byte size of output " << i << " != the size expected, given size " << output.Size()
                    << ", expected size " << kernel_output->size()
                    << ", output shape: " << kernel_output->GetShapeVector();
      return kLiteError;
    }
    auto output_device_address = output.device_address();
    if (output_device_address != nullptr && output_device_address->GetMutablePtr() != nullptr) {
      auto device_ptr = output_device_address->GetMutablePtr();
      kernel_args_.outputs[i]->SetData(std::make_shared<kernel::Address>(device_ptr, output.Size()));
      kernel_args_.outputs[i]->SetHostData(nullptr);
    } else {
      if (output.Size() == 0) {
        kernel_args_.outputs[i]->SetHostData(std::make_shared<kernel::Address>(nullptr, output.Size()));
      } else {
        kernel_args_.outputs[i]->SetHostData(std::make_shared<kernel::Address>(output.data_c(), output.Size()));
      }
      kernel_args_.outputs[i]->SetData(nullptr);
    }
    kernel_args_.outputs[i]->set_device_id(output.device_info().device_id_);
  }
  return kSuccess;
}

Status SingleOpInferSession::RunGraph(uint32_t, const std::vector<tensor::Tensor> &inputs,
                                      std::vector<tensor::Tensor> *outputs) {
  if (outputs == nullptr) {
    MS_LOG(ERROR) << "outputs cannot be nullptr";
    return kLiteError;
  }
  if (kernel_mod_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been built";
    return kLiteError;
  }
  MS_LOG(DEBUG) << "SingleOpInferSession::RunGraph with input and outputs";
  std::vector<ShapeVector> new_shapes;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(new_shapes), [](auto &t) { return t.shape_c(); });
  auto ret = OnNewInputShapes(new_shapes);
  if (ret != kSuccess) {
    return ret;
  }
  ret = InitInputOutputData(inputs, outputs);
  if (ret != kSuccess) {
    return ret;
  }
  try {
    std::vector<kernel::KernelTensor *> ignore_datas;
    if (!kernel_mod_->Launch(ignore_datas, ignore_datas, ignore_datas, nullptr)) {
      MS_LOG(ERROR) << "Failed to launch kernel";
      return kLiteError;
    }
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "Failed to launch kernel, exception: " << e.what();
    return kLiteError;
  }
  SetBackOutputIfDynamic(outputs);
  return kSuccess;
}

Status SingleOpInferSession::OnNewInputShapes(const std::vector<ShapeVector> &new_shapes) {
  if (kernel_mod_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been built";
    return kLiteError;
  }
  if (inputs_.size() != new_shapes.size()) {
    MS_LOG(ERROR) << "Graph inputs size " << inputs_.size() << " != resize input size " << new_shapes.size();
    return kLiteError;
  }
  auto input_changed = false;
  for (size_t i = 0; i < inputs_.size(); i++) {
    auto new_shape = new_shapes[i];
    if (std::any_of(new_shape.begin(), new_shape.end(), [](auto dim) { return dim < 0; })) {
      MS_LOG(ERROR) << "New shape of input " << i << " cannot be dynamic, new shape: " << new_shape;
      return kLiteError;
    }
    MS_CHECK_TRUE_RET(inputs_[i] != nullptr, kLiteError);
    if (inputs_[i]->Shape() != new_shapes[i]) {
      input_changed = true;
      MS_CHECK_TRUE_RET(kernel_args_.inputs[i] != nullptr, kLiteError);
      kernel_args_.inputs[i]->SetShapeVector(new_shapes[i]);
      MS_LOG(INFO) << "Set kernel args shape: " << kernel_args_.inputs[i]->GetShapeVector() << ", "
                   << inputs_[i]->Shape();
    }
  }
  if (!input_changed) {
    return kSuccess;
  }
  MS_LOG(INFO) << "SingleOpInferSession::Resize";

  if (kernel_mod_->Resize(kernel_args_.inputs, kernel_args_.outputs) != kSuccess) {
    MS_LOG(ERROR) << "Failed to resize custom ascend kernel";
    for (size_t i = 0; i < inputs_.size(); i++) {
      kernel_args_.inputs[i]->SetShapeVector(inputs_[i]->Shape());
    }
    return kLiteError;
  }
  // shapes of inputs and outputs should be updated in CustomAscendKernelMod::Resize
  for (size_t i = 0; i < inputs_.size(); i++) {
    auto input = inputs_[i];
    MS_CHECK_TRUE_RET(input != nullptr, kLiteNullptr);
    auto input_tensor = std::dynamic_pointer_cast<TensorDefaultImpl>(inputs_[i]);
    MS_CHECK_TRUE_MSG(input_tensor != nullptr, kLiteNullptr, "cast to TensorDefaultImpl failed");
    input_tensor->SetShape(kernel_args_.inputs[i]->GetShapeVector());
    MS_CHECK_TRUE_RET(malloced_data_size_.find(input) != malloced_data_size_.end(), kLiteError);
    if (input_tensor->DataSize() > malloced_data_size_.at(input)) {
      auto data_size = input_tensor->DataSize();
      void *data_buf = kernel::AscendAllocatorPlugin::GetInstance().MallocHost(data_size);
      MS_CHECK_TRUE_MSG(data_buf != nullptr, kLiteNullptr, "malloc on host failed");
      input_tensor->SetAclHostData(data_buf);
      malloced_data_size_[input] = data_size;
    }
  }
  for (size_t i = 0; i < outputs_.size(); i++) {
    outputs_[i]->SetShape(kernel_args_.outputs[i]->GetShapeVector());
  }
  return kSuccess;
}

Status SingleOpInferSession::Resize(uint32_t, const std::vector<tensor::Tensor> &,
                                    const std::vector<std::vector<int64_t>> &dims) {
  return OnNewInputShapes(dims);
}

std::vector<MutableTensorImplPtr> SingleOpInferSession::GetOutputs(uint32_t) { return outputs_; }
std::vector<MutableTensorImplPtr> SingleOpInferSession::GetInputs(uint32_t) { return inputs_; }
std::vector<std::string> SingleOpInferSession::GetOutputNames(uint32_t) { return output_names_; }
std::vector<std::string> SingleOpInferSession::GetInputNames(uint32_t) { return input_names_; }

MutableTensorImplPtr SingleOpInferSession::GetOutputByTensorName(uint32_t, const std::string &tensor_name) {
  for (size_t idx = 0; idx < output_names_.size(); ++idx) {
    if (output_names_[idx] == tensor_name) {
      if (idx < outputs_.size()) {
        return outputs_[idx];
      }
    }
  }
  MS_LOG(ERROR) << "Can't found tensor name " << tensor_name;
  return nullptr;
}

MutableTensorImplPtr SingleOpInferSession::GetInputByTensorName(uint32_t, const std::string &tensor_name) {
  for (size_t idx = 0; idx < input_names_.size(); ++idx) {
    if (input_names_[idx] == tensor_name) {
      if (idx < inputs_.size()) {
        return inputs_[idx];
      }
    }
  }
  MS_LOG(ERROR) << "Can't found tensor name " << tensor_name;
  return nullptr;
}

void SingleOpInferSession::AscendFinalize() {
  auto kernel_name = lite::kNameCustomAscend;
  std::shared_ptr<kernel::KernelMod> kernel_mod = kernel::Factory<kernel::KernelMod>::Instance().Create(kernel_name);
  if (kernel_mod == nullptr) {
    MS_LOG(INFO) << "Create kernel mod failed: " << kernel_name;
    return;
  }
  (void)kernel_mod->Finalize();
}

Status SingleOpInferSession::Finalize() {
  SingleOpInferSession::AscendFinalize();
  return kSuccess;
}

static std::shared_ptr<InferSession> SingleOpSessionCreator(const std::shared_ptr<Context> &ctx,
                                                            const ConfigInfos &config_infos) {
  auto session = std::make_shared<SingleOpInferSession>();
  auto ret = session->Init(ctx, config_infos);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Init session failed.";
    return nullptr;
  }
  return session;
}
REG_SESSION(kSingleOpSession, SingleOpSessionCreator);
}  // namespace mindspore
