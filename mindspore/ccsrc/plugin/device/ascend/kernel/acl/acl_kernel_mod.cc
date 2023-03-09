/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/acl/acl_kernel_mod.h"

#include <vector>
#include <map>
#include <set>
#include "runtime/rt.h"
#include "ir/tensor.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/common_utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "runtime/device/kernel_runtime.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace {
const char kNAttrName[] = "N";
const mindspore::HashSet<std::string> kAclUnConst = {kScatterNdOpName};

std::map<uint32_t, tensor::TensorPtr> CheckValueDependEdge(
  const CNodePtr &node, const std::map<uint32_t, tensor::TensorPtr> &value_depend_list) {
  if (value_depend_list.empty()) {
    return {};
  }
  auto depends_list = abstract::GetValueDependArgIndices(node);
  if (depends_list.empty() || AnfAlgo::IsDynamicShapeSkipExecute(node)) {
    MS_LOG(DEBUG) << "The node " << node->fullname_with_scope() << " has no infer depend.";
    return {};
  }

  if (kAclUnConst.count(common::AnfAlgo::GetCNodeName(node)) != 0) {
    return {};
  }

  std::map<uint32_t, tensor::TensorPtr> depend_res;
  for (auto index : depends_list) {
    auto iter = value_depend_list.find(LongToSize(index));
    if (iter != value_depend_list.end()) {
      depend_res.emplace(iter->first, iter->second);
    }
  }
  return depend_res;
}
}  // namespace

int AclKernelMod::UpdateInput(const CNodePtr &node, const runtime::OpRuntimeInfoPtr &node_op_runtime_info,
                              const std::map<uint32_t, tensor::TensorPtr> &value_depend_list) {
  bool need_cache_input_names = node_op_runtime_info != nullptr && node_op_runtime_info->acl_runtime_info_ != nullptr &&
                                node_op_runtime_info->acl_runtime_info_->use() &&
                                !node_op_runtime_info->acl_runtime_info_->is_dynamic_input_size();
  const auto &input_names = (need_cache_input_names) ? node_op_runtime_info->acl_runtime_info_->input_names()
                                                     : AclUtils::GetOpInputAnchorNames(node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
  input_size_list_.clear();
  input_size_list_.resize(input_names.size(), kSizeMax);
  if (input_names.size() != input_desc_list_.size()) {
    MS_LOG(EXCEPTION) << "Fail to update op input desc: " << node->fullname_with_scope()
                      << ", input names size: " << input_names.size()
                      << ", input_desc_list_ size: " << input_desc_list_.size();
  }

  const auto &depend_edge = CheckValueDependEdge(node, value_depend_list);
  for (size_t i = 0; i < input_num; ++i) {
    auto index = AclUtils::GetInputKernelIdxByGraphIdx(node, i);
    if (index < 0) {
      continue;
    }
    auto [input, idx] = common::AnfAlgo::GetPrevNodeOutput(node, i);
    auto op_runtime_info = input->user_data<runtime::OpRuntimeInfo>();

    ShapeVector ori_shape;
    ShapeVector input_shape;
    std::string input_format;
    TypeId input_type;
    size_t input_size = 0;

    if (op_runtime_info == nullptr) {
      input_type = AnfAlgo::GetOutputDeviceDataType(input, idx);
      ori_shape = common::AnfAlgo::GetOutputInferShape(input, idx);
      input_shape = AnfAlgo::GetOutputDeviceShape(input, idx);
      input_format = AnfAlgo::GetOutputFormat(input, idx);
      auto type_size = GetTypeByte(TypeIdToType(input_type));
      input_size = type_size * SizeOf(input_shape);
    } else {
      input_size = op_runtime_info->output_tensor_size(idx);
      input_type = op_runtime_info->output_type(idx);
      ori_shape = op_runtime_info->output_infer_shape(idx);
      input_shape = op_runtime_info->output_device_shape(idx);
      input_format = op_runtime_info->output_format(idx);
    }

    input_size_list_[index] = (input_size == 0) ? kSizeMax : input_size;
    if (input_type == kMetaTypeNone) {
      continue;
    }
    auto ori_format = IsOneOf3DFormat(input_format) ? kOpFormat_NCDHW : kOpFormat_DEFAULT;
    if (!opt::NeedInsertTransData(ori_shape, input_format)) {
      MS_LOG_DEBUG << "Set format of " << node->fullname_with_scope() << " to origin format";
      input_shape = ori_shape;
      input_format = ori_format;
    }
    AclUtils::UpdateShape(node, &ori_shape, &input_format);
    if (op_runtime_info != nullptr && op_runtime_info->acl_runtime_info_ != nullptr &&
        op_runtime_info->acl_runtime_info_->use() && !op_runtime_info->acl_runtime_info_->is_dynamic_input_size() &&
        input_desc_list_[index] != nullptr) {
      input_desc_list_[index]->SetShape(GeShape(input_shape));
      input_desc_list_[index]->SetOriginShape(GeShape(ori_shape));
      continue;
    }

    auto input_desc = GeOpConvertor::GetTensorDesc(input_shape, input_type, input_format, ori_shape, ori_format);
    MS_EXCEPTION_IF_NULL(input_desc);
    input_desc->SetName(input_names[index]);
    input_desc_list_[index] = input_desc;

    auto value_iter = depend_edge.find(i);
    if (value_iter != depend_edge.end()) {
      const_input_list_[index] = value_iter->second;
    }
  }
  return 0;
}

void AclKernelMod::UpdateOutput(const AnfNodePtr &node, const runtime::OpRuntimeInfoPtr &node_op_runtime_info) {
  bool node_acl_runtime_info_legal = node_op_runtime_info != nullptr &&
                                     node_op_runtime_info->acl_runtime_info_ != nullptr &&
                                     node_op_runtime_info->acl_runtime_info_->use();
  size_t output_num = AnfAlgo::GetOutputTensorNum(node);
  const auto &output_names =
    (node_acl_runtime_info_legal && !node_op_runtime_info->acl_runtime_info_->is_dynamic_output_size())
      ? node_op_runtime_info->acl_runtime_info_->output_names()
      : AclUtils::GetOpOutputAnchorNames(node);
  if (output_names.size() != output_desc_list_.size()) {
    MS_LOG(EXCEPTION) << "Fail to update op output desc: " << node->fullname_with_scope()
                      << ", output names size: " << output_names.size()
                      << ", output_desc_list_ size: " << output_desc_list_.size();
  }
  for (size_t i = 0; i < output_num; ++i) {
    auto index = AclUtils::GetOutputKernelIdxByGraphIdx(node, i);
    if (index < 0) {
      continue;
    }
    TypeId output_type;
    ShapeVector ori_shape;
    ShapeVector output_shape;
    std::string output_format;

    if (node_op_runtime_info == nullptr) {
      output_type = AnfAlgo::GetOutputDeviceDataType(node, i);
      ori_shape = common::AnfAlgo::GetOutputInferShape(node, i);
      output_shape = AnfAlgo::GetOutputDeviceShape(node, i);
      output_format = AnfAlgo::GetOutputFormat(node, i);
    } else {
      output_type = node_op_runtime_info->output_type(i);
      ori_shape = node_op_runtime_info->output_infer_shape(i);
      output_shape = node_op_runtime_info->output_device_shape(i);
      output_format = node_op_runtime_info->output_format(i);
    }

    auto ori_format = IsOneOf3DFormat(output_format) ? kOpFormat_NCDHW : kOpFormat_DEFAULT;
    if (!opt::NeedInsertTransData(ori_shape, output_format)) {
      MS_LOG_DEBUG << "Set format of " << node->fullname_with_scope() << " to origin format";
      output_shape = ori_shape;
      output_format = ori_format;
    }
    AclUtils::UpdateShape(node, &ori_shape, &output_format);
    if (node_acl_runtime_info_legal && !node_op_runtime_info->acl_runtime_info_->is_dynamic_output_size() &&
        output_desc_list_[index] != nullptr) {
      output_desc_list_[index]->SetShape(GeShape(output_shape));
      output_desc_list_[index]->SetOriginShape(GeShape(ori_shape));
      continue;
    }

    auto output_desc = GeOpConvertor::GetTensorDesc(output_shape, output_type, output_format, ori_shape, ori_format);
    MS_EXCEPTION_IF_NULL(output_desc);
    output_desc->SetName(output_names[index]);
    output_desc_list_[index] = output_desc;
  }
}

int AclKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                         const std::vector<KernelTensorPtr> &outputs,
                         const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  auto node_op_runtime_info = node->user_data<runtime::OpRuntimeInfo>();
  // init acl runtime info
  if (node_op_runtime_info != nullptr && node_op_runtime_info->acl_runtime_info_ != nullptr &&
      !node_op_runtime_info->acl_runtime_info_->use()) {
    node_op_runtime_info->acl_runtime_info_->SetUse(true);
    const auto &dynamic_input_names = GeOpConvertor::GetAclDynamicInputNames(node);
    const auto &dynamic_output_names = GeOpConvertor::GetAclDynamicOutputNames(node);
    node_op_runtime_info->acl_runtime_info_->SetIsDynamicInputSize(!dynamic_input_names.empty());
    node_op_runtime_info->acl_runtime_info_->SetIsDynamicOutputSize(!dynamic_output_names.empty());
    if (dynamic_input_names.empty()) {
      node_op_runtime_info->acl_runtime_info_->SetInputNames(AclUtils::GetOpInputAnchorNames(node));
    }
    if (dynamic_output_names.empty()) {
      node_op_runtime_info->acl_runtime_info_->SetOutputNames(AclUtils::GetOpOutputAnchorNames(node));
    }
  }

  // Update input size list & input desc list
  auto ret = UpdateInput(cnode, node_op_runtime_info, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  // Resize & Update output size list
  AscendKernelMod::UpdateOutputSizeList();
  // Update Output desc list
  UpdateOutput(node, node_op_runtime_info);
  need_skip_execute_ = AnfAlgo::IsDynamicShapeSkipExecute(cnode);

  return 0;
}

void AclKernelMod::UpdateReduceAxisAttr(const AnfNodePtr &node) {
  if (!common::AnfAlgo::IsReduceOp(op_type_)) {
    return;
  }
  auto cnode = node->cast<CNodePtr>();
  if (!common::AnfAlgo::HasNodeAttr(kAttrAxis, cnode)) {
    return;
  }
  opt::NormalizeReduceAttrAxis(cnode);
}

void AclKernelMod::ProcessAttribute(const std::shared_ptr<AclOpDesc> &op_desc_ptr,
                                    const std::vector<string> &input_names) {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  const auto &attr_to_input_maps = GeOpConvertor::GetNeedAddInput(node, true);
  UpdateReduceAxisAttr(node);
  auto attr_list = GeOpConvertor::GetAttrAndValue(node, true);
  for (auto &[attr_name, value] : attr_list) {
    if (value == nullptr) {
      MS_LOG(INFO) << "Current node's attr [" << attr_name << "] is nullptr";
      continue;
    }
    if (attr_to_input_maps.count(attr_name) != 0) {
      auto to_input_name = attr_to_input_maps.at(attr_name);
      auto iter = std::find(input_names.begin(), input_names.end(), to_input_name);
      if (iter == input_names.end()) {
        MS_LOG(EXCEPTION) << "Error input name!" << to_input_name;
      }
      op_desc_ptr->ProcessAclAttrs(attr_name, value, SET_ACL_INPUT);
      continue;
    }
    op_desc_ptr->ProcessAclAttrs(attr_name, value, SET_ACL_ATTR);
  }
}

bool AclKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                          const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (stream_ptr == nullptr) {
    MS_LOG(ERROR) << "stream_ptr should not be nullptr.";
    return false;
  }
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  if (need_skip_execute_) {
    // Skip reduce if axis is a empty Tensor (shape = 0)
    MS_LOG(INFO) << "The node " << node->fullname_with_scope() << " need skip.";
    // cppcheck-suppress unreadVariable
    auto lock = device::KernelRuntime::LockRuntime(stream_ptr);
    rtError_t status = aclrtMemcpyAsync(outputs[0]->addr, inputs[0]->size, inputs[0]->addr, inputs[0]->size,
                                        ACL_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
    if (status != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "AclrtMemcpyAsync failed for " << node->fullname_with_scope();
    }

    MS_LOG(INFO) << "Execute node:" << node->fullname_with_scope() << " success.";
    return true;
  }

  auto node_op_runtime_info = node->user_data<runtime::OpRuntimeInfo>();
  bool node_acl_runtime_info_legal = node_op_runtime_info != nullptr &&
                                     node_op_runtime_info->acl_runtime_info_ != nullptr &&
                                     node_op_runtime_info->acl_runtime_info_->use();
  const auto &input_names =
    (node_acl_runtime_info_legal && !node_op_runtime_info->acl_runtime_info_->is_dynamic_input_size())
      ? node_op_runtime_info->acl_runtime_info_->input_names()
      : AclUtils::GetOpInputAnchorNames(node);
  const auto &output_names =
    (node_acl_runtime_info_legal && !node_op_runtime_info->acl_runtime_info_->is_dynamic_output_size())
      ? node_op_runtime_info->acl_runtime_info_->output_names()
      : AclUtils::GetOpOutputAnchorNames(node);

  auto op_desc_ptr = std::make_shared<AclOpDesc>(op_type_, node);
  MS_EXCEPTION_IF_NULL(op_desc_ptr);
  op_desc_ptr->AddTensorDesc(input_desc_list_, output_desc_list_);
  op_desc_ptr->AddDataBuf(inputs, input_size_list_, outputs, output_size_list_, input_names, output_names,
                          const_input_list_);
  ProcessAttribute(op_desc_ptr, input_names);
  op_desc_ptr->ClearNullTensor();

  // cppcheck-suppress unreadVariable
  auto lock = device::KernelRuntime::LockRuntime(stream_ptr);
  // Current enable binary->fuzz->stable mode.
  auto set_compile_flag = ACL_SUCCESS;
  if (is_dynamic_) {
    set_compile_flag = aclopSetCompileFlag(ACL_OP_COMPILE_FUZZ);
  } else {
    set_compile_flag = aclopSetCompileFlag(ACL_OP_COMPILE_DEFAULT);
  }
  if (set_compile_flag != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Acl set compile mode failed! op_name is " << op_type_ << " and error flag is "
                  << set_compile_flag;
    return false;
  }

  MS_LOG(DEBUG) << "Start aclopCompileAndExecute of node: " << node->fullname_with_scope() << " op_type_:" << op_type_;
  if (op_desc_ptr->input_tensor_desc().size() != op_desc_ptr->input_tensor_data().size()) {
    MS_LOG(ERROR) << "For input, the size of tensor_desc and tensor_data is inconsistent! node: "
                  << node->fullname_with_scope();
    return false;
  }
  if (op_desc_ptr->output_tensor_desc().size() != op_desc_ptr->output_tensor_data().size()) {
    MS_LOG(ERROR) << "For output, the size of tensor_desc and tensor_data is inconsistent! node: "
                  << node->fullname_with_scope();
    return false;
  }
  bool ret = aclopCompileAndExecute(const_cast<char *>(op_type_.c_str()), op_desc_ptr->input_tensor_desc().size(),
                                    op_desc_ptr->input_tensor_desc().data(), op_desc_ptr->input_tensor_data().data(),
                                    op_desc_ptr->output_tensor_desc().size(), op_desc_ptr->output_tensor_desc().data(),
                                    op_desc_ptr->output_tensor_data().data(), op_desc_ptr->acl_attr(), ACL_ENGINE_SYS,
                                    ACL_COMPILE_SYS, nullptr, stream_ptr);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Acl compile and execute failed, node:" << node->fullname_with_scope();
    return false;
  }

  MS_LOG(DEBUG) << "Success launch of node: " << node->fullname_with_scope();
  return true;
}

std::vector<TaskInfoPtr> AclKernelMod::GenTask(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                               const std::vector<AddressPtr> &, uint32_t) {
  return {};
}

void AclKernelMod::SyncData() {}
}  // namespace kernel
}  // namespace mindspore
