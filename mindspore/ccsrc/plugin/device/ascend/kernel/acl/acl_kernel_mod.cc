/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "ir/tensor.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/common_utils.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "plugin/device/ascend/kernel/acl/acl_kernel_utils.h"

namespace mindspore {
namespace kernel {
int AclKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                         const std::vector<KernelTensorPtr> &outputs,
                         const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  if (!common::AnfAlgo::IsDynamicShape(cnode)) {
    MS_LOG(EXCEPTION) << "The node is not dynamic shape: " << cnode->fullname_with_scope();
  }

  // Update input size list
  for (size_t i = 0; i < input_size_list_.size(); ++i) {
    TypeId type_id = AnfAlgo::GetInputDeviceDataType(node, i);
    auto type_size = GetTypeByte(TypeIdToType(type_id));
    auto shape = AnfAlgo::GetInputDeviceShape(node, i);
    if (IsDynamic(shape)) {
      MS_LOG(ERROR) << "Please check infer op shape before resize, error input index is:" << i;
      return 1;
    }
    input_size_list_[i] = type_size * SizeOf(shape);
  }

  // Update output size list
  AscendKernelMod::UpdateOutputSizeList();
  return 0;
}

bool AclKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                          const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (stream_ptr == nullptr) {
    MS_LOG(ERROR) << "stream_ptr should not be nullptr.";
    return false;
  }
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(INFO) << "Start launch of node: " << cnode->fullname_with_scope();

  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(node);
  if (input_num != inputs.size() || output_num != outputs.size()) {
    MS_LOG(EXCEPTION) << "Node's input or output size is invalid! node is " << node->DebugString();
  }

  auto op_desc_ptr = std::make_unique<AclOpDesc>(node_name_);
  MS_EXCEPTION_IF_NULL(op_desc_ptr);
  op_desc_ptr->AddInputTensor(node, input_num, inputs, input_size_list_, node_name_);
  op_desc_ptr->AddOutputTensor(node, output_num, outputs, output_size_list_, node_name_);

  auto primitive = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &attr : primitive->attrs()) {
    auto &value = attr.second;
    MS_EXCEPTION_IF_NULL(value);
    op_desc_ptr->AddTensorAttr(attr.first, value, node_name_);
  }

  if (aclopSetCompileFlag(aclCompileFlag::ACL_OP_COMPILE_FUZZ)) {
    MS_LOG(ERROR) << "Acl set compile mode failed! op_name is " << node_name_;
    return false;
  }

  MS_LOG(INFO) << "Start aclopCompileAndExecute of node: " << cnode->fullname_with_scope();
  bool ret =
    aclopCompileAndExecute(const_cast<char *>(node_name_.c_str()), input_num, op_desc_ptr->input_tensor_desc().data(),
                           op_desc_ptr->input_tensor_data().data(), output_num,
                           op_desc_ptr->output_tensor_desc().data(), op_desc_ptr->output_tensor_data().data(),
                           op_desc_ptr->acl_attr(), ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr, stream_ptr);
  if (ret) {
    MS_LOG(ERROR) << "Acl compile and execute failed! op_name is " << node_name_ << " and op info is "
                  << node->DebugString();
    return false;
  }

  MS_LOG(INFO) << "Success launch of node: " << cnode->fullname_with_scope();
  return true;
}

std::vector<TaskInfoPtr> AclKernelMod::GenTask(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                               const std::vector<AddressPtr> &, uint32_t) {
  return {};
}

void AclKernelMod::SyncData() {}
}  // namespace kernel
}  // namespace mindspore
