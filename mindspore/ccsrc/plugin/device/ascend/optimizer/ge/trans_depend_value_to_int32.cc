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
#include "plugin/device/ascend/optimizer/ge/trans_depend_value_to_int32.h"

#include <string>
#include <vector>
#include <memory>
#include <set>
#include "ops/conv_pool_op_name.h"
#include "ops/array_op_name.h"
#include "abstract/ops/primitive_infer_map.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "utils/anf_utils.h"

namespace mindspore::opt {
namespace {
tensor::TensorPtr TransValueToInt32(const AnfNodePtr &input) {
  auto ori_value_node = input->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(ori_value_node);
  auto ori_value = ori_value_node->value();
  MS_EXCEPTION_IF_NULL(ori_value);
  if (!ori_value->isa<tensor::Tensor>()) {
    MS_LOG(INFO) << "Value is not tensor";
    return nullptr;
  }
  auto tensor = ori_value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);
  // case1: tensor no (data & empty)
  if (tensor->data().const_data() == nullptr && !tensor->has_user_data(kTensorValueIsEmpty)) {
    MS_LOG(INFO) << "Const input data ptr is null and no empty tensor.";
    return nullptr;
  }
  TypePtr data_type = tensor->Dtype();
  MS_EXCEPTION_IF_NULL(data_type);
  TypeId type_id = data_type->type_id();
  if (type_id != kNumberTypeInt64) {
    MS_LOG(INFO) << "Tensor type is not int64, it is " << TypeIdLabel(type_id);
    return nullptr;
  }
  tensor::TensorPtr new_tensor = std::make_shared<tensor::Tensor>(kInt32->type_id(), tensor->shape());
  auto *ori_data = static_cast<int64_t *>(tensor->data_c());
  auto *new_data = static_cast<int32_t *>(new_tensor->data_c());
  for (int i = 0; i < SizeToInt(tensor->data().size()); ++i) {
    new_data[i] = static_cast<int32_t>(ori_data[i]);
  }
  // add device info
  TensorTypePtr tensor_type = std::make_shared<TensorType>(kInt32);
  tensor::DeviceInfo device_info{kOpFormat_DEFAULT, tensor_type};
  new_tensor->set_device_info(device_info);
  return new_tensor;
}
}  // namespace

const AnfNodePtr TransDependValueToInt32::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                  const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(node);

  // if node has depend value
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  static const std::set<std::string> kSkipNodeName = {kScatterNdOpName, kAdaptiveAvgPool2DGradOpName};
  auto node_name = AnfUtils::GetCNodeName(cnode);
  if (kSkipNodeName.find(node_name) != kSkipNodeName.end()) {
    return nullptr;
  }
  auto depend_set = abstract::GetValueDependArgIndices(cnode);
  if (depend_set.empty()) {
    return nullptr;
  }
  // trans depend value from int64 to int32 in call acl mode
  auto inputs = cnode->inputs();
  std::vector<AnfNodePtr> new_inputs = {inputs.at(0)};
  const auto &manager = kernel_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &users = manager->node_users();
  for (int i = 0; i < SizeToInt(inputs.size()) - 1; ++i) {
    auto input = inputs.at(i + 1);
    if (depend_set.count(i) == 0) {
      (void)new_inputs.emplace_back(input);
      continue;
    }
    if (!input->isa<ValueNode>()) {
      (void)new_inputs.emplace_back(input);
      continue;
    }
    auto tensor = TransValueToInt32(input);
    if (!tensor) {
      (void)new_inputs.emplace_back(input);
      continue;
    }
    auto new_value_node = kernel_graph->NewValueNode(tensor);
    // If the input used by other node, which can not be removed
    if (auto it = users.find(input); it != users.end() && it->second.size() <= kIndex1) {
      kernel_graph->RemoveNodeFromGraph(input);
    }
    (void)new_inputs.emplace_back(new_value_node);
  }
  auto new_node = kernel_graph->NewCNodeWithInfos(new_inputs, cnode);
  new_node->set_abstract(cnode->abstract());
  new_node->set_inputs(new_inputs);
  return new_node;
}
}  // namespace mindspore::opt
