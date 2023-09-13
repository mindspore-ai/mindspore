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
#include "backend/common/graph_kernel/core/value_depend_op_utils.h"

#include <memory>
#include <vector>

#include "base/base.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/lite_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "ops/primitive_c.h"
#include "utils/anf_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::graphkernel {
const std::unordered_map<std::string, HashSet<size_t>> &ValueDependOpUtils::GetOpIndexInfo() {
  static const std::unordered_map<std::string, HashSet<size_t>> op_idx_info_ = {
    {prim::kPrimReshape->name(), {1}},
    {prim::kPrimReduceMax->name(), {1}},
    {prim::kPrimExpandDims->name(), {1}},
    {prim::kPrimReduceMin->name(), {1}},
    {prim::kPrimReduceSum->name(), {1}},
    {prim::kPrimTranspose->name(), {1}},
    {prim::kPrimTile->name(), {1}},
    {prim::kPrimReduceMean->name(), {1}},
    {prim::kPrimSlice->name(), {1, 2}},
    {prim::kPrimStridedSlice->name(), {1, 2, 3}},
    {prim::kPrimOneHot->name(), {1}},
    {prim::kPrimReduceFusion->name(), {1}},
    {prim::kPrimConstantOfShape->name(), {0}},
    {prim::kPrimGather->name(), {2}},
    {prim::kPrimTupleGetItem->name(), {1}},
    {prim::kPrimUnsortedSegmentSum->name(), {2}},
    {prim::kPrimCumSum->name(), {1}}};
  return op_idx_info_;
}

bool ValueDependOpUtils::IsConstInput(const AnfNodePtr &node) {
  auto prim = GetCNodePrimitive(node);
  if (prim != nullptr) {
    const auto &op_index_info = GetOpIndexInfo();
    auto iter = op_index_info.find(prim->name());
    if (iter != op_index_info.end()) {
      auto inputs = node->cast<CNodePtr>()->inputs();
      for (const auto &i : iter->second) {
        if (i + 1 < inputs.size() && inputs[i + 1] != nullptr) {
          auto input_node = inputs[i + 1];
          ValuePtr value = nullptr;
          if (input_node->isa<ValueNode>()) {
            auto value_node = input_node->cast<ValueNodePtr>();
            value = value_node->value();
          } else if (input_node->isa<Parameter>()) {
            auto parameter_node = input_node->cast<ParameterPtr>();
            value = parameter_node->abstract()->BuildValue();
          }
          if (value == nullptr) {
            return false;
          }
          auto tensor = value->cast<tensor::TensorPtr>();
          if (tensor != nullptr && tensor->data().const_data() == nullptr) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

bool ValueDependOpUtils::AddConstInputToAttr(const CNodePtr &cnode, const HashSet<size_t> &input_idx) {
  auto primitive = GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);
  primitive = primitive->Clone();
  MS_EXCEPTION_IF_NULL(primitive);
  auto input_names = primitive->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    const auto &op_index_info = GetOpIndexInfo();
    if (op_index_info.find(primitive->name()) == op_index_info.end()) {
      MS_LOG(INFO) << "input_names are nullptr in cnode[" + cnode->DebugString() + "]";
      return false;
    }
    const auto &op_primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
    auto const iter = op_primc_fns.find(primitive->name());
    if (iter == op_primc_fns.end()) {
      MS_LOG(INFO) << "Can't find " << primitive->name() << " op's primitive in primitiveC!";
      return false;
    }
    auto prim = iter->second();
    if (prim != nullptr) {
      input_names = prim->GetAttr(kAttrInputNames);
      (void)primitive->AddAttr(kAttrInputNames, prim->GetAttr(kAttrInputNames));
      (void)primitive->AddAttr(kAttrOutputNames, prim->GetAttr(kAttrOutputNames));
    }
  }
  auto input_names_vec = GetValue<std::vector<std::string>>(input_names);
  auto inputs = cnode->inputs();
  for (size_t i = 0; i < inputs.size() - 1; ++i) {
    auto input_node = inputs[i + 1];
    MS_EXCEPTION_IF_NULL(input_node);
    if (input_idx.count(i) != 0) {
      if (i >= input_names_vec.size()) {
        MS_LOG(INFO) << "Index " << i << " is larger than input names size [" << input_names_vec.size() << "]";
        return false;
      }
      ValuePtr value = nullptr;
      if (input_node->isa<ValueNode>()) {
        auto value_node = input_node->cast<ValueNodePtr>();
        value = value_node->value();
      } else if (input_node->isa<Parameter>()) {
        auto parameter_node = input_node->cast<ParameterPtr>();
        value = parameter_node->abstract()->BuildValue();
      }
      if (value == nullptr) {
        MS_LOG(DEBUG) << input_names_vec[i] << "'s Value is null.";
        return false;
      }
      if (value->isa<ValueAny>()) {
        MS_LOG(DEBUG) << input_names_vec[i] << "'s Value is ValueAny.";
        return false;
      }
      if (!value->isa<tensor::Tensor>()) {
        primitive->set_attr(input_names_vec[i], value);
        continue;
      }
      auto value_vector = CheckAndConvertUtils::CheckTensorIntValue(input_names_vec[i], value, primitive->name());
      auto tensor = value->cast<tensor::TensorPtr>();
      auto tensor_shape = tensor->shape_c();
      if (tensor_shape.empty()) {
        primitive->set_attr(input_names_vec[i], MakeValue(value_vector[0]));
      } else {
        primitive->set_attr(input_names_vec[i], MakeValue(value_vector));
      }
    }
  }
  cnode->set_input(0, std::make_shared<ValueNode>(primitive));
  return true;
}

}  // namespace mindspore::graphkernel
