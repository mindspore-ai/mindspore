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
#include "mindspore/ccsrc/backend/common/graph_kernel/core/convert_op_input_attr.h"
#include "mindspore/ccsrc/backend/common/graph_kernel/core/graph_kernel_callback.h"

#include "mindspore/core/ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/anf_utils.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"

namespace mindspore::graphkernel {
std::map<std::string, HashSet<size_t>> ConvertOpUtils::op_idx_info_ = {{prim::kPrimReshape->name(), {1}},
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
bool ConvertOpUtils::ConstInputToAttr(const CNodePtr &cnode, const HashSet<size_t> &input_idx) {
  AnfNodePtrList new_inputs;
  auto primitive = GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);
  auto input_names = primitive->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    if (op_idx_info_.find(primitive->name()) == op_idx_info_.end()) {
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
  new_inputs.push_back(inputs[0]);
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
      if (value != nullptr) {
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
      } else {
        MS_LOG(ERROR) << input_names_vec[i] << "'s Value is null!";
      }
    } else {
      new_inputs.push_back(inputs[i + 1]);
    }
  }
  if (new_inputs.size() != inputs.size()) {
    cnode->set_inputs(new_inputs);
    auto cb = Callback::Instance();
    MS_EXCEPTION_IF_NULL(cb);
    cb->SetBasicNodeKernelInfo(
      cnode, {{cb->GetOutputShape(cnode, 0), cb->GetOutputType(cnode, 0), cb->GetOutputFormat(cnode, 0)}});
  }
  return true;
}
void ConvertOpUtils::ConvertAttrToInput(const AnfNodePtr &node) {
#ifndef MSLITE_ENABLE_GRAPH_KERNEL
  if (Callback::Instance()->GetTargetFromContext() == kAscendDevice) {
    return;
  }
#endif
  auto cnode = dyn_cast<CNode>(node);
  auto primitive = GetCNodePrimitive(cnode);
  if (primitive == nullptr) {
    return;
  }
  auto attr2input_map = ConvertOpUtils::GetOpIndexInfo();
  if (attr2input_map.count(primitive->name()) != 0) {
    auto input_names = primitive->GetAttr(kAttrInputNames);
    AnfNodePtrList inputs = cnode->inputs();
    AnfNodePtrList new_inputs{inputs[0]};
    auto input_names_vec = GetValue<std::vector<std::string>>(input_names);
    auto attrs_map = attr2input_map[primitive->name()];
    size_t j = 1;
    for (size_t i = 0; i < input_names_vec.size(); ++i) {
      if (attrs_map.count(i) != 0) {
        auto value = primitive->GetAttr(input_names_vec[i]);
        ValueNodePtr value_node;
        // Adaptive for TupleGetItem, this op can not make attr to a tensor input.
        if (primitive->name() == prim::kPrimTupleGetItem->name()) {
          value_node = std::make_shared<ValueNode>(value);
          value_node->set_abstract(value->ToAbstract());
        } else if (value->isa<Scalar>()) {
          auto value_scalar = GetValue<int64_t>(value);
          auto tensor_ptr = std::make_shared<tensor::Tensor>(value_scalar, kInt64);
          value_node = std::make_shared<ValueNode>(tensor_ptr);
          value_node->set_abstract(tensor_ptr->ToAbstract());
        } else if (value->isa<ValueTuple>()) {
          auto value_tuple = GetValue<std::vector<int64_t>>(value);
          auto tensor_ptr = std::make_shared<tensor::Tensor>(value_tuple, kInt64);
          value_node = std::make_shared<ValueNode>(tensor_ptr);
          value_node->set_abstract(tensor_ptr->ToAbstract());
        } else {
          MS_LOG(EXCEPTION) << "The value should be a scalar or value tuple";
        }
        Callback::Instance()->SetEmptyKernelInfo(value_node);
        new_inputs.push_back(value_node);
      } else {
        if (j >= inputs.size()) {
          MS_LOG(EXCEPTION) << "Index " << j << " is larger than input size [" << inputs.size() << "]";
        }
        new_inputs.push_back(inputs[j]);
        j++;
      }
    }
    cnode->set_inputs(new_inputs);
  }
}
}  // namespace mindspore::graphkernel
