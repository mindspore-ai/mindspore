/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/const_to_attr_strided_slice_grad.h"
#include <memory>
#include <vector>
#include <string>
#include "ir/primitive.h"
#include "utils/ms_context.h"
#include "include/common/utils/utils.h"
#include "abstract/abstract_value.h"
#include "backend/common/pass/const_input_to_attr.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
const size_t strides_index = 5;

bool GetStridesValues(const CNodePtr &strided_slice_grad, ValuePtrList *strides_values) {
  MS_EXCEPTION_IF_NULL(strided_slice_grad);
  MS_EXCEPTION_IF_NULL(strides_values);
  constexpr size_t kSizeChange = 6;
  if (strided_slice_grad->size() < kSizeChange) {
    MS_LOG(DEBUG) << "Op strided_slice_grad's inputs size less than 6, graph not changed";
    return false;
  }
  auto strides_input = strided_slice_grad->input(strides_index);
  MS_EXCEPTION_IF_NULL(strides_input);
  auto strides_value_node = strides_input->cast<ValueNodePtr>();
  if (strides_value_node == nullptr) {
    MS_LOG(DEBUG) << "strides is not a value node.";
    return false;
  }
  auto value = strides_value_node->value();
  if (value == nullptr) {
    MS_LOG(DEBUG) << "strides has no value.";
    return false;
  }
  auto value_tuple = value->cast<ValueTuplePtr>();
  if (value_tuple == nullptr) {
    MS_LOG(DEBUG) << "strides is not a value tuple.";
    return false;
  }
  *strides_values = value_tuple->value();
  return true;
}

bool CheckValues(const ValuePtrList &strides_values) {
  if (strides_values.empty()) {
    MS_LOG(DEBUG) << "strides_values is empty";
    return false;
  }
  for (auto &value : strides_values) {
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<Scalar>()) {
      auto scalar = value->cast<ScalarPtr>();
      MS_EXCEPTION_IF_NULL(scalar);
      if (scalar->isa<Int32Imm>()) {
        if (GetValue<int>(scalar) != 1) {
          MS_LOG(DEBUG) << "StridedSliceGrad has no 1 value";
          return false;
        }
      } else if (scalar->isa<Int64Imm>()) {
        if (GetValue<int64_t>(scalar) != 1) {
          MS_LOG(DEBUG) << "StridedSliceGrad has no 1 value";
          return false;
        }
      } else {
        MS_LOG(DEBUG) << "Strides value is not an integer";
        return false;
      }
    } else {
      MS_LOG(DEBUG) << "The value " << value << "of tuple is not a scalar";
      return false;
    }
  }
  return true;
}

bool CheckAttrs(const CNodePtr &strided_slice_grad) {
  MS_EXCEPTION_IF_NULL(strided_slice_grad);
  if (!common::AnfAlgo::HasNodeAttr(kAttrNewAxisMask, strided_slice_grad) ||
      !common::AnfAlgo::HasNodeAttr(kAttrShrinkAxisMask, strided_slice_grad)) {
    MS_LOG(INFO) << "new_axis_mask or shrink_axis_mask not exist in cnode[" + strided_slice_grad->DebugString() + "]";
    return false;
  }
  auto new_axis_mask = common::AnfAlgo::GetNodeAttr<int64_t>(strided_slice_grad, kAttrNewAxisMask);
  auto shrink_axis_mask = common::AnfAlgo::GetNodeAttr<int64_t>(strided_slice_grad, kAttrShrinkAxisMask);
  if (new_axis_mask != 0 || shrink_axis_mask != 0) {
    MS_LOG(INFO) << "new_axis_mask or shrink_axis_mask not equal 0";
    return false;
  }
  return true;
}
}  // namespace

const BaseRef ConstToAttrStridedSliceGradPass::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto strided_slice_grad_prim = std::make_shared<Primitive>(kStridedSliceGradOpName);
  return VectorRef({strided_slice_grad_prim, Xs});
}

const AnfNodePtr ConstToAttrStridedSliceGradPass::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                          const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto strided_slice_grad = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(strided_slice_grad);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);

  if (ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    if (!CheckAttrs(strided_slice_grad)) {
      MS_LOG(INFO) << "Check strided_slice_grad's attrs failed, graph not changed";
      return nullptr;
    }

    ValuePtrList strides_values;
    if (!GetStridesValues(strided_slice_grad, &strides_values)) {
      return nullptr;
    }

    if (!CheckValues(strides_values)) {
      MS_LOG(INFO) << "Check strides' values failed, graph not changed";
      return nullptr;
    }
  }

  return ConstInputToAttr(strided_slice_grad, {1, 2, 3, 4});
}
}  // namespace opt
}  // namespace mindspore
