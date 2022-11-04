/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "checker/concat_checker.h"
#include <vector>
#include <string>
#include <climits>
#include "common/anf_util.h"
#include "common/op_enum.h"

namespace mindspore {
namespace dpico {
namespace {
bool CheckConcatInputW(const ShapeVector &input_shape, int64_t axis, int64_t input_w) {
  bool supported = true;
  if (input_shape.size() == kDims2 && input_w > UINT_MAX) {
    supported = false;
  } else if (static_cast<size_t>(axis) == kAxis1 && input_w > UINT_MAX) {
    supported = false;
  } else if (static_cast<size_t>(axis) == kAxis2 && input_w > kMaxInputWOf2Dims) {
    supported = false;
  } else if (static_cast<size_t>(axis) == kAxis3 && input_w > kMaxInputWOf4Dims) {
    supported = false;
  }
  if (!supported) {
    MS_LOG(WARNING) << "input_w " << input_w << " is unsupported by dpico.";
  }
  return supported;
}
}  // namespace
bool ConcatChecker::Check(api::CNodePtr op, int32_t output_num, mindspore::Format format) {
  auto primitive = api::GetValueNode<api::PrimitivePtr>(op->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr." << op->fullname_with_scope();
    return false;
  }
  int64_t axis = 0;
  auto axis_ptr = primitive->GetAttr(ops::kAxis);
  if (axis_ptr != nullptr) {
    axis = api::GetValue<int64_t>(axis_ptr);
    if (axis < kAxisLowerBound || axis > kAxisUpperBound) {
      MS_LOG(WARNING) << op->fullname_with_scope() << "'s axis should in range [-4, 3], but in fact it's " << axis;
      return false;
    }
  }

  if (op->inputs().size() - 1 > static_cast<size_t>(kMaxBottomNum)) {
    MS_LOG(WARNING) << op->fullname_with_scope() << "'s bottom size:" << (op->inputs().size() - 1)
                    << " is greater than " << kMaxBottomNum;
    return false;
  }

  std::vector<int64_t> input_shape;
  for (size_t i = 1; i < op->inputs().size(); i++) {
    auto input = op->input(i);
    if (input->cast<api::ParameterPtr>() != nullptr && input->cast<api::ParameterPtr>()->has_default()) {
      MS_LOG(WARNING) << "there is offline data in concat, which dpico is unsupported. " << op->fullname_with_scope();
      return false;
    }
    if (GetInputShapeFromCNode(op, i, &input_shape) == RET_OK && !input_shape.empty()) {
      int64_t input_w;
      if (GetWidth(input_shape, format, &input_w) != RET_OK) {
        MS_LOG(ERROR) << "get input_w failed." << op->fullname_with_scope();
        return false;
      }
      if (!CheckConcatInputW(input_shape, axis, input_w)) {
        return false;
      }
    }
  }
  return true;
}

OpCheckerRegistrar g_ConcatChecker("Concat", new ConcatChecker());
}  // namespace dpico
}  // namespace mindspore
