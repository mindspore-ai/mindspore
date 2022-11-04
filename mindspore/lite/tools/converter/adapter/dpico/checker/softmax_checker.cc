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

#include "checker/softmax_checker.h"
#include <vector>
#include <string>
#include "common/anf_util.h"
#include "common/op_enum.h"
#include "common/check_base.h"

namespace mindspore {
namespace dpico {
namespace {
constexpr int kMaxVectorC = 65536;
bool CheckVectorAndTensorChannel(const std::vector<int64_t> &input_shape, mindspore::Format format) {
  if (input_shape.size() == kDims2) {
    int64_t vec_channel;
    if (GetVectorChannel(input_shape, &vec_channel) != RET_OK) {
      MS_LOG(ERROR) << "get vector channel failed";
      return false;
    }
    if (vec_channel > kMaxVectorC) {
      return false;
    }
  } else if (input_shape.size() == kDims4) {
    int64_t tensor_channel;
    if (GetTensorChannel(input_shape, format, &tensor_channel) != RET_OK) {
      MS_LOG(ERROR) << "get tensor channel failed";
      return false;
    }
    if (tensor_channel > kMaxInputWOf4Dims) {
      return false;
    }
  }
  return true;
}
}  // namespace
bool SoftmaxChecker::Check(api::CNodePtr op, int32_t output_num, mindspore::Format format) {
  std::vector<int64_t> input_shape;
  if (GetInputShapeFromCNode(op, kInputIndex1, &input_shape) == RET_OK && !input_shape.empty()) {
    auto primitive = api::GetValueNode<api::PrimitivePtr>(op->input(0));
    MS_CHECK_TRUE_MSG(primitive != nullptr, false, "primitive is nullptr.");
    auto axis_ptr = primitive->GetAttr(ops::kAxis);
    if (axis_ptr != nullptr) {
      auto axis_data = api::GetValue<std::vector<int64_t>>(axis_ptr);
      auto axis = axis_data[0];
      if (axis < 0) {
        int64_t input_shape_len = static_cast<int64_t>(input_shape.size());
        axis = (axis + input_shape_len) % input_shape_len;
        std::vector<int64_t> axes = {axis};
        (void)primitive->AddAttr(ops::kAxis, api::MakeValue(axes));
      }
      if (static_cast<size_t>(axis) != kAxis1 && static_cast<size_t>(axis) != kAxis2 &&
          static_cast<size_t>(axis) != kAxis3) {
        MS_LOG(WARNING) << "axis val only supports 1/2/3 by dpico. " << op->fullname_with_scope();
        return false;
      }
      if (axis == 1) {
        if (!CheckVectorAndTensorChannel(input_shape, format)) {
          return false;
        }
      } else if (static_cast<size_t>(axis) == kAxis3) {
        int64_t input_w;
        if (GetWidth(input_shape, format, &input_w) != RET_OK) {
          MS_LOG(ERROR) << "get input_w failed";
          return false;
        }
        if (input_w > kMaxInputWOf4Dims) {
          return false;
        }
      }
    }
  }
  return true;
}

OpCheckerRegistrar g_SoftmaxChecker("Softmax", new SoftmaxChecker());
}  // namespace dpico
}  // namespace mindspore
