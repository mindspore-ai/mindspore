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

#include "checker/arithmetic_checker.h"
#include <string>
#include <vector>
#include <unordered_set>
#include "common/anf_util.h"
#include "common/check_base.h"
#include "common/op_enum.h"

namespace mindspore {
namespace dpico {
bool ArithmeticChecker::Check(api::CNodePtr op, int32_t output_num, mindspore::Format format) {
  if (!CheckInputW(op, 1, format, kMaxInputWOf4Dims)) {
    MS_LOG(WARNING) << "input_w is not supported. " << op->fullname_with_scope();
    return false;
  }
  auto prim = api::GetValueNode<api::PrimitivePtr>(op->input(0));
  MS_CHECK_TRUE_MSG(prim != nullptr, false, "prim is nullptr." << op->fullname_with_scope());

  if (op->inputs().size() != kDims3) {
    MS_LOG(ERROR) << op->fullname_with_scope() << " only supports 2 inputs.";
    return false;
  }

  if (HasOfflineData(op->input(kInputIndex1)) || HasOfflineData(op->input(kInputIndex2))) {
    MS_LOG(DEBUG) << "parameter node don't need to check. " << op->fullname_with_scope();
    return true;
  }

  ShapeVector input_1_shape;
  ShapeVector input_2_shape;
  if (GetInputShapeFromCNode(op, kInputIndex1, &input_1_shape) != RET_OK ||
      GetInputShapeFromCNode(op, kInputIndex2, &input_2_shape) != RET_OK) {
    MS_LOG(ERROR) << "get input shape failed. " << op->fullname_with_scope();
    return false;
  }

  if (input_1_shape == input_2_shape) {
    return true;
  }

  std::unordered_set<std::string> supported_broadcast_ops = {"AddFusion", "SubFusion", "MulFusion"};
  std::string op_type_name;
  if (GetPrimitiveType(op, &op_type_name) != RET_OK) {
    MS_LOG(ERROR) << "get cnode primitive type failed:" << op->fullname_with_scope();
    return false;
  }
  if (supported_broadcast_ops.find(op_type_name) == supported_broadcast_ops.end()) {
    MS_LOG(WARNING) << op->fullname_with_scope() << " don't support broadcast by dpico.";
    return false;
  }
  if (input_1_shape.size() != input_2_shape.size()) {
    MS_LOG(WARNING) << op->fullname_with_scope() << " input dim_size should be the same.";
    return false;
  }
  if (input_1_shape.size() == kDims4) {
    bool has_same_val = false;
    for (size_t i = 1; i < kDims4; i++) {
      if (input_1_shape.at(i) == input_2_shape.at(i)) {
        has_same_val = true;
      }
    }
    if (!has_same_val) {
      MS_LOG(WARNING) << "don't support broadcast chw by dpico. " << op->fullname_with_scope();
      return false;
    }
  }
  return true;
}

OpCheckerRegistrar g_AddFusionChecker("AddFusion", new ArithmeticChecker());
OpCheckerRegistrar g_SubFusionChecker("SubFusion", new ArithmeticChecker());
OpCheckerRegistrar g_MulFusionChecker("MulFusion", new ArithmeticChecker());
OpCheckerRegistrar g_DivFusionChecker("DivFusion", new ArithmeticChecker());
OpCheckerRegistrar g_MaximumChecker("Maximum", new ArithmeticChecker());
OpCheckerRegistrar g_MinimumChecker("Minimum", new ArithmeticChecker());
OpCheckerRegistrar g_SquaredDifferenceChecker("SquaredDifference", new ArithmeticChecker());
OpCheckerRegistrar g_X_DIV_YChecker("X_DIV_Y", new ArithmeticChecker());
OpCheckerRegistrar g_X_LOG_YChecker("X_LOG_Y", new ArithmeticChecker());
OpCheckerRegistrar g_BiasAddChecker("BiasAdd", new ArithmeticChecker());
}  // namespace dpico
}  // namespace mindspore
