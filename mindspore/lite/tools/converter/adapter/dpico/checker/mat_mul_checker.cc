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

#include "checker/mat_mul_checker.h"
#include <vector>
#include <string>
#include "common/anf_util.h"
#include "common/op_attr.h"
#include "common/op_enum.h"

namespace mindspore {
namespace dpico {
namespace {
bool CheckInputShapeForMatrix(const api::CNodePtr &cnode, const api::PrimitivePtr &primitive) {
  ShapeVector input_1_shape;
  ShapeVector input_2_shape;
  if (GetInputShapeFromCNode(cnode, kInputIndex1, &input_1_shape) != RET_OK ||
      GetInputShapeFromCNode(cnode, kInputIndex2, &input_2_shape) != RET_OK) {
    MS_LOG(ERROR) << "get input shape failed. " << cnode->fullname_with_scope();
    return false;
  }
  if (!input_1_shape.empty() && !input_2_shape.empty()) {
    if (input_1_shape.size() != input_2_shape.size()) {
      return false;
    }
    if (input_1_shape.size() > kDims2) {
      for (size_t i = 0; i < input_1_shape.size() - kInputIndex2; i++) {
        if (input_1_shape.at(i) != 1 || input_2_shape.at(i) != 1) {
          return false;
        }
      }
    }
    (void)primitive->AddAttr(kDim1, api::MakeValue<int64_t>(input_1_shape.at(input_1_shape.size() - kInputIndex2)));
    (void)primitive->AddAttr(kDim2, api::MakeValue<int64_t>(input_1_shape.at(input_1_shape.size() - 1)));
    (void)primitive->AddAttr(kDim3, api::MakeValue<int64_t>(input_2_shape.at(input_2_shape.size() - kInputIndex2)));
  }
  return true;
}
bool CheckInputShapeForFc(const api::CNodePtr &cnode) {
  ShapeVector input_shape;
  if (GetInputShapeFromCNode(cnode, kInputIndex1, &input_shape) == RET_OK) {
    return input_shape.size() == dpico::kDims2 && input_shape.at(0) == 1;
  }
  return false;
}
}  // namespace
bool MatMulChecker::Check(api::CNodePtr op, int32_t output_num, mindspore::Format format) {
  if (!CheckInputW(op, 1, format, kMaxInputWOf4Dims)) {
    MS_LOG(WARNING) << "input_w is not supported. " << op->fullname_with_scope();
    return false;
  }
  auto primitive = api::GetValueNode<api::PrimitivePtr>(op->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr." << op->fullname_with_scope();
    return false;
  }
  if (primitive->GetAttr(ops::kTransposeA) != nullptr) {
    auto transpose_a = api::GetValue<bool>(primitive->GetAttr(ops::kTransposeA));
    if (transpose_a) {
      return false;
    }
  }
  if (op->inputs().size() < kInputIndex3) {
    MS_LOG(ERROR) << "Matmul should have 2 inputs at least, but is " << (op->inputs().size() - 1);
    return false;
  }
  bool transpose_b = false;
  if (primitive->GetAttr(ops::kTransposeB) != nullptr) {
    transpose_b = api::GetValue<bool>(primitive->GetAttr(ops::kTransposeB));
  } else {
    (void)primitive->AddAttr(ops::kTransposeB, api::MakeValue<bool>(false));
  }
  if (!HasOfflineData(op->input(kInputIndex1))) {
    if (!HasOfflineData(op->input(kInputIndex2))) {
      if (transpose_b) {
        if (!CheckInputShapeForMatrix(op, primitive)) {
          return false;
        }
        (void)primitive->AddAttr(kOperatorType, api::MakeValue("Matrix"));
        return true;
      } else {
        return false;
      }
    } else {
      if (CheckInputShapeForFc(op)) {
        (void)primitive->AddAttr(kOperatorType, api::MakeValue("FullConnection"));
        return true;
      } else {
        MS_LOG(WARNING) << "only supports input N = 1 by dpico. " << op->fullname_with_scope();
        return false;
      }
    }
  }
  return false;
}

OpCheckerRegistrar g_GemmChecker("Gemm", new MatMulChecker());
OpCheckerRegistrar g_MatMulChecker("MatMulFusion", new MatMulChecker());
}  // namespace dpico
}  // namespace mindspore
