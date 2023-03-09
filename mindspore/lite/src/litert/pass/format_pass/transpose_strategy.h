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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_PASS_TRANSPOSE_STRATEGY_H_
#define MINDSPORE_LITE_SRC_RUNTIME_PASS_TRANSPOSE_STRATEGY_H_

#include <map>
#include <set>
#include <vector>
#include <unordered_map>
#include "src/litert/kernel_exec.h"
#include "src/litert/pass/format_pass/pass_utils.h"

namespace mindspore::lite::pass {
// bool value determines whether the kernel has axis attribute or not.
// If bool value is true, the single kernel can be processd only for NHWC2NCHW or NCHW2NHWC.
static const std::unordered_map<schema::PrimitiveType, bool> dynamic_format_kernel_lists = {
  {schema::PrimitiveType_Abs, false},
  {schema::PrimitiveType_Activation, false},
  {schema::PrimitiveType_AddFusion, false},
  {schema::PrimitiveType_AddN, false},
  {schema::PrimitiveType_ArgMaxFusion, true},
  {schema::PrimitiveType_ArgMinFusion, true},
  {schema::PrimitiveType_Cast, false},
  {schema::PrimitiveType_Ceil, false},
  {schema::PrimitiveType_Clip, false},
  {schema::PrimitiveType_Concat, true},
  {schema::PrimitiveType_Cos, false},
  {schema::PrimitiveType_Crop, true},
  {schema::PrimitiveType_DivFusion, false},
  {schema::PrimitiveType_Elu, false},
  {schema::PrimitiveType_Eltwise, false},
  {schema::PrimitiveType_Equal, false},
  {schema::PrimitiveType_ExpFusion, false},
  {schema::PrimitiveType_Floor, false},
  {schema::PrimitiveType_FloorDiv, false},
  {schema::PrimitiveType_FloorMod, false},
  {schema::PrimitiveType_Greater, false},
  {schema::PrimitiveType_GreaterEqual, false},
  {schema::PrimitiveType_Less, false},
  {schema::PrimitiveType_LessEqual, false},
  {schema::PrimitiveType_Log, false},
  {schema::PrimitiveType_LogicalAnd, false},
  {schema::PrimitiveType_LogicalNot, false},
  {schema::PrimitiveType_LogicalOr, false},
  {schema::PrimitiveType_Maximum, false},
  {schema::PrimitiveType_Minimum, false},
  {schema::PrimitiveType_Mod, false},
  {schema::PrimitiveType_MulFusion, false},
  {schema::PrimitiveType_Neg, false},
  {schema::PrimitiveType_NotEqual, false},
  {schema::PrimitiveType_PowFusion, false},
  {schema::PrimitiveType_QuantDTypeCast, false},
  {schema::PrimitiveType_RealDiv, false},
  {schema::PrimitiveType_Round, false},
  {schema::PrimitiveType_Rsqrt, false},
  {schema::PrimitiveType_Sin, false},
  {schema::PrimitiveType_SliceFusion, true},
  {schema::PrimitiveType_Softmax, true},
  {schema::PrimitiveType_Split, true},
  {schema::PrimitiveType_Sqrt, false},
  {schema::PrimitiveType_Squeeze, true},
  {schema::PrimitiveType_Square, false},
  {schema::PrimitiveType_SquaredDifference, false},
  {schema::PrimitiveType_Stack, true},
  {schema::PrimitiveType_StridedSlice, true},
  {schema::PrimitiveType_SubFusion, false},
  {schema::PrimitiveType_Unsqueeze, true},
  {schema::PrimitiveType_Unstack, true},
  {schema::PrimitiveType_LogSoftmax, true},
  {schema::PrimitiveType_Erf, false},
};

static TransInfoPair NHWC2NCHWTrans = {Format::NHWC, Format::NCHW};
static TransInfoPair NCHW2NHWCTrans = {Format::NCHW, Format::NHWC};

class TransposeStrategy {
 public:
  TransposeStrategy() = default;
  ~TransposeStrategy() = default;

  size_t GetTransCount(const std::vector<kernel::KernelExec *> &kernels, TransInfoPair *trans_info);
  bool CheckFusion(const kernel::KernelExec *kernel, TransInfoPair *pre_trans, TransInfoPair *post_trans);
  int ChangeKernelAxis(kernel::KernelExec *kernel, const TransInfoPair &post_trans);
};
}  // namespace mindspore::lite::pass
#endif  // MINDSPORE_LITE_SRC_RUNTIME_PASS_TRANSPOSE_STRATEGY_H_
