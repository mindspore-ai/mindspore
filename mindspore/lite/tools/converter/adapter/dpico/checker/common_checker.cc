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

#include "checker/common_checker.h"
#include <vector>
#include <string>
#include "common/op_enum.h"

namespace mindspore {
namespace dpico {
bool CommonChecker::Check(CNodePtr op, int32_t output_num, mindspore::Format format) {
  if (!CheckInputW(op, 1, format, kMaxInputWOf4Dims)) {
    MS_LOG(WARNING) << "input_w is not supported. " << op->fullname_with_scope();
    return false;
  }
  return true;
}

OpCheckerRegistrar g_AbsChecker("Abs", new CommonChecker());
OpCheckerRegistrar g_AddFusionChecker("AddFusion", new CommonChecker());
OpCheckerRegistrar g_SubFusionChecker("SubFusion", new CommonChecker());
OpCheckerRegistrar g_MulFusionChecker("MulFusion", new CommonChecker());
OpCheckerRegistrar g_DivFusionChecker("DivFusion", new CommonChecker());
OpCheckerRegistrar g_MaximumChecker("Maximum", new CommonChecker());
OpCheckerRegistrar g_MinimumChecker("Minimum", new CommonChecker());
OpCheckerRegistrar g_SquaredDifferenceChecker("SquaredDifference", new CommonChecker());
OpCheckerRegistrar g_X_DIV_YChecker("X_DIV_Y", new CommonChecker());
OpCheckerRegistrar g_X_LOG_YChecker("X_LOG_Y", new CommonChecker());
OpCheckerRegistrar g_BiasAddChecker("BiasAdd", new CommonChecker());
OpCheckerRegistrar g_NegChecker("Neg", new CommonChecker());
OpCheckerRegistrar g_EluChecker("Elu", new CommonChecker());
OpCheckerRegistrar g_NormalizeChecker("Normalize", new CommonChecker());
OpCheckerRegistrar g_PreluChecker("PReLUFusion", new CommonChecker());
OpCheckerRegistrar g_UpsampleChecker("Upsample", new CommonChecker());
OpCheckerRegistrar g_GatherChecker("Gather", new CommonChecker());
OpCheckerRegistrar g_ClipChecker("Clip", new CommonChecker());
OpCheckerRegistrar g_ShapeChecker("Shape", new CommonChecker());
OpCheckerRegistrar g_TileChecker("TileFusion", new CommonChecker());
OpCheckerRegistrar g_UnSqueezeChecker("Unsqueeze", new CommonChecker());
OpCheckerRegistrar g_BnllChecker("Bnll", new CommonChecker());
OpCheckerRegistrar g_CropChecker("Crop", new CommonChecker());
OpCheckerRegistrar g_RoiAlignChecker("RoiAlign", new CommonChecker());
OpCheckerRegistrar g_CastChecker("Cast", new CommonChecker());
OpCheckerRegistrar g_BiasChecker("Bias", new CommonChecker());
}  // namespace dpico
}  // namespace mindspore
