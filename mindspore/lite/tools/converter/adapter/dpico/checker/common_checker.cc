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
bool CommonChecker::Check(api::CNodePtr op, int32_t output_num, mindspore::Format format) {
  if (!CheckInputW(op, 1, format, kMaxInputWOf4Dims)) {
    MS_LOG(WARNING) << "input_w is not supported. " << op->fullname_with_scope();
    return false;
  }
  return true;
}

OpCheckerRegistrar g_AbsChecker("Abs", new CommonChecker());
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
