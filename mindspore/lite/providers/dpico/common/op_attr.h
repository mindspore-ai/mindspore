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

#ifndef MINDSPORE_LITE_PROVIDERS_DPICO_COMMON_OP_ATTR_H_
#define MINDSPORE_LITE_PROVIDERS_DPICO_COMMON_OP_ATTR_H_

namespace mindspore {
constexpr auto kAclConfigPath = "AclConfigPath";
constexpr auto kDetectionPostProcess = "DetectionPostProcess";
constexpr auto kDpico = "dpico";
constexpr auto kInputsShape = "inputs_shape";
constexpr auto kGTotalT = "GTotalT";
constexpr auto kMaxRoiNum = "MaxRoiNum";
constexpr auto kMinHeight = "MinHeight";
constexpr auto kMinWidth = "MinWidth";
constexpr auto kModelSharingKey = "inner_sharing_workspace";
constexpr auto kModelSharingPrepareKey = "inner_calc_workspace_size";
constexpr auto kModelSharingSection = "inner_common";
constexpr auto kNetType = "net_type";
constexpr auto kNmsThreshold = "NmsThreshold";
constexpr auto kOutputsFormat = "outputs_format";
constexpr auto kOutputsShape = "outputs_shape";
constexpr auto kScoreThreshold = "ScoreThreshold";
}  // namespace mindspore
#endif  // MINDSPORE_LITE_PROVIDERS_DPICO_COMMON_OP_ATTR_H_
