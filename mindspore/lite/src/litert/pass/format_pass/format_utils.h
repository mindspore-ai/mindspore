/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_LITERT_PASS_FORMAT_PASS_FORMAT_UTILS_H_
#define MINDSPORE_LITE_SRC_LITERT_PASS_FORMAT_PASS_FORMAT_UTILS_H_

#include <string>
#include <unordered_map>
#include <vector>

namespace mindspore::lite::pass {
static const std::unordered_map<std::string, std::vector<size_t>> cloud_format_kernel_list = {
  {"AvgPool", {0}},
  {"MaxPool", {0}},
  {"BatchNorm", {0}},
  {"BatchNormWithActivation", {0}},
  {"BatchNormWithAddAndActivation", {0}},
  {"BatchToSpace", {0}},
  {"Conv2D", {0, 1}},
  {"Conv2DTranspose", {0, 1}},
  {"DepthToSpace", {0}},
  {"FusedBatchNorm", {0}},
  {"InstanceNorm", {0}},
  {"LRN", {0}},
  {"PReLU", {0}},
  {"Resize", {0}},
  {"ROIPooling", {0}},
  {"SGD", {0}},
  {"SpaceToBatch", {0}},
  {"SpaceToBatchND", {0}},
  {"SpaceToDepth", {0}},
};
}  // namespace mindspore::lite::pass
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_NNACL_FORMAT_PASS_H_
