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

#ifndef MINDSPORE_LITE_PROVIDERS_DPICO_COMMON_CUSTOM_ENUM_H_
#define MINDSPORE_LITE_PROVIDERS_DPICO_COMMON_CUSTOM_ENUM_H_

namespace mindspore {
namespace lite {
enum AclModelType : size_t { kCnn = 0, kRoi = 1, kRecurrent = 2 };
enum DetectParam : size_t { kNmsThreshold = 0, kScoreThreshold = 1, kMinHeight = 2, kMinWidth = 3 };
enum DetectBoxParam : size_t {
  kTopLeftX = 0,
  kDetectBoxParamBegin = kTopLeftX,
  kTopLeftY = 1,
  kBottomRightX = 2,
  kBottomRightY = 3,
  kScore = 4,
  kClassId = 5,
  kDetectBoxParamEnd
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_PROVIDERS_DPICO_COMMON_CUSTOM_ENUM_H_
