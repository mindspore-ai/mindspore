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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_COMMON_OP_ENUM_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_COMMON_OP_ENUM_H_

namespace mindspore {
namespace dpico {
constexpr size_t kDims1 = 1;
constexpr size_t kDims2 = 2;
constexpr size_t kDims3 = 3;
constexpr size_t kDims4 = 4;
constexpr size_t kAxis1 = 1;
constexpr size_t kAxis2 = 2;
constexpr size_t kAxis3 = 3;
constexpr size_t kInputIndex1 = 1;
constexpr size_t kInputIndex2 = 2;
constexpr size_t kInputIndex3 = 3;
constexpr size_t kInputIndex4 = 4;
constexpr size_t kInputIndex5 = 5;
constexpr int kMaxInputWOf4Dims = 4096;
constexpr int kMaxInputWOf2Dims = 16384;
constexpr int kMaxNumOutput = 32768;
constexpr int kMaxTopNum = 32;
constexpr int kMaxBottomNum = 32;
constexpr int kAxisLowerBound = -4;
constexpr int kAxisUpperBound = 3;
constexpr size_t kMaxLineCount = 9999;
}  // namespace dpico
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_COMMON_OP_ENUM_H_
