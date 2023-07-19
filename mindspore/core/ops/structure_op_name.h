/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_BASE_STRUCTURE_OP_NAME_H_
#define MINDSPORE_CORE_BASE_STRUCTURE_OP_NAME_H_

namespace mindspore {
// String
constexpr auto kStringEqOpName = "string_eq";
constexpr auto kStringLtOpName = "string_lt";
constexpr auto kStringGtOpName = "string_gt";
constexpr auto kStringLeOpName = "string_le";
constexpr auto kStringGeOpName = "string_ge";
constexpr auto kStringConcatOpName = "string_concat";
constexpr auto kStringNotOpName = "string_not";
constexpr auto kStringInOpName = "string_in";
constexpr auto kStringMulOpName = "string_mul";
constexpr auto kStringGetItemOpName = "string_getitem";

constexpr auto kGetNextOpName = "GetNext";
constexpr auto kGetNextFromQueueOpName = "GetNextFromQueue";
constexpr auto kDynamicGetNextV2OpName = "DynamicGetNextV2";

// Statements
constexpr auto kVmapStackAssignOpName = "VmapStackAssign";
constexpr auto kVmapUnstackAssignOpName = "VmapUnstackAssign";
constexpr auto kSliceGetItemOpName = "SliceGetItem";
constexpr auto kCondOpName = "Cond";
constexpr auto kDynamicBroadcastGradientArgsOpName = "DynamicBroadcastGradientArgs";

constexpr auto kHistogramFixedWidthOpName = "HistogramFixedWidth";
constexpr auto kHistogramFixedWidthDOpName = "HistogramFixedWidthD";
constexpr auto kStackDestroyOpName = "StackDestroy";
constexpr auto kStackInitOpName = "StackInit";
constexpr auto kStackPopOpName = "StackPop";
constexpr auto kStackPushOpName = "StackPush";
constexpr auto kStopGradientOpName = "StopGradient";
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_STRUCTURE_OP_NAME_H_
