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

#include <vector>
#include <string>
#include <set>
#include <utility>
#include <map>
#include <unordered_map>
#include "schema/ops_generated.h"
#include "base/core_ops.h"
#include "include/lite_types.h"
#ifndef MINDSPORE_LITE_SRC_PASS_PARALLEL_SPLIT_STRATEGY_H_
#define MINDSPORE_LITE_SRC_PASS_PARALLEL_SPLIT_STRATEGY_H_

namespace mindspore {
namespace opt {
constexpr auto PARALLEL_NAME_SUFFIX = "_parallel";

constexpr auto kParallelPrimitiveIndex = 0;

const std::vector<int64_t> kSplitDefaultRatio = {0, 0};

// user's device to split, only split to cpu && gpu, no support npu
const std::vector<std::string> kSplitDevTypes = {"cpu", "gpu"};

using Strategys = std::vector<std::vector<std::vector<int64_t>>>;

constexpr auto kDeviceTypeNone = -1;
// strategy format is NHWC-KHWC
constexpr int32_t kAxisN = 0;
constexpr int32_t kAxisCIn = 3;
constexpr int32_t kAxisCOut = 0;
constexpr int32_t kAxisH = 1;
constexpr int32_t kAxisW = 2;

constexpr auto kDefaultBatch = 1;

constexpr auto kShapeN = 0;
constexpr auto kShapeH = 1;
constexpr auto kShapeW = 2;
constexpr auto kShapeC = 3;

constexpr auto kIndexH = 0;
constexpr auto kIndexW = 1;

constexpr auto kPadUp = 0;
constexpr auto kPadDown = 1;
constexpr auto kPadLeft = 2;
constexpr auto kPadRight = 3;

enum SplitMode {
  NoSplit = 0,
  SplitN = 1,
  SplitH = 2,
  SplitCIN = 3,
  SplitCOUT = 4,
};

struct SplitStrategy {
  Strategys strategys{};
  std::vector<std::string> dev_types{};
  size_t dev_num{0};
  SplitMode split_mode_{NoSplit};
};

// this is a map for key: <primitive,is_depth_wise>  value: parallel_op_name
const std::map<std::pair<PrimitivePtr, bool>, std::string> kParallelOpNames = {
  {{prim::kPrimConv2D, false}, "Conv2D"},
  {{prim::kPrimConv2DFusion, false}, "Conv2D"},
  {{prim::kPrimConv2D, true}, "DepthwiseConv2D"},
  {{prim::kPrimConv2DFusion, true}, "DepthwiseConv2D"}};

const std::map<std::string, lite::DeviceType> kSupportSplitedDevices = {
  {"cpu", lite::DeviceType::DT_CPU}, {"gpu", lite::DeviceType::DT_GPU}, {"npu", lite::DeviceType::DT_NPU}};

// this is a map for key: primitive  value: schema_primitive_id
const std::unordered_map<PrimitivePtr, std::pair<schema::PrimitiveType, TypeId>> kParallelSchemaId = {
  {prim::kPrimConv2D, {schema::PrimitiveType_Conv2DFusion, kNumberTypeFloat32}},
  {prim::kPrimConv2DFusion, {schema::PrimitiveType_Conv2DFusion, kNumberTypeFloat32}}};

// this is an artificial restriction that if user split conv, we limit total FLOPs bigger than
// 2 * output_H * output_W * (in_C * kW * kH +1) * out_C >= 100
// FLOPs ~= output_H * output_W * (in_C * kW * kH) * out_C
// FLOPs ~= (input_h/stride_h)*(input_w/stride_w)*in_C * kW * kH) * out_C
// etc. (12/1)*(12/1)*(1*3*3)*128/1024 = 162kFLPOs
constexpr auto kUserFLOPs = 100;
constexpr auto kPerFlops = 1024;

int64_t ApproximateFLOPs(const std::vector<int64_t> &strides, const std::vector<int64_t> &input_shae,
                         const std::vector<int64_t> &weight_shape);

std::unordered_map<std::string, opt::SplitStrategy> ParserSplitStrategy(
  const std::vector<int64_t> &parallel_compute_rates, const std::vector<std::string> &parallel_devices,
  SplitMode split_mode);

}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_PASS_PARALLEL_SPLIT_STRATEGY_H_
