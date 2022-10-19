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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_PARAMS_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_PARAMS_H_
#include <map>
#include <string>
#include <vector>
#include <set>
#include "schema/inner/model_generated.h"
#include "ops/core_ops.h"
namespace mindspore::lite::quant {
enum WeightQuantType {
  FIXED_BIT_PER_CHANNEL = 0,
  FIXED_BIT_PER_LAYER = 1,
  MIXED_BIT_PER_LAYER = 2,
};
constexpr size_t k1Bit = 1;
constexpr size_t k2Bit = 2;
constexpr size_t k8Bit = 8;
constexpr size_t k10Bit = 10;
constexpr size_t k16Bit = 16;
constexpr size_t k32Bit = 32;
constexpr size_t kBitNumPerByte = 8;
constexpr size_t kMaxNum1024 = 1024;
constexpr size_t kMillisecondsBase = 10;
constexpr float kDelta = 0.1;
constexpr float kRatio = 10.0;
constexpr int kCpuBindMode = 1;
constexpr int kPrimIndex = 0;
constexpr int kPrimOffset = 1;
constexpr int kU8ZeroPointOffset = 128;
constexpr int kQuantRange = 127;
constexpr int kInt8LeftRange = -128;
constexpr int kInt8RightRange = 127;
constexpr int kMinIterations = 40;

const std::set<PrimitivePtr> kHasBiasOperator = {prim::kPrimConv2DFusion, prim::kPrimConv2dTransposeFusion,
                                                 prim::kPrimMatMulFusion, prim::kPrimFullConnection,
                                                 prim::kPrimLayerNormFusion};
const std::set<PrimitivePtr> kUint8toFP32Operator = {prim::kPrimDetectionPostProcess};
const std::set<TypeId> kFullQuantDType = {kNumberTypeInt8, kNumberTypeUInt8, kNumberTypeFloat32};

enum ActivationQuantizedMethod {
  MAX_MIN = 0,
  KL = 1,
  REMOVAL_OUTLIER = 2,
};

enum TargetDevice {
  CPU,
  KIRIN,
  NVGPU,
  DSP,
};

enum DebugMode {
  FAST,
  DETAIL,
};

enum CastNodeType {
  kNone,
  kQuant,
  kDeQuant,
};

enum InsertDirection {
  FORWARD,
  BACKWARD,
};

struct CommonQuantParam {
  schema::QuantType quant_type = schema::QuantType_QUANT_NONE;
  int bit_num = 8;
  int min_quant_weight_size = 0;
  int min_quant_weight_channel = 16;
  bool is_debug = false;
  std::string debug_info_save_path;
  DebugMode debug_mode = DETAIL;
  std::set<std::string> skip_quant_node;
  int thread_num = 4;
  bool enable_encode = true;
};

struct MixedBitWeightQuantParam {
  double init_scale = 0.02;
  bool auto_tune = false;
  bool use_cv_data = false;
  int max_iterations = kMinIterations;
};

struct FullQuantParam {
  ActivationQuantizedMethod activation_quant_method = MAX_MIN;
  bool bias_correction = true;
  bool per_channel = true;
  TargetDevice target_device = CPU;
};
}  // namespace mindspore::lite::quant

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_PARAMS_H_
