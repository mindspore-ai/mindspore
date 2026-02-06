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

#ifndef MINDSPORE_OPS_OP_ENUM_H_
#define MINDSPORE_OPS_OP_ENUM_H_
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

#include "mindapi/base/macros.h"

namespace mindspore {
namespace ops {
OPS_API int64_t StringToEnumImpl(const std::string &op_name, const std::string &arg_name,
                                 const std::string &enum_string);

// Only the current mindspore/core/include/mindapi/base/types.h and other files do not have
// corresponding enumerations and then add new enumerations.
// The `enum` is used here instead of `enum class` because the current backend enum is
// represented by `int`. The `enum` is more convenient than `enum class` compare with int.
enum Direction : int64_t { UNIDIRECTIONAL = 0 };

enum CellType : int64_t { CELL_TYPE_LSTM = 0 };

enum Group : int64_t { SYNC_BN_GROUP0 = 0 };

enum InterpolationMode : int64_t { BILINEAR = 0, NEAREST = 1, BICUBIC = 2 };

enum Indexing : int64_t { IJ = 0, XY = 1 };

enum RoundingMode : int64_t { ROUND = 0, TRUNC = 1, FLOOR = 2, CEIL = 3 };

enum NormMode : int64_t { BACKWARD = 0, FORWARD = 1, ORTHO = 2 };

enum GridSamplerPaddingMode : int64_t { ZEROS = 0, BORDER = 1, REFLECTION = 2 };

enum KVCacheAlignMode : int64_t { RIGHT = 0, LEFT = 1 };

enum FASInputLayoutMode : int64_t {
  BSH = 0,
  BNSD = 1,
  SBH = 2,
  BSND = 3,
  TND = 4,
  TH = 5,
  NSD = 6,
  SH = 7,
  BNSD_BSND = 8
};

enum PagedAttentionKVCacheQuantMode : int64_t { DEFAULT = 0, PERTOKEN = 1 };

enum PagedAttentionMaskMode : int64_t { MASK_DEFAULT = 0, TRAPEZOIDAL = 1 };

enum MLAMode : int64_t { MASK_NONE = 0, MASK_NORM = 1, MASK_ALIBI = 2, MASK_SPEC = 3, MASK_FREE = 4 };

enum ErrorMode : int64_t { CYCLE = 0, SPECIFIC = 1 };

enum FlipMode : int64_t { BITFLIP = 0, BITFLIP_DESIGNED = 1, MULTIPLY = 2, MULTIPLY_MAX = 3 };

enum LinalgQrMode : int64_t { REDUCED = 0, COMPLETE = 1, R = 2 };

enum Approximate : int64_t { NONE = 0, TANH = 1 };

enum Device : int64_t { DEVICE_ASCEND = 0, DEVICE_NPU_LOWER = 1, DEVICE_CPU = 2, DEVICE_CPU_LOWER = 3 };

enum QuantMode : int64_t { STATIC = 0, DYNAMIC = 1 };

enum CompareOp : int64_t { EQUAL = 0, GREATER_THAN = 1, LESS_THAN = 2 };

enum SignalOpMode : int64_t { SET = 0, ADD = 1 };
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_OPS_OP_ENUM_H_
