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

#include "plugin/device/gpu/kernel/arrays/tensor_scatter_arithmetic_gpu_kernel.h"

namespace mindspore {
namespace kernel {
#define REG_TENSOR_SCATTER_OP(OP_ID, MS_IN_TYPE_0, MS_IN_TYPE_1, MS_IN_TYPE_2, MS_OUT_TYPE, REAL_IN_OUT_TYPE, \
                              REAL_INDEX_TYPE)                                                                \
  MS_REG_GPU_KERNEL_TWO(OP_ID,                                                                                \
                        KernelAttr()                                                                          \
                          .AddInputAttr(MS_IN_TYPE_0)                                                         \
                          .AddInputAttr(MS_IN_TYPE_1)                                                         \
                          .AddInputAttr(MS_IN_TYPE_2)                                                         \
                          .AddOutputAttr(MS_OUT_TYPE),                                                        \
                        TensorScatterArithmeticGpuKernelMod, REAL_IN_OUT_TYPE, REAL_INDEX_TYPE)

// TensorScatterUpdate
REG_TENSOR_SCATTER_OP(TensorScatterUpdate, kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeFloat16, kNumberTypeFloat16,
                      half, int)
REG_TENSOR_SCATTER_OP(TensorScatterUpdate, kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32,
                      float, int)
REG_TENSOR_SCATTER_OP(TensorScatterUpdate, kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeFloat64, kNumberTypeFloat64,
                      double, int)
REG_TENSOR_SCATTER_OP(TensorScatterUpdate, kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt8, kNumberTypeInt8, char,
                      int)
REG_TENSOR_SCATTER_OP(TensorScatterUpdate, kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeUInt8, kNumberTypeUInt8,
                      uchar, int)
REG_TENSOR_SCATTER_OP(TensorScatterUpdate, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int,
                      int)
REG_TENSOR_SCATTER_OP(TensorScatterUpdate, kNumberTypeBool, kNumberTypeInt32, kNumberTypeBool, kNumberTypeBool, bool,
                      int)

REG_TENSOR_SCATTER_OP(TensorScatterUpdate, kNumberTypeFloat16, kNumberTypeInt64, kNumberTypeFloat16, kNumberTypeFloat16,
                      half, int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterUpdate, kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeFloat32, kNumberTypeFloat32,
                      float, int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterUpdate, kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeFloat64, kNumberTypeFloat64,
                      double, int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterUpdate, kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt8, kNumberTypeInt8, char,
                      int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterUpdate, kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeUInt8, kNumberTypeUInt8,
                      uchar, int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterUpdate, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt32, int,
                      int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterUpdate, kNumberTypeBool, kNumberTypeInt64, kNumberTypeBool, kNumberTypeBool, bool,
                      int64_t)

// TensorScatterMin, no support bool data type
REG_TENSOR_SCATTER_OP(TensorScatterMin, kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeFloat16, kNumberTypeFloat16,
                      half, int)
REG_TENSOR_SCATTER_OP(TensorScatterMin, kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32,
                      float, int)
REG_TENSOR_SCATTER_OP(TensorScatterMin, kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeFloat64, kNumberTypeFloat64,
                      double, int)
REG_TENSOR_SCATTER_OP(TensorScatterMin, kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt8, kNumberTypeInt8, char, int)
REG_TENSOR_SCATTER_OP(TensorScatterMin, kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeUInt8, kNumberTypeUInt8, uchar,
                      int)
REG_TENSOR_SCATTER_OP(TensorScatterMin, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int,
                      int)

REG_TENSOR_SCATTER_OP(TensorScatterMin, kNumberTypeFloat16, kNumberTypeInt64, kNumberTypeFloat16, kNumberTypeFloat16,
                      half, int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterMin, kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeFloat32, kNumberTypeFloat32,
                      float, int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterMin, kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeFloat64, kNumberTypeFloat64,
                      double, int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterMin, kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt8, kNumberTypeInt8, char,
                      int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterMin, kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeUInt8, kNumberTypeUInt8, uchar,
                      int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterMin, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt32, int,
                      int64_t)

// TensorScatterMax, no support bool data type
REG_TENSOR_SCATTER_OP(TensorScatterMax, kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeFloat16, kNumberTypeFloat16,
                      half, int)
REG_TENSOR_SCATTER_OP(TensorScatterMax, kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32,
                      float, int)
REG_TENSOR_SCATTER_OP(TensorScatterMax, kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeFloat64, kNumberTypeFloat64,
                      double, int)
REG_TENSOR_SCATTER_OP(TensorScatterMax, kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt8, kNumberTypeInt8, char, int)
REG_TENSOR_SCATTER_OP(TensorScatterMax, kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeUInt8, kNumberTypeUInt8, uchar,
                      int)
REG_TENSOR_SCATTER_OP(TensorScatterMax, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int,
                      int)
REG_TENSOR_SCATTER_OP(TensorScatterMax, kNumberTypeFloat16, kNumberTypeInt64, kNumberTypeFloat16, kNumberTypeFloat16,
                      half, int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterMax, kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeFloat32, kNumberTypeFloat32,
                      float, int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterMax, kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeFloat64, kNumberTypeFloat64,
                      double, int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterMax, kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt8, kNumberTypeInt8, char,
                      int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterMax, kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeUInt8, kNumberTypeUInt8, uchar,
                      int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterMax, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt32, int,
                      int64_t)

// TensorScatterAdd, no support bool data type
REG_TENSOR_SCATTER_OP(TensorScatterAdd, kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeFloat16, kNumberTypeFloat16,
                      half, int)
REG_TENSOR_SCATTER_OP(TensorScatterAdd, kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32,
                      float, int)
REG_TENSOR_SCATTER_OP(TensorScatterAdd, kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeFloat64, kNumberTypeFloat64,
                      double, int)
REG_TENSOR_SCATTER_OP(TensorScatterAdd, kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt8, kNumberTypeInt8, char, int)
REG_TENSOR_SCATTER_OP(TensorScatterAdd, kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeUInt8, kNumberTypeUInt8, uchar,
                      int)
REG_TENSOR_SCATTER_OP(TensorScatterAdd, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int,
                      int)
REG_TENSOR_SCATTER_OP(TensorScatterAdd, kNumberTypeFloat16, kNumberTypeInt64, kNumberTypeFloat16, kNumberTypeFloat16,
                      half, int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterAdd, kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeFloat32, kNumberTypeFloat32,
                      float, int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterAdd, kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeFloat64, kNumberTypeFloat64,
                      double, int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterAdd, kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt8, kNumberTypeInt8, char,
                      int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterAdd, kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeUInt8, kNumberTypeUInt8, uchar,
                      int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterAdd, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt32, int,
                      int64_t)

// TensorScatterSub, no support bool data type
REG_TENSOR_SCATTER_OP(TensorScatterSub, kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeFloat16, kNumberTypeFloat16,
                      half, int)
REG_TENSOR_SCATTER_OP(TensorScatterSub, kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32,
                      float, int)
REG_TENSOR_SCATTER_OP(TensorScatterSub, kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeFloat64, kNumberTypeFloat64,
                      double, int)
REG_TENSOR_SCATTER_OP(TensorScatterSub, kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt8, kNumberTypeInt8, char, int)
REG_TENSOR_SCATTER_OP(TensorScatterSub, kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeUInt8, kNumberTypeUInt8, uchar,
                      int)
REG_TENSOR_SCATTER_OP(TensorScatterSub, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int,
                      int)
REG_TENSOR_SCATTER_OP(TensorScatterSub, kNumberTypeFloat16, kNumberTypeInt64, kNumberTypeFloat16, kNumberTypeFloat16,
                      half, int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterSub, kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeFloat32, kNumberTypeFloat32,
                      float, int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterSub, kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeFloat64, kNumberTypeFloat64,
                      double, int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterSub, kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt8, kNumberTypeInt8, char,
                      int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterSub, kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeUInt8, kNumberTypeUInt8, uchar,
                      int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterSub, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt32, int,
                      int64_t)

// TensorScatterMul, no support bool data type
REG_TENSOR_SCATTER_OP(TensorScatterMul, kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeFloat16, kNumberTypeFloat16,
                      half, int)
REG_TENSOR_SCATTER_OP(TensorScatterMul, kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32,
                      float, int)
REG_TENSOR_SCATTER_OP(TensorScatterMul, kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeFloat64, kNumberTypeFloat64,
                      double, int)
REG_TENSOR_SCATTER_OP(TensorScatterMul, kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt8, kNumberTypeInt8, char, int)
REG_TENSOR_SCATTER_OP(TensorScatterMul, kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeUInt8, kNumberTypeUInt8, uchar,
                      int)
REG_TENSOR_SCATTER_OP(TensorScatterMul, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int,
                      int)
REG_TENSOR_SCATTER_OP(TensorScatterMul, kNumberTypeFloat16, kNumberTypeInt64, kNumberTypeFloat16, kNumberTypeFloat16,
                      half, int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterMul, kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeFloat32, kNumberTypeFloat32,
                      float, int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterMul, kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeFloat64, kNumberTypeFloat64,
                      double, int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterMul, kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt8, kNumberTypeInt8, char,
                      int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterMul, kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeUInt8, kNumberTypeUInt8, uchar,
                      int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterMul, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt32, int,
                      int64_t)

// TensorScatterDiv, no support bool data type
REG_TENSOR_SCATTER_OP(TensorScatterDiv, kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeFloat16, kNumberTypeFloat16,
                      half, int)
REG_TENSOR_SCATTER_OP(TensorScatterDiv, kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32,
                      float, int)
REG_TENSOR_SCATTER_OP(TensorScatterDiv, kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeFloat64, kNumberTypeFloat64,
                      double, int)
REG_TENSOR_SCATTER_OP(TensorScatterDiv, kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt8, kNumberTypeInt8, char, int)
REG_TENSOR_SCATTER_OP(TensorScatterDiv, kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeUInt8, kNumberTypeUInt8, uchar,
                      int)
REG_TENSOR_SCATTER_OP(TensorScatterDiv, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int,
                      int)
REG_TENSOR_SCATTER_OP(TensorScatterDiv, kNumberTypeFloat16, kNumberTypeInt64, kNumberTypeFloat16, kNumberTypeFloat16,
                      half, int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterDiv, kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeFloat32, kNumberTypeFloat32,
                      float, int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterDiv, kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeFloat64, kNumberTypeFloat64,
                      double, int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterDiv, kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt8, kNumberTypeInt8, char,
                      int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterDiv, kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeUInt8, kNumberTypeUInt8, uchar,
                      int64_t)
REG_TENSOR_SCATTER_OP(TensorScatterDiv, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt32, int,
                      int64_t)
}  // namespace kernel
}  // namespace mindspore
