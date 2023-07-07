/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#include "masked_fill.h"

#include <limits>
#include <complex>

#include "cpu_ops_kernel.h"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kMaskedFillInputNum = 3;
const uint32_t kMaskedFillOutputNum = 1;
const char *const kMaskedFill = "MaskedFill";
const int64_t kParallelDataNum = 6 * 1024;
const int64_t kParallelDataNumMid = 33 * 1024;
const int64_t kParallelDataNumSameShape = 7 * 1024;
const int64_t kParallelDataNumSameShapeMid = 35 * 1024;
}  // namespace

namespace aicpu {
uint32_t MaskedFillMaskCheck(const CpuKernelContext &ctx) {
  Tensor *mask = ctx.Input(kSecondInputIndex);
  DataType mask_type = mask->GetDataType();
  KERNEL_CHECK_FALSE(mask_type == DT_BOOL, KERNEL_STATUS_PARAM_INVALID, "[MaskedFill] mask type must be bool.");
  return KERNEL_STATUS_OK;
}

template <class T>
uint32_t GetValueFromVariousTypesMaskedFill(const CpuKernelContext &ctx, T &value) {
  Tensor *value_tensor = ctx.Input(kThirdInputIndex);
  auto value_type = value_tensor->GetDataType();
  auto value_ptr = value_tensor->GetData();
  double raw_value = 0;
  std::complex<double> raw_complex_value = 0;
  switch (value_type) {
    case DT_BOOL:
      raw_value = static_cast<double>(reinterpret_cast<bool *>(value_ptr)[0]);
      break;
    case DT_FLOAT16:
      raw_value = static_cast<double>(reinterpret_cast<Eigen::half *>(value_ptr)[0]);
      break;
    case DT_FLOAT:
      raw_value = static_cast<double>(reinterpret_cast<float *>(value_ptr)[0]);
      break;
    case DT_DOUBLE:
      raw_value = static_cast<double>(reinterpret_cast<double *>(value_ptr)[0]);
      break;
    case DT_INT8:
      raw_value = static_cast<double>(reinterpret_cast<int8_t *>(value_ptr)[0]);
      break;
    case DT_INT16:
      raw_value = static_cast<double>(reinterpret_cast<int16_t *>(value_ptr)[0]);
      break;
    case DT_INT32:
      raw_value = static_cast<double>(reinterpret_cast<int32_t *>(value_ptr)[0]);
      break;
    case DT_INT64:
      raw_value = static_cast<double>(reinterpret_cast<int64_t *>(value_ptr)[0]);
      break;
    case DT_UINT8:
      raw_value = static_cast<double>(reinterpret_cast<uint8_t *>(value_ptr)[0]);
      break;
    case DT_UINT16:
      raw_value = static_cast<double>(reinterpret_cast<uint16_t *>(value_ptr)[0]);
      break;
    case DT_UINT32:
      raw_value = static_cast<double>(reinterpret_cast<uint32_t *>(value_ptr)[0]);
      break;
    case DT_UINT64:
      raw_value = static_cast<double>(reinterpret_cast<uint64_t *>(value_ptr)[0]);
      break;
    case DT_COMPLEX64:
      raw_complex_value = static_cast<std::complex<double>>(reinterpret_cast<std::complex<float> *>(value_ptr)[0]);
      break;
    case DT_COMPLEX128:
      raw_complex_value = static_cast<std::complex<double>>(reinterpret_cast<std::complex<double> *>(value_ptr)[0]);
      break;
    default:
      KERNEL_LOG_ERROR("MaskedFill Invalid value type [%s]", DTypeStr(value_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if constexpr ((std::is_same_v<T, std::complex<float>>) || (std::is_same_v<T, std::complex<double>>)) {
    value = static_cast<T>(raw_complex_value);
  } else {
    if (raw_value > static_cast<double>(std::numeric_limits<T>::max()) ||
        raw_value < static_cast<double>(std::numeric_limits<T>::lowest())) {
      KERNEL_LOG_ERROR("[MaskedFill] value out of [%s] range [%f]", DTypeStr(value_type).c_str(), raw_value);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    value = static_cast<T>(raw_value);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t NoBcastCompute(CpuKernelContext &ctx) {
  auto x_ptr = reinterpret_cast<T *>(ctx.Input(kFirstInputIndex)->GetData());
  auto mask_ptr = reinterpret_cast<bool *>(ctx.Input(kSecondInputIndex)->GetData());
  T value = static_cast<T>(0);
  KERNEL_CHECK_FALSE(GetValueFromVariousTypesMaskedFill<T>(ctx, value) == KERNEL_STATUS_OK, KERNEL_STATUS_PARAM_INVALID,
                     "[MaskedFill] value cannot be safely converted to target "
                     "type without overflow.");
  auto y_ptr = reinterpret_cast<T *>(ctx.Output(kFirstOutputIndex)->GetData());

  int64_t x_data_num = ctx.Input(kFirstInputIndex)->NumElements();
  int64_t mask_data_num = ctx.Input(kSecondInputIndex)->NumElements();
  int64_t data_num = ctx.Output(kFirstOutputIndex)->NumElements();

  BcastShapeType type = BcastShapeType::SAME_SHAPE;
  if (x_data_num == mask_data_num) {
    type = BcastShapeType::SAME_SHAPE;
  } else if (x_data_num == 1) {
    type = BcastShapeType::X_ONE_ELEMENT;
  } else {
    type = BcastShapeType::Y_ONE_ELEMENT;
  }

  auto shard_masked_fill = [&](int64_t start, int64_t end) {
    switch (type) {
      case BcastShapeType::SAME_SHAPE:
        for (int64_t i = start; i < end; ++i) {
          y_ptr[i] = mask_ptr[i] ? value : x_ptr[i];
        }
        break;
      case BcastShapeType::X_ONE_ELEMENT:
        for (int64_t i = start; i < end; ++i) {
          y_ptr[i] = mask_ptr[i] ? value : x_ptr[0];
        }
        break;
      case BcastShapeType::Y_ONE_ELEMENT:
        for (int64_t i = start; i < end; ++i) {
          y_ptr[i] = mask_ptr[0] ? value : x_ptr[i];
        }
        break;
      default:
        KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(type));
        break;
    }
  };
  if (data_num >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

    if (data_num <= kParallelDataNumSameShapeMid) {
      max_core_num = std::min(max_core_num, static_cast<int64_t>(4));  // up to 4 cpu cores
    }
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num is 0");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_masked_fill),
                        "MaskedFill Compute failed.");
  } else {
    shard_masked_fill(0, data_num);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t BcastCompute(CpuKernelContext &ctx, const Bcast &bcast) {
  auto x_ptr = reinterpret_cast<T *>(ctx.Input(kFirstInputIndex)->GetData());
  auto mask_ptr = reinterpret_cast<bool *>(ctx.Input(kSecondInputIndex)->GetData());
  T value = static_cast<T>(0);
  KERNEL_CHECK_FALSE(GetValueFromVariousTypesMaskedFill<T>(ctx, value) == KERNEL_STATUS_OK, KERNEL_STATUS_PARAM_INVALID,
                     "[MaskedFill] value cannot be safely converted to target "
                     "type without overflow.");
  auto y_ptr = reinterpret_cast<T *>(ctx.Output(kFirstOutputIndex)->GetData());
  int64_t data_num = ctx.Output(kFirstOutputIndex)->NumElements();
  auto shard_masked_fill = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      y_ptr[i] = mask_ptr[bcast.GetBroadcastYIndex(i)] ? value : x_ptr[bcast.GetBroadcastXIndex(i)];
    }
  };
  if (data_num >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (data_num <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, static_cast<int64_t>(4));  // up to 4 cpu cores
    }
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num is 0");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_masked_fill),
                        "MaskedFill Compute failed.");
  } else {
    shard_masked_fill(0, data_num);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MaskedFillCompute(CpuKernelContext &ctx) {
  auto x = ctx.Input(kFirstInputIndex);
  auto x_shape = x->GetTensorShape()->GetDimSizes();
  int64_t x_elements_nums = x->NumElements();

  auto mask = ctx.Input(kSecondInputIndex);
  auto mask_shape = mask->GetTensorShape()->GetDimSizes();
  int64_t mask_elements_nums = mask->NumElements();

  bool isNeedBcast = (x_shape == mask_shape) || (x_elements_nums == 1) || (mask_elements_nums == 1);

  if (isNeedBcast) {
    return NoBcastCompute<T>(ctx);
  } else {
    Bcast bcast(x_shape, mask_shape);
    if (!bcast.IsValid()) {
      KERNEL_LOG_ERROR("[%s] broadcast failed.", ctx.GetOpType().c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
    return BcastCompute<T>(ctx, bcast);
  }
  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

uint32_t MaskedFillCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kMaskedFillInputNum, kMaskedFillOutputNum),
                      "MaskedFill check input and output number failed.");
  KERNEL_HANDLE_ERROR(MaskedFillMaskCheck(ctx), "MaskedFillMaskCheck failed.");
  auto data_type = ctx.Input(kFirstInputIndex)->GetDataType();
  switch (data_type) {
    case DT_BOOL:
      return MaskedFillCompute<bool>(ctx);
    case DT_FLOAT16:
      return MaskedFillCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return MaskedFillCompute<float>(ctx);
    case DT_DOUBLE:
      return MaskedFillCompute<double>(ctx);
    case DT_INT8:
      return MaskedFillCompute<int8_t>(ctx);
    case DT_INT16:
      return MaskedFillCompute<int16_t>(ctx);
    case DT_INT32:
      return MaskedFillCompute<int32_t>(ctx);
    case DT_INT64:
      return MaskedFillCompute<int64_t>(ctx);
    case DT_UINT8:
      return MaskedFillCompute<uint8_t>(ctx);
    case DT_UINT16:
      return MaskedFillCompute<uint16_t>(ctx);
    case DT_UINT32:
      return MaskedFillCompute<uint32_t>(ctx);
    case DT_UINT64:
      return MaskedFillCompute<uint64_t>(ctx);
    case DT_COMPLEX64:
      return MaskedFillCompute<std::complex<float>>(ctx);
    case DT_COMPLEX128:
      return MaskedFillCompute<std::complex<double>>(ctx);
    default:
      KERNEL_LOG_ERROR("MaskedFill kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kMaskedFill, MaskedFillCpuKernel);
}  // namespace aicpu