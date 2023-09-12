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

#include "ms_kernel/reduce_mean.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include "common/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kReduceMean = "ReduceMean";
constexpr uint32_t kIndex0 = 0;

#define REDUCEMEAN_COMPUTE_CASE(DTYPE, TYPE1, TYPE2, CTX)    \
  case (DTYPE): {                                            \
    uint32_t result = ReduceMeanCompute<TYPE1, TYPE2>(CTX);  \
    if (result != KERNEL_STATUS_OK) {                        \
      KERNEL_LOG_ERROR("ReduceMean kernel compute failed."); \
      return result;                                         \
    }                                                        \
    break;                                                   \
  }

#define REDUCEMEAN_COMPUTE_CASE_CP(DTYPE, TYPE1, TYPE2, CTX)        \
  case (DTYPE): {                                                   \
    uint32_t result = ReduceMeanCompute_Complex<TYPE1, TYPE2>(CTX); \
    if (result != KERNEL_STATUS_OK) {                               \
      KERNEL_LOG_ERROR("ReduceMean kernel compute failed.");        \
      return result;                                                \
    }                                                               \
    break;                                                          \
  }

#define REDUCEMEAN_COMPUTE_CASE_ALL(TYPE, CTX)                               \
  REDUCEMEAN_COMPUTE_CASE_CP(DT_COMPLEX64, std::complex<float>, TYPE, CTX)   \
  REDUCEMEAN_COMPUTE_CASE_CP(DT_COMPLEX128, std::complex<double>, TYPE, CTX) \
  REDUCEMEAN_COMPUTE_CASE(DT_DOUBLE, double, TYPE, CTX)                      \
  REDUCEMEAN_COMPUTE_CASE(DT_FLOAT, float, TYPE, CTX)                        \
  REDUCEMEAN_COMPUTE_CASE(DT_FLOAT16, Eigen::half, TYPE, CTX)                \
  REDUCEMEAN_COMPUTE_CASE(DT_INT8, int8_t, TYPE, CTX)                        \
  REDUCEMEAN_COMPUTE_CASE(DT_INT16, int16_t, TYPE, CTX)                      \
  REDUCEMEAN_COMPUTE_CASE(DT_INT32, int32_t, TYPE, CTX)                      \
  REDUCEMEAN_COMPUTE_CASE(DT_INT64, int64_t, TYPE, CTX)                      \
  REDUCEMEAN_COMPUTE_CASE(DT_UINT8, uint8_t, TYPE, CTX)                      \
  REDUCEMEAN_COMPUTE_CASE(DT_UINT16, uint16_t, TYPE, CTX)                    \
  REDUCEMEAN_COMPUTE_CASE(DT_UINT32, uint32_t, TYPE, CTX)                    \
  REDUCEMEAN_COMPUTE_CASE(DT_UINT64, uint64_t, TYPE, CTX)
}  // namespace

namespace aicpu {
template <typename T>
T ComplexDiv(T sum, int64_t num) {
  T res;
  auto real = sum.real();
  auto imag = sum.imag();
  res.real(real / num);
  res.imag(imag / num);
  return res;
}

uint32_t ReduceMeanCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  uint32_t input_num = ctx.GetInputsSize();
  uint32_t output_num = ctx.GetOutputsSize();
  if (input_num != 2 || output_num != 1) {
    KERNEL_LOG_ERROR("The number of input or output parameters does not match.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *input_data = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(input_data->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input[0] failed.")
  Tensor *axes_data = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(axes_data->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input[1] failed.")
  Tensor *output_data = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output_data->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output[0] failed.");
  DataType data_type = ctx.Input(0)->GetDataType();
  DataType axes_type = ctx.Input(1)->GetDataType();
  switch (axes_type) {
    case DT_INT32:
      switch (data_type) {
        REDUCEMEAN_COMPUTE_CASE_ALL(int32_t, ctx)
        default:
          KERNEL_LOG_ERROR("Input[0] data type[%s] not supported.", DTypeStr(data_type).c_str());
          return KERNEL_STATUS_PARAM_INVALID;
      }
      break;
    case DT_INT64:
      switch (data_type) {
        REDUCEMEAN_COMPUTE_CASE_ALL(int64_t, ctx)
        default:
          KERNEL_LOG_ERROR("Input[0] data type[%s] not supported.", DTypeStr(data_type).c_str());
          return KERNEL_STATUS_PARAM_INVALID;
      }
      break;
    default:
      KERNEL_LOG_ERROR("Input[1] data type[%s] not supported.", DTypeStr(axes_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

/*
Calculate the mean of the corresponding dimension data
Rule: except for the specified dimension, a set of data with other
dimensions unchanged participate in the calculation of a mean.
e.g.    input_x : float array[2][2][2]={1,2,3,4,5,6,7,8}
        axes : [1 , 2]
        output:[2.5, 6.5]
        2.5 is calculated from array[0][0][0], array[0][0][1],
                                array[0][1][0] and array[0][1][1]
The same group of data addresses involved in calculating the
mean consists of one same base address and different offset addresses
input_data_address = base_address + offset_address
*/
template <typename T1, typename T2>
uint32_t ReduceMeanCpuKernel::ReduceMeanCompute(const CpuKernelContext &ctx) {
  Tensor *input_data = ctx.Input(0);
  auto input_data_addr = reinterpret_cast<T1 *>(input_data->GetData());
  const int64_t input_data_num = input_data->NumElements();
  auto input_data_shape = input_data->GetTensorShape();
  const int32_t input_data_dims = input_data_shape->GetDims();
  std::vector<int64_t> input_data_dimsize = input_data_shape->GetDimSizes();
  std::vector<int64_t> dims_addr(input_data_dims);
  dims_addr[input_data_dims - 1] = 1;
  int64_t addr_tmp = 1;
  for (int32_t i = input_data_dims - 2; i > -1; i--) {
    addr_tmp *= input_data_dimsize[i + 1];
    dims_addr[i] = addr_tmp;
  }
  Tensor *output_data = ctx.Output(0);
  auto output_data_shape = output_data->GetTensorShape();
  auto output_data_addr = reinterpret_cast<T1 *>(output_data->GetData());
  const int64_t output_data_num = output_data->NumElements();
  Tensor *axes_data = ctx.Input(1);
  auto axes_data_addr = reinterpret_cast<T2 *>(axes_data->GetData());
  int64_t axes_data_num = axes_data->NumElements();
  // Check the effectiveness of the value of axes
  for (int64_t i = 0; i < axes_data_num; i++) {
    if ((*(axes_data_addr + i) >= input_data_dims) || (*(axes_data_addr + i) < -input_data_dims)) {
      KERNEL_LOG_ERROR("The value of axes is incorrect.");
      return KERNEL_STATUS_PARAM_INVALID;
    } else if (*(axes_data_addr + i) < 0) {
      *(axes_data_addr + i) += input_data_dims;
    }
  }
  std::sort(axes_data_addr, axes_data_addr + axes_data_num);
  std::vector<T2> axes_data_norepeat;
  for (int64_t i = 0; i < axes_data_num - 1; i++) {
    T2 value = axes_data_addr[i];
    if (value == axes_data_addr[i + 1]) {
      axes_data_num--;
      continue;
    }
    axes_data_norepeat.push_back(value);
  }
  axes_data_norepeat.push_back(axes_data_addr[axes_data_num - 1]);
  // deal with attr
  auto attr_value = ctx.GetAttr("keep_dims");
  bool keep_dims;
  if (attr_value == nullptr) {
    keep_dims = false;
  } else {
    keep_dims = static_cast<bool>(attr_value->GetBool());
  }
  if (axes_data_num == input_data_dims) {
    if (keep_dims) {
      std::vector<int64_t> dims_new(axes_data_num, 1);
      output_data_shape->SetDimSizes(dims_new);
    } else {
      std::vector<int64_t> dims_new(1, 1);
      output_data_shape->SetDimSizes(dims_new);
    }
    T1 data_sum = static_cast<T1>(0);
    for (int64_t i = 0; i < input_data_num; i++) {
      data_sum += input_data_addr[i];
    }
    output_data_addr[0] = data_sum / input_data_num;
  } else {
    std::vector<int64_t> dims_new(input_data_shape->GetDimSizes());
    if (keep_dims) {
      for (auto iter = axes_data_norepeat.cbegin(); iter != axes_data_norepeat.cend(); iter++) {
        dims_new[*iter] = 1;
      }
    } else {
      for (auto iter = axes_data_norepeat.rbegin(); iter != axes_data_norepeat.rend(); iter++) {
        dims_new.erase(dims_new.begin() + (*iter));
      }
    }
    output_data_shape->SetDimSizes(dims_new);
    // Extract unspecified dimensions
    std::vector<T2> dims_base;
    const int32_t axes_data_num_const = axes_data_num;
    const int32_t dims_base_num = input_data_dims - axes_data_num_const;
    for (T2 i = 0; i < (T2)input_data_dims; i++) {
      bool cflag = true;
      for (int64_t j = 0; j < axes_data_num_const; j++) {
        if (axes_data_norepeat[j] == i) {
          cflag = false;
          break;
        }
      }
      if (cflag) {
        dims_base.push_back(i);
      }
    }
    int64_t addr_stride[axes_data_num_const];
    addr_tmp = 1;
    addr_stride[axes_data_num_const - 1] = addr_tmp;
    for (int32_t i = axes_data_num_const - 2; i > -1; i--) {
      addr_tmp *= input_data_dimsize[axes_data_norepeat[i + 1]];
      addr_stride[i] = addr_tmp;
    }
    int64_t offset_num = addr_tmp * input_data_dimsize[axes_data_norepeat[0]];
    if ((input_data_num > 256 * 1024 && input_data_num / output_data_num > 256) || (output_data_num > 1024)) {
      uint32_t min_core_num = 1;
      int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
      if (max_core_num > output_data_num) {
        max_core_num = output_data_num;
      }
      auto shard_compute = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
          int64_t output_i_addr = 0;
          int64_t seq_tmp = i;
          for (int32_t j = dims_base_num - 1; j > -1; j--) {
            int64_t next = seq_tmp / input_data_dimsize[dims_base[j]];
            int64_t loc = seq_tmp % input_data_dimsize[dims_base[j]];
            seq_tmp = next;
            output_i_addr += loc * dims_addr[dims_base[j]];
            if (seq_tmp == 0) {
              break;
            }
          }
          T1 data_sum = input_data_addr[output_i_addr];
          // In the array, the actual address of the element participating in the calculation.
          int64_t addr_offset = 0;
          for (int64_t j = 1; j < offset_num; j++) {
            int32_t stride = axes_data_num_const - 1;
            for (int32_t k = stride - 1; k > -1; k--) {
              if (j % addr_stride[k] == 0) {
                addr_offset -=
                  (input_data_dimsize[axes_data_norepeat[stride]] - 1) * dims_addr[axes_data_norepeat[stride]];
                stride = k;
                continue;
              }
              break;
            }
            addr_offset += dims_addr[axes_data_norepeat[stride]];
            data_sum += input_data_addr[output_i_addr + addr_offset];
          }
          output_data_addr[i] = data_sum / offset_num;
        }
      };
      KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, output_data_num, output_data_num / max_core_num, shard_compute),
        "ReduceMean Compute failed.");
    } else {
      for (int64_t i = 0; i < output_data_num; i++) {
        // In the array, the actual address of the output.
        int64_t output_i_addr = 0;
        int64_t seq_tmp = i;
        for (int32_t j = dims_base_num - 1; j > -1; j--) {
          int64_t next = seq_tmp / input_data_dimsize[dims_base[j]];
          int64_t loc = seq_tmp % input_data_dimsize[dims_base[j]];
          seq_tmp = next;
          output_i_addr += loc * dims_addr[dims_base[j]];
          if (seq_tmp == 0) {
            break;
          }
        }
        T1 data_sum = input_data_addr[output_i_addr];
        // In the array, the actual address of the element participating in the calculation.
        int64_t addr_offset = 0;
        for (int64_t j = 1; j < offset_num; j++) {
          int32_t stride = axes_data_num_const - 1;
          for (int32_t k = stride - 1; k > -1; k--) {
            if (j % addr_stride[k] == 0) {
              addr_offset -=
                (input_data_dimsize[axes_data_norepeat[stride]] - 1) * dims_addr[axes_data_norepeat[stride]];
              stride = k;
              continue;
            }
            break;
          }
          addr_offset += dims_addr[axes_data_norepeat[stride]];
          data_sum += input_data_addr[output_i_addr + addr_offset];
        }
        output_data_addr[i] = data_sum / offset_num;
      }
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T1, typename T2>
uint32_t ReduceMeanCpuKernel::ReduceMeanCompute_Complex(const CpuKernelContext &ctx) {
  Tensor *input_data = ctx.Input(0);
  auto input_data_addr = reinterpret_cast<T1 *>(input_data->GetData());
  const int64_t input_data_num = input_data->NumElements();
  auto input_data_shape = input_data->GetTensorShape();
  const int32_t input_data_dims = input_data_shape->GetDims();
  std::vector<int64_t> input_data_dimsize = input_data_shape->GetDimSizes();
  std::vector<int64_t> dims_addr(input_data_dims);
  dims_addr[input_data_dims - 1] = 1;
  int64_t addr_tmp = 1;
  for (int32_t i = input_data_dims - 2; i > -1; i--) {
    addr_tmp *= input_data_dimsize[i + 1];
    dims_addr[i] = addr_tmp;
  }
  Tensor *output_data = ctx.Output(0);
  auto output_data_shape = output_data->GetTensorShape();
  auto output_data_addr = reinterpret_cast<T1 *>(output_data->GetData());
  const int64_t output_data_num = output_data->NumElements();
  Tensor *axes_data = ctx.Input(1);
  auto axes_data_addr = reinterpret_cast<T2 *>(axes_data->GetData());
  int64_t axes_data_num = axes_data->NumElements();
  // Check the effectiveness of the value of axes
  for (int64_t i = 0; i < axes_data_num; i++) {
    if ((*(axes_data_addr + i) >= input_data_dims) || (*(axes_data_addr + i) < -input_data_dims)) {
      KERNEL_LOG_ERROR("The value of axes is incorrect.");
      return KERNEL_STATUS_PARAM_INVALID;
    } else if (*(axes_data_addr + i) < 0) {
      *(axes_data_addr + i) += input_data_dims;
    }
  }
  std::sort(axes_data_addr, axes_data_addr + axes_data_num);
  std::vector<T2> axes_data_norepeat;
  for (int64_t i = 0; i < axes_data_num - 1; i++) {
    T2 value = axes_data_addr[i];
    if (value == axes_data_addr[i + 1]) {
      axes_data_num--;
      continue;
    }
    axes_data_norepeat.push_back(value);
  }
  axes_data_norepeat.push_back(axes_data_addr[axes_data_num - 1]);
  // deal with attr
  auto attr_value = ctx.GetAttr("keep_dims");
  bool keep_dims;
  if (attr_value == nullptr) {
    keep_dims = false;
  } else {
    keep_dims = static_cast<bool>(attr_value->GetBool());
  }
  if (axes_data_num == input_data_dims) {
    if (keep_dims) {
      std::vector<int64_t> dims_new(axes_data_num, 1);
      output_data_shape->SetDimSizes(dims_new);
    } else {
      std::vector<int64_t> dims_new(1, 1);
      output_data_shape->SetDimSizes(dims_new);
    }
    T1 data_sum = static_cast<T1>(0);
    for (int64_t i = 0; i < input_data_num; i++) {
      data_sum += input_data_addr[i];
    }
    output_data_addr[kIndex0] = ComplexDiv<T1>(data_sum, input_data_num);
  } else {
    std::vector<int64_t> dims_new(input_data_shape->GetDimSizes());
    if (keep_dims) {
      for (auto iter = axes_data_norepeat.cbegin(); iter != axes_data_norepeat.cend(); iter++) {
        dims_new[*iter] = 1;
      }
    } else {
      for (auto iter = axes_data_norepeat.rbegin(); iter != axes_data_norepeat.rend(); iter++) {
        dims_new.erase(dims_new.begin() + (*iter));
      }
    }
    output_data_shape->SetDimSizes(dims_new);
    // Extract unspecified dimensions
    std::vector<T2> dims_base;
    const int32_t axes_data_num_const = axes_data_num;
    const int32_t dims_base_num = input_data_dims - axes_data_num_const;
    for (T2 i = 0; i < (T2)input_data_dims; i++) {
      bool cflag = true;
      for (int64_t j = 0; j < axes_data_num_const; j++) {
        if (axes_data_norepeat[j] == i) {
          cflag = false;
          break;
        }
      }
      if (cflag) {
        dims_base.push_back(i);
      }
    }
    int64_t addr_stride[axes_data_num_const];
    addr_tmp = 1;
    addr_stride[axes_data_num_const - 1] = addr_tmp;
    for (int32_t i = axes_data_num_const - 2; i > -1; i--) {
      addr_tmp *= input_data_dimsize[axes_data_norepeat[i + 1]];
      addr_stride[i] = addr_tmp;
    }
    int64_t offset_num = addr_tmp * input_data_dimsize[axes_data_norepeat[0]];
    if ((input_data_num > 256 * 1024 && input_data_num / output_data_num > 256) || (output_data_num > 1024)) {
      uint32_t min_core_num = 1;
      int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
      if (max_core_num > output_data_num) {
        max_core_num = output_data_num;
      }
      auto shard_compute = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
          int64_t output_i_addr = 0;
          int64_t seq_tmp = i;
          for (int32_t j = dims_base_num - 1; j > -1; j--) {
            int64_t next = seq_tmp / input_data_dimsize[dims_base[j]];
            int64_t loc = seq_tmp % input_data_dimsize[dims_base[j]];
            seq_tmp = next;
            output_i_addr += loc * dims_addr[dims_base[j]];
            if (seq_tmp == 0) {
              break;
            }
          }
          T1 data_sum = input_data_addr[output_i_addr];
          // In the array, the actual address of the element participating in the calculation.
          int64_t addr_offset = 0;
          for (int64_t j = 1; j < offset_num; j++) {
            int32_t stride = axes_data_num_const - 1;
            for (int32_t k = stride - 1; k > -1; k--) {
              if (j % addr_stride[k] == 0) {
                addr_offset -=
                  (input_data_dimsize[axes_data_norepeat[stride]] - 1) * dims_addr[axes_data_norepeat[stride]];
                stride = k;
                continue;
              }
              break;
            }
            addr_offset += dims_addr[axes_data_norepeat[stride]];
            data_sum += input_data_addr[output_i_addr + addr_offset];
          }
          output_data_addr[i] = ComplexDiv<T1>(data_sum, offset_num);
        }
      };
      KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, output_data_num, output_data_num / max_core_num, shard_compute),
        "ReduceMean Compute failed.");
    } else {
      for (int64_t i = 0; i < output_data_num; i++) {
        // In the array, the actual address of the output.
        int64_t output_i_addr = 0;
        int64_t seq_tmp = i;
        for (int32_t j = dims_base_num - 1; j > -1; j--) {
          int64_t next = seq_tmp / input_data_dimsize[dims_base[j]];
          int64_t loc = seq_tmp % input_data_dimsize[dims_base[j]];
          seq_tmp = next;
          output_i_addr += loc * dims_addr[dims_base[j]];
          if (seq_tmp == 0) {
            break;
          }
        }
        T1 data_sum = input_data_addr[output_i_addr];
        // In the array, the actual address of the element participating in the calculation.
        int64_t addr_offset = 0;
        for (int64_t j = 1; j < offset_num; j++) {
          int32_t stride = axes_data_num_const - 1;
          for (int32_t k = stride - 1; k > -1; k--) {
            if (j % addr_stride[k] == 0) {
              addr_offset -=
                (input_data_dimsize[axes_data_norepeat[stride]] - 1) * dims_addr[axes_data_norepeat[stride]];
              stride = k;
              continue;
            }
            break;
          }
          addr_offset += dims_addr[axes_data_norepeat[stride]];
          data_sum += input_data_addr[output_i_addr + addr_offset];
        }
        output_data_addr[i] = ComplexDiv<T1>(data_sum, offset_num);
      }
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kReduceMean, ReduceMeanCpuKernel);
}  // namespace aicpu
