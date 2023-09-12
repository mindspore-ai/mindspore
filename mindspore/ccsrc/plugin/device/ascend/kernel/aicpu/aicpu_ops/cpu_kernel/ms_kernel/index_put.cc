/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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

#include "cpu_kernel/ms_kernel/index_put.h"
#include <securec.h>
#include <Eigen/Dense>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>

#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "cpu_kernel/common/status.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t IndexPutInputNum = aicpu::kDynamicInput;
const uint32_t IndexPutOutputNum = 1;
const char *kIndexPut = "IndexPut";
const uint32_t INPUT_NUM = 2;
// when input data size is more than kParallelDataNumSameShape, use Parallel
// func
const size_t kParallelDataNumSameShape = 128 * 1024;
const size_t kParallelDataNumMid = 512 * 1024;

#define INDEXPUT_COMPUTE_CASE(DTYPE, TYPE, DTYPE0, CTX)    \
  case (DTYPE): {                                          \
    uint32_t result;                                       \
    if ((DTYPE0) == DT_INT32) {                            \
      result = IndexPutCompute<TYPE, int32_t>(CTX);        \
    } else {                                               \
      result = IndexPutCompute<TYPE, int64_t>(CTX);        \
    }                                                      \
    if (result != KERNEL_STATUS_OK) {                      \
      KERNEL_LOG_ERROR("IndexPut kernel compute failed."); \
      return result;                                       \
    }                                                      \
    break;                                                 \
  }
}  // namespace

namespace aicpu {
uint32_t IndexPutCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, IndexPutInputNum, IndexPutOutputNum),
                      "indexput check input and output number failed");
  KERNEL_HANDLE_ERROR(IndexPutParmCheck(ctx), "indexput check params failed");

  auto data_type = ctx.Input(0)->GetDataType();
  auto data_type_0 = ctx.Input(ctx.GetInputsSize() - 1)->GetDataType();

  switch (data_type) {
    INDEXPUT_COMPUTE_CASE(DT_FLOAT16, Eigen::half, data_type_0, ctx)
    INDEXPUT_COMPUTE_CASE(DT_FLOAT, float, data_type_0, ctx)
    INDEXPUT_COMPUTE_CASE(DT_DOUBLE, double, data_type_0, ctx)
    INDEXPUT_COMPUTE_CASE(DT_INT32, int32_t, data_type_0, ctx)
    INDEXPUT_COMPUTE_CASE(DT_UINT8, uint8_t, data_type_0, ctx)
    INDEXPUT_COMPUTE_CASE(DT_INT16, int16_t, data_type_0, ctx)
    INDEXPUT_COMPUTE_CASE(DT_INT8, int8_t, data_type_0, ctx)
    INDEXPUT_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, data_type_0, ctx)
    INDEXPUT_COMPUTE_CASE(DT_INT64, int64_t, data_type_0, ctx)
    INDEXPUT_COMPUTE_CASE(DT_UINT16, uint16_t, data_type_0, ctx)
    INDEXPUT_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, data_type_0, ctx)
    INDEXPUT_COMPUTE_CASE(DT_UINT32, uint32_t, data_type_0, ctx)
    INDEXPUT_COMPUTE_CASE(DT_UINT64, uint64_t, data_type_0, ctx)
    default:
      KERNEL_LOG_ERROR("indexput kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t IndexPutCpuKernel::IndexPutParmCheck(const CpuKernelContext &ctx) {
  Tensor *input_0 = ctx.Input(0);
  Tensor *input_1 = ctx.Input(1);
  Tensor *indices_data = ctx.Input(2);
  Tensor *output = ctx.Output(0);
  AttrValue *accumulate_attr_ptr = ctx.GetAttr("accumulate");
  auto tensorshapes = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_NULLPTR(input_0->GetData(), KERNEL_STATUS_PARAM_INVALID, "get input 0 data failed.")
  KERNEL_CHECK_NULLPTR(input_1->GetData(), KERNEL_STATUS_PARAM_INVALID, "get input 1 data failed.")
  KERNEL_CHECK_NULLPTR(indices_data->GetData(), KERNEL_STATUS_PARAM_INVALID, "get indices data failed.")
  KERNEL_CHECK_NULLPTR(output->GetData(), KERNEL_STATUS_PARAM_INVALID, "get output  data failed.")
  KERNEL_CHECK_NULLPTR(accumulate_attr_ptr, KERNEL_STATUS_PARAM_INVALID, "get accumulate  data failed.")

  DataType input0_type = input_0->GetDataType();
  DataType input1_type = input_1->GetDataType();
  DataType indices_type = indices_data->GetDataType();
  KERNEL_CHECK_FALSE((input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID,
                     "the data type of input0[%s] need be same with"
                     "input1[%s].",
                     DTypeStr(input0_type).c_str(), DTypeStr(input1_type).c_str())
  KERNEL_CHECK_FALSE((indices_type == DT_INT32 || indices_type == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                     "the data type of indices[%s] need DT_INT32 or DT_INT64", DTypeStr(indices_type).c_str())
  if (accumulate_attr_ptr) {
    int64_t accumulate_data = accumulate_attr_ptr->GetInt();
    KERNEL_CHECK_FALSE((accumulate_data == 0 || accumulate_data == 1), KERNEL_STATUS_PARAM_INVALID,
                       "accumulate must be 1 or 0.");
  }
  KERNEL_CHECK_FALSE(ctx.GetInputsSize() - INPUT_NUM <= tensorshapes.size(), KERNEL_STATUS_PARAM_INVALID,
                     "too many indices for tensor of dimension [%d] (got [%d])", tensorshapes.size(),
                     ctx.GetInputsSize() - INPUT_NUM);
  int64_t maxnum = indices_data->NumElements();
  for (size_t i = 2; i < ctx.GetInputsSize(); ++i) {
    if (ctx.Input(i)->NumElements() > maxnum) {
      maxnum = ctx.Input(i)->NumElements();
    }
  }
  KERNEL_CHECK_FALSE((input_1->NumElements() == 1 || input_1->NumElements() == maxnum ||
                      input_1->NumElements() == tensorshapes[tensorshapes.size() - 1]),
                     KERNEL_STATUS_PARAM_INVALID, "shape mismatch");

  KERNEL_LOG_DEBUG(
    "indexputcpukernel[%s],input0:size[%llu],"
    "input1:size[%llu],output:size[%llu]",
    ctx.GetOpType().c_str(), input_0->GetDataSize(), input_1->GetDataSize(), output->GetDataSize());
  return KERNEL_STATUS_OK;
}

void IndexPutCpuKernel::Transpose(std::vector<std::vector<int64_t>> *A) const {
  uint32_t old_width = (*A)[0].size();
  uint32_t old_height = (*A).size();
  std::vector<std::vector<int64_t>> temp(old_width, std::vector<int64_t>(old_height));
  for (int64_t i = 0; (size_t)i < old_width; ++i) {
    for (int64_t j = 0; (size_t)j < old_height; ++j) {
      temp[i][j] = (*A)[j][i];
    }
  }
  *A = temp;
}

int64_t IndexPutCpuKernel::Multiplicative(const std::vector<int64_t> &tensorshapes, int64_t start, int64_t end) {
  int64_t result = 1;
  for (int64_t i = start; i < end; i++) {
    result *= tensorshapes[i];
  }
  return result;
}

template <typename T>
bool IndexPutCpuKernel::ComputeNospecial(std::vector<int64_t> x1_shape, T *x2, size_t x2_nums,
                                         std::vector<std::vector<int64_t>> indices_value, T *y, int accumulate) {
  size_t x1_shape_size = x1_shape.size();
  size_t idxli = indices_value.size();
  size_t idxcol = indices_value[0].size();
  if (x2_nums == 0) {
    KERNEL_LOG_ERROR("invalid x2 input, please check!");
    return false;
  }
  for (size_t i = 0; i < idxli; ++i) {
    size_t offset = 0;
    for (size_t j = 0; j < idxcol; ++j) {
      offset += indices_value[i][j] * Multiplicative(x1_shape, j + 1, x1_shape_size);
    }
    size_t v_idx = i % x2_nums;
    y[offset] = (accumulate == 0) ? x2[v_idx] : y[offset] + x2[v_idx];
  }
  return true;
}

template <typename T>
bool IndexPutCpuKernel::ComputeSpecial(std::vector<int64_t> x1_shape, T *x2, size_t x2_nums,
                                       std::vector<std::vector<int64_t>> indices_value, T *y, int accumulate) {
  size_t x1_shape_size = x1_shape.size();
  size_t idxli = indices_value.size();
  size_t idxcol = indices_value[0].size();
  size_t strides = Multiplicative(x1_shape, indices_value.size(), x1_shape_size);
  if (x2_nums == 0) {
    KERNEL_LOG_ERROR("invalid x2 input, please check!");
    return false;
  }
  for (size_t i = 0; i < idxcol; i++) {
    size_t offset = 0;
    for (size_t j = 0; j < idxli; j++) {
      offset += indices_value[j][i] * Multiplicative(x1_shape, j + 1, x1_shape_size);
    }
    for (size_t j = 0; j < strides; j++) {
      y[offset + j] = (accumulate == 0) ? x2[j % x2_nums] : y[offset + j] + x2[j % x2_nums];
    }
  }
  return true;
}

template <typename T, typename T0>
uint32_t IndexPutCpuKernel::IndexPutCompute(const CpuKernelContext &ctx) {
  AttrValue *accumulata_data = ctx.GetAttr("accumulate");
  uint64_t accumulata_value = accumulata_data->GetInt();
  auto *x1 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto *x2 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto *y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto tensorshapes = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  auto shapes_size = tensorshapes.size();
  size_t data_num = static_cast<size_t>(ctx.Input(0)->NumElements());
  size_t nums = static_cast<size_t>(ctx.Input(1)->NumElements());

  std::vector<std::vector<int64_t>> indices_value(ctx.GetInputsSize() - INPUT_NUM);
  for (size_t i = 2; i < ctx.GetInputsSize(); i++) {
    auto *linetensor = reinterpret_cast<T0 *>(ctx.Input(i)->GetData());
    std::vector<int64_t> iline(ctx.Input(i)->NumElements());
    for (size_t j = 0; static_cast<int64_t>(j) < ctx.Input(i)->NumElements(); j++) {
      linetensor[j] = (linetensor[j] < 0) ? linetensor[j] + tensorshapes[i - INPUT_NUM] : linetensor[j];
      if (linetensor[j] < 0 || linetensor[j] >= tensorshapes[i - INPUT_NUM]) {
        KERNEL_LOG_ERROR("invalid indices input[%d]", i - INPUT_NUM);
        return KERNEL_STATUS_PARAM_INVALID;
      }
      iline[j] = linetensor[j];
    }
    indices_value[i - INPUT_NUM] = iline;
  }
  size_t maxl = 0;
  for (size_t i = 0; i < indices_value.size(); i++) {
    maxl = std::max(indices_value[i].size(), maxl);
  }
  for (size_t i = 0; i < indices_value.size(); i++) {
    while (indices_value[i].size() != maxl) {
      indices_value[i].push_back(indices_value[i][0]);
    }
  }
  if (data_num >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t temp_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    uint32_t max_core_num = (temp_core_num >= data_num)
                              ? data_num
                              : (data_num <= kParallelDataNumMid) ? std::min(temp_core_num, 4U) : temp_core_num;
    auto sharder_index_put = [&](int64_t start, int64_t end) {
      size_t length = (end - start) * sizeof(T);
      (void)memcpy_s(y + start, length, x1 + start, length);
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_index_put),
                        "IndexPut Compute failed.");
  } else {
    (void)memcpy_s(y, data_num * sizeof(T), x1, data_num * sizeof(T));
  }
  bool flag = true;
  if (indices_value.size() == shapes_size) {
    (void)Transpose(&indices_value);
    flag = ComputeNospecial<T>(tensorshapes, x2, nums, indices_value, y, accumulata_value);
  } else {
    flag = ComputeSpecial<T>(tensorshapes, x2, nums, indices_value, y, accumulata_value);
  }
  if (flag == false) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kIndexPut, IndexPutCpuKernel);
}  // namespace aicpu
