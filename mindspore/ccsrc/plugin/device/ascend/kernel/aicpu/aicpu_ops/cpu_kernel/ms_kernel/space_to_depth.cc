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
#ifndef AICPU_KERNELS_SPACETODEPTH_CC_
#define AICPU_KERNELS_SPACETODEPTH_CC_

#include "space_to_depth.h"

#include "Eigen/Core"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include <iostream>
#include <thread>
#include <unordered_map>
#include <mutex>

namespace {
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 1;
const char *kSpaceToDepth = "SpaceToDepth";

#define SPACETODEPTH_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                              \
    uint32_t result = DoCompute<TYPE>(CTX);                    \
    if (result != KERNEL_STATUS_OK) {                          \
      KERNEL_LOG_ERROR("SpaceToDepth kernel compute failed."); \
      return result;                                           \
    }                                                          \
    break;                                                     \
  }
}  // namespace

namespace aicpu {
template <typename T>
uint32_t SpaceToDepthCpuKernel::DoCompute(CpuKernelContext &ctx) {
  std::cout << "in DoCompute." << std::endl;
  auto input_shape = ctx.Input(0)->GetTensorShape();
  auto output_shape = ctx.Output(0)->GetTensorShape();
  auto input_dims = input_shape->GetDimSizes();
  std::vector<std::string> attr_name1 = {"data_format"};
  AttrValue *attr_data_format = ctx.GetAttr("data_format");
  std::vector<std::string> attr_name2 = {"block_size"};
  AttrValue *attr_block_size = ctx.GetAttr("block_size");
  data_format_ = (attr_data_format == nullptr) ? "NHWC" : (attr_data_format->GetString());
  int64_t block_size = (attr_block_size == nullptr) ? 2 : (attr_block_size->GetInt());
  const int64_t zero = 0;
  const int64_t n_nhwc = 0;
  const int64_t h_nhwc = 1;
  const int64_t w_nhwc = 2;
  const int64_t c_nhwc = 3;
  const int64_t n_nchw = 0;
  const int64_t c_nchw = 1;
  const int64_t h_nchw = 2;
  const int64_t w_nchw = 3;
  if (block_size == zero && block_size * block_size == zero) {
    return KERNEL_STATUS_PARAM_INVALID;
  }

  std::vector<int64_t> output_dims;
  if (data_format_ == "NHWC") {
    KERNEL_CHECK_FALSE((input_dims[h_nhwc] % block_size == zero || input_dims[w_nhwc] % block_size == zero),
                       KERNEL_STATUS_PARAM_INVALID, "Height and Weight must can be divided by block_size.");
    output_dims = {input_dims[n_nhwc], input_dims[h_nhwc] / block_size, input_dims[w_nhwc] / block_size,
                   input_dims[c_nhwc] * block_size * block_size};
    output_shape->SetDimSizes(output_dims);
    input_dims = {input_dims[n_nhwc], input_dims[c_nhwc], input_dims[h_nhwc], input_dims[w_nhwc]};
    output_dims = {output_dims[n_nhwc], output_dims[c_nhwc], output_dims[h_nhwc], output_dims[w_nhwc]};
  } else if (data_format_ == "NCHW") {
    KERNEL_CHECK_FALSE((input_dims[h_nchw] % block_size == 0 || input_dims[w_nchw] % block_size == 0),
                       KERNEL_STATUS_PARAM_INVALID, "Height and Weight must can be divided by block_size.");
    output_dims = {input_dims[n_nchw], input_dims[c_nchw] * block_size * block_size, input_dims[h_nchw] / block_size,
                   input_dims[w_nchw] / block_size};
    output_shape->SetDimSizes(output_dims);
  }

  auto input = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t x = 0;
  const size_t data_num = (size_t)ctx.Input(0)->NumElements();

  for (size_t i = 0; i < data_num; i = i + block_size) {
    for (size_t j = i; j < block_size + i; ++j) {
      if (j % (output_dims[h_nhwc] * output_dims[c_nhwc]) == 0) {
        x = -1;
      }
      if (j % input_dims[h_nhwc] == 0) {
        ++x;
      }
      size_t number = 0, output_pos = 0;
      size_t loc = j / input_dims[h_nhwc];
      number += (loc / (input_dims[w_nhwc] * input_dims[c_nhwc])) * input_dims[w_nhwc] * input_dims[c_nhwc];
      // Mark the position of this segment of the vector in the entire segment.
      number += (output_dims[h_nhwc] * output_dims[c_nhwc] / input_dims[h_nhwc]) *
                (loc / (output_dims[h_nhwc] * output_dims[c_nhwc] / input_dims[h_nhwc]));
      // Label the position of the block within a segment of the vector.
      number += ((loc % input_dims[c_nhwc]) / block_size) * block_size * block_size;
      // Mark the relative position within the small block.
      number += loc % block_size + block_size * (x / input_dims[c_nhwc]);
      output_pos = j % input_dims[h_nhwc] + number * input_dims[h_nhwc];
      output[output_pos] = input[j];
    }
  }

  return KERNEL_STATUS_OK;
}  // DoCompute

uint32_t SpaceToDepthCpuKernel::STDParamCheck(CpuKernelContext &ctx) {
  // check params
  auto input = ctx.Input(0);
  auto output = ctx.Output(0);

  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "SpaceToDepth check input and output number failed.");

  KERNEL_LOG_DEBUG(
    "SpaceToDepthCpuKernel[%s], input0: size[%llu];"
    "output: size[%llu].",
    ctx.GetOpType().c_str(), input->GetDataSize(), output->GetDataSize());

  // check data_format

  std::vector<std::string> attr_name1 = {"data_format"};
  AttrValue *attr_data_format = ctx.GetAttr("data_format");
  data_format_ = (attr_data_format == nullptr) ? "NHWC" : (attr_data_format->GetString());

  KERNEL_CHECK_FALSE((data_format_ == "NHWC" || data_format_ == "NCHW"), KERNEL_STATUS_PARAM_INVALID,
                     "The data_format must be NCHW, NHWC or NCHW_VECT_C, but got: [%s]", data_format_);

  // check block_size
  std::vector<std::string> attr_name2 = {"block_size"};
  const int64_t min_block_size = 2;
  int64_t block_size = ctx.GetAttr("block_size")->GetInt();
  KERNEL_CHECK_FALSE((block_size >= min_block_size), KERNEL_STATUS_PARAM_INVALID,
                     "The value of block_size must be greater than 2");
  return KERNEL_STATUS_OK;
}

uint32_t SpaceToDepthCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(STDParamCheck(ctx), "SpaceToDepth check params failed.");
  Tensor *input0_tensor = ctx.Input(0);
  auto input_data_type = input0_tensor->GetDataType();

  switch (input_data_type) {
    SPACETODEPTH_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    SPACETODEPTH_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    SPACETODEPTH_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    SPACETODEPTH_COMPUTE_CASE(DT_FLOAT, float, ctx)
    SPACETODEPTH_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    SPACETODEPTH_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    SPACETODEPTH_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    SPACETODEPTH_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    SPACETODEPTH_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    SPACETODEPTH_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    SPACETODEPTH_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    SPACETODEPTH_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    SPACETODEPTH_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    SPACETODEPTH_COMPUTE_CASE(DT_QINT8, int8_t, ctx)
    SPACETODEPTH_COMPUTE_CASE(DT_QINT16, int16_t, ctx)
    SPACETODEPTH_COMPUTE_CASE(DT_QINT32, int32_t, ctx)
    default:
      KERNEL_LOG_ERROR("SpaceToDepth kernel data type[%s] not support.", DTypeStr(input_data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kSpaceToDepth, SpaceToDepthCpuKernel);
}  // namespace aicpu
#endif  // AICPU_KERNELS_SPACETODEPTH_CC_
