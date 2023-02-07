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
#include "max_unpool_2d.h"

#include <cmath>
#include <iostream>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
constexpr int64_t kParallelDataNums = 1024;
const char *kMaxUnpool2D = "MaxUnpool2D";

#define SWITCH_PARALLEL(SHARD, end_num, ctx)                                                                        \
  if (end_num <= kParallelDataNums) {                                                                               \
    for (size_t i = 0; i < size_t(end_num); i++) {                                                                  \
      SHARD(i, i + 1);                                                                                              \
    }                                                                                                               \
  } else {                                                                                                          \
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, end_num, 1, SHARD), "MaxUnpool2D #SHARD Compute failed."); \
  }

}  // namespace

namespace aicpu {
template <typename DATA_T>
uint32_t MaxUnpool2DCpuKernel::MaxUnpool2D_COMPUTE_CASE(CpuKernelContext &ctx, DataType indices_type) {
  // Compute by indices_type
  switch (indices_type) {
    case DT_INT32:
      return MaxUnpool2DCompute<DATA_T, int32_t>(ctx);
    case DT_INT64:
      return MaxUnpool2DCompute<DATA_T, int64_t>(ctx);
    default:
      KERNEL_LOG_ERROR("indices_type [%s] must be in [{DT_INT32, DT_INT64}].", DTypeStr(indices_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

uint32_t MaxUnpool2DCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "MaxUnpool2D check input and output number failed.");
  KERNEL_HANDLE_ERROR(MaxUnpool2DCheck(ctx), "MaxUnpool2D check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  auto indices_type = ctx.Input(1)->GetDataType();
  switch (data_type) {
    case DT_INT8:
      return MaxUnpool2D_COMPUTE_CASE<int8_t>(ctx, indices_type);
    case DT_INT16:
      return MaxUnpool2D_COMPUTE_CASE<int16_t>(ctx, indices_type);
    case DT_INT32:
      return MaxUnpool2D_COMPUTE_CASE<int32_t>(ctx, indices_type);
    case DT_INT64:
      return MaxUnpool2D_COMPUTE_CASE<int64_t>(ctx, indices_type);
    case DT_UINT8:
      return MaxUnpool2D_COMPUTE_CASE<uint8_t>(ctx, indices_type);
    case DT_UINT16:
      return MaxUnpool2D_COMPUTE_CASE<uint16_t>(ctx, indices_type);
    case DT_UINT32:
      return MaxUnpool2D_COMPUTE_CASE<uint32_t>(ctx, indices_type);
    case DT_UINT64:
      return MaxUnpool2D_COMPUTE_CASE<uint64_t>(ctx, indices_type);
    case DT_FLOAT16:
      return MaxUnpool2D_COMPUTE_CASE<Eigen::half>(ctx, indices_type);
    case DT_FLOAT:
      return MaxUnpool2D_COMPUTE_CASE<float>(ctx, indices_type);
    case DT_DOUBLE:
      return MaxUnpool2D_COMPUTE_CASE<double>(ctx, indices_type);
    default:
      KERNEL_LOG_ERROR("MaxUnpool2D kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t MaxUnpool2DCpuKernel::MaxUnpool2DCheck(CpuKernelContext &ctx) {
  DataType input0Type = ctx.Input(0)->GetDataType();
  DataType outputType = ctx.Output(0)->GetDataType();
  KERNEL_CHECK_FALSE((input0Type == outputType), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of output [%d] need be same with "
                     "input0 [%d].",
                     outputType, input0Type)

  KERNEL_LOG_INFO(
    "MaxUnpool2DCpuKernel[%s], input0: size[%llu];"
    "input1: size[%llu], output: size[%llu].",
    ctx.GetOpType().c_str(), ctx.Input(0)->GetDataSize(), ctx.Input(1)->GetDataSize(), ctx.Output(0)->GetDataSize());

  return KERNEL_STATUS_OK;
}

template <typename DATA_T, typename INDICES_T>
uint32_t MaxUnpool2DCpuKernel::MaxUnpool2DCompute(CpuKernelContext &ctx) {
  Tensor *input = ctx.Input(0);
  Tensor *indices = ctx.Input(1);
  Tensor *output = ctx.Output(0);
  std::string dataFormat = "NCHW";
  if (ctx.GetAttr("data_format") != nullptr) {
    dataFormat = ctx.GetAttr("data_format")->GetString();
  }
  int32_t NIndex, CIndex, HIndex, WIndex;
  bool error = false;
  if (dataFormat == "NHWC") {
    NIndex = 0;
    CIndex = 3;
    HIndex = 1;
    WIndex = 2;
    auto inputShape = input->GetTensorShape();
    int64_t numBatch = inputShape->GetDimSize(NIndex);
    int64_t inputHeight = inputShape->GetDimSize(HIndex);
    int64_t inputWidth = inputShape->GetDimSize(WIndex);
    int64_t numChannels = inputShape->GetDimSize(CIndex);
    auto output_shape = output->GetTensorShape();
    int64_t oheight = output_shape->GetDimSize(HIndex);
    int64_t owidth = output_shape->GetDimSize(WIndex);

    auto *rawInput = reinterpret_cast<DATA_T *>(input->GetData());
    auto *rawIndices = reinterpret_cast<INDICES_T *>(indices->GetData());
    auto *rawOutput = reinterpret_cast<DATA_T *>(output->GetData());
    for (int s = 0; s < numBatch * oheight * owidth * numChannels; s++) {
      rawOutput[s] = (DATA_T)0;
    }

    auto shard = [&](int64_t start, int64_t end) {
      for (int64_t n = start; n < end; n++) {
        int64_t nOutputOffset = n * numChannels * owidth * oheight;
        int64_t nInputOffset = n * numChannels * inputWidth * inputHeight;
        DATA_T *output_p_k = rawOutput + nOutputOffset;
        DATA_T *input_p_k = rawInput + nInputOffset;
        INDICES_T *ind_p_k = rawIndices + nInputOffset;

        int64_t maxp;
        for (int64_t k = 0; k < numChannels; k++) {
          for (int64_t i = 0; i < inputHeight; i++) {
            for (int64_t j = 0; j < inputWidth; j++) {
              maxp = ind_p_k[i * inputWidth * numChannels + j * numChannels + k];
              if (maxp < 0 || maxp >= owidth * oheight) {
                error = true;
                KERNEL_LOG_ERROR(
                  "MaxUnpool2D:  output_size H_out * W_out "
                  "should be bigger than argmax, now H_out is [%ld], "
                  "and W_out is [%ld], but one of the values in argmax is "
                  "[%ld].",
                  oheight, owidth, maxp);
              } else {
                output_p_k[maxp * numChannels + k] = input_p_k[i * inputWidth * numChannels + j * numChannels + k];
              }
            }
          }
        }
      }
    };
    SWITCH_PARALLEL(shard, numBatch, ctx);
  } else {
    NIndex = 0;
    CIndex = 1;
    HIndex = 2;
    WIndex = 3;
    auto inputShape = input->GetTensorShape();
    int64_t numBatch = inputShape->GetDimSize(NIndex);
    int64_t inputHeight = inputShape->GetDimSize(HIndex);
    int64_t inputWidth = inputShape->GetDimSize(WIndex);
    int64_t numChannels = inputShape->GetDimSize(CIndex);

    auto output_shape = output->GetTensorShape();
    int64_t oheight = output_shape->GetDimSize(HIndex);
    int64_t owidth = output_shape->GetDimSize(WIndex);
    auto *rawInput = reinterpret_cast<DATA_T *>(input->GetData());
    auto *rawIndices = reinterpret_cast<INDICES_T *>(indices->GetData());
    auto *rawOutput = reinterpret_cast<DATA_T *>(output->GetData());

    for (int s = 0; s < numBatch * oheight * owidth * numChannels; s++) {
      rawOutput[s] = (DATA_T)0;
    }
    auto shard = [&](int64_t start, int64_t end) {
      for (int64_t n = start; n < end; n++) {
        int64_t nOutputOffset = n * numChannels * owidth * oheight;
        int64_t nInputOffset = n * numChannels * inputWidth * inputHeight;
        int64_t k = 0;
        for (k = 0; k < numChannels; k++) {
          int64_t finalOutputOffset = nOutputOffset + k * owidth * oheight;
          int64_t finalInputOffset = nInputOffset + k * inputWidth * inputHeight;
          DATA_T *output_p_k = rawOutput + finalOutputOffset;
          DATA_T *input_p_k = rawInput + finalInputOffset;
          INDICES_T *ind_p_k = rawIndices + finalInputOffset;

          int64_t maxp;
          for (int64_t i = 0; i < inputHeight; i++) {
            for (int64_t j = 0; j < inputWidth; j++) {
              maxp = ind_p_k[i * inputWidth + j];
              if (maxp < 0 || maxp >= owidth * oheight) {
                error = true;
                KERNEL_LOG_ERROR(
                  "MaxUnpool2D:  output_size H_out * W_out "
                  "should be bigger than argmax, now H_out is [%ld], "
                  "and W_out is [%ld], but one of the values in argmax is "
                  "[%ld].",
                  oheight, owidth, maxp);
              } else {
                output_p_k[maxp] = input_p_k[i * inputWidth + j];
              }
            }
          }
        }
      }
    };
    SWITCH_PARALLEL(shard, numBatch, ctx);
  }

  if (error == true) {
    return KERNEL_STATUS_PARAM_INVALID;
  } else {
    return KERNEL_STATUS_OK;
  }
}

REGISTER_CPU_KERNEL(kMaxUnpool2D, MaxUnpool2DCpuKernel);
}  // namespace aicpu