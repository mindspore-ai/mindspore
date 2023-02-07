
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

#include "max_pool_3d_grad_with_argmax.h"
#include <iostream>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 3;
const char *kMaxPool3DGradWithArgmax = "MaxPool3DGradWithArgmax";

#define MAX_POOL3D_GRAD_WITH_ARGMAX_COMPUTE_CASE(DTYPE, INTYPE, ARGTYPE, CTX) \
  case (DTYPE): {                                                             \
    uint32_t result = MaxPool3DGradWithArgmaxCompute<INTYPE, ARGTYPE>(CTX);   \
    if (result != KERNEL_STATUS_OK) {                                         \
      KERNEL_LOG_ERROR("MaxPool3DGradWithArgmax kernel compute failed.");     \
      return result;                                                          \
    }                                                                         \
    break;                                                                    \
  }
}  // namespace

namespace aicpu {
uint32_t MaxPool3DGradWithArgmaxCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  std::vector<std::string> attr_names = {"ksize", "strides", "pads", "dilation"};
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum, attr_names),
                      "MaxPool3DGradWithArgmax check input and output number failed.");
  KERNEL_HANDLE_ERROR(MaxPool3DGradWithArgmaxParamCheck(ctx), "MaxPool3DGradWithArgmax check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  auto argmax_type = ctx.Input(2)->GetDataType();
  if (argmax_type == DT_INT32) {
    switch (data_type) {
      MAX_POOL3D_GRAD_WITH_ARGMAX_COMPUTE_CASE(DT_INT8, int8_t, int32_t, ctx)
      MAX_POOL3D_GRAD_WITH_ARGMAX_COMPUTE_CASE(DT_INT16, int16_t, int32_t, ctx)
      MAX_POOL3D_GRAD_WITH_ARGMAX_COMPUTE_CASE(DT_INT32, int32_t, int32_t, ctx)
      MAX_POOL3D_GRAD_WITH_ARGMAX_COMPUTE_CASE(DT_INT64, int64_t, int32_t, ctx)
      MAX_POOL3D_GRAD_WITH_ARGMAX_COMPUTE_CASE(DT_UINT8, uint8_t, int32_t, ctx)
      MAX_POOL3D_GRAD_WITH_ARGMAX_COMPUTE_CASE(DT_UINT16, uint16_t, int32_t, ctx)
      MAX_POOL3D_GRAD_WITH_ARGMAX_COMPUTE_CASE(DT_UINT32, uint32_t, int32_t, ctx)
      MAX_POOL3D_GRAD_WITH_ARGMAX_COMPUTE_CASE(DT_UINT64, uint64_t, int32_t, ctx)
      MAX_POOL3D_GRAD_WITH_ARGMAX_COMPUTE_CASE(DT_FLOAT16, Eigen::half, int32_t, ctx)
      MAX_POOL3D_GRAD_WITH_ARGMAX_COMPUTE_CASE(DT_FLOAT, float, int32_t, ctx)
      MAX_POOL3D_GRAD_WITH_ARGMAX_COMPUTE_CASE(DT_DOUBLE, double, int32_t, ctx)
      default:
        KERNEL_LOG_ERROR("MaxPool3DWithArgmax kernel input data type [%s] not support.", DTypeStr(data_type).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
  } else if (argmax_type == DT_INT64) {
    switch (data_type) {
      MAX_POOL3D_GRAD_WITH_ARGMAX_COMPUTE_CASE(DT_INT8, int8_t, int64_t, ctx)
      MAX_POOL3D_GRAD_WITH_ARGMAX_COMPUTE_CASE(DT_INT16, int16_t, int64_t, ctx)
      MAX_POOL3D_GRAD_WITH_ARGMAX_COMPUTE_CASE(DT_INT32, int32_t, int64_t, ctx)
      MAX_POOL3D_GRAD_WITH_ARGMAX_COMPUTE_CASE(DT_INT64, int64_t, int64_t, ctx)
      MAX_POOL3D_GRAD_WITH_ARGMAX_COMPUTE_CASE(DT_UINT8, uint8_t, int64_t, ctx)
      MAX_POOL3D_GRAD_WITH_ARGMAX_COMPUTE_CASE(DT_UINT16, uint16_t, int64_t, ctx)
      MAX_POOL3D_GRAD_WITH_ARGMAX_COMPUTE_CASE(DT_UINT32, uint32_t, int64_t, ctx)
      MAX_POOL3D_GRAD_WITH_ARGMAX_COMPUTE_CASE(DT_UINT64, uint64_t, int64_t, ctx)
      MAX_POOL3D_GRAD_WITH_ARGMAX_COMPUTE_CASE(DT_FLOAT16, Eigen::half, int64_t, ctx)
      MAX_POOL3D_GRAD_WITH_ARGMAX_COMPUTE_CASE(DT_FLOAT, float, int64_t, ctx)
      MAX_POOL3D_GRAD_WITH_ARGMAX_COMPUTE_CASE(DT_DOUBLE, double, int64_t, ctx)
      default:
        KERNEL_LOG_ERROR("MaxPool3DGradWithArgmax kernel input data type [%s] not support.",
                         DTypeStr(data_type).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
  } else {
    KERNEL_LOG_ERROR(
      "MaxPool3DGradWithArgmax kernel input_argmax data type [%s] not "
      "support.",
      DTypeStr(argmax_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t MaxPool3DGradWithArgmaxCpuKernel::MaxPool3DGradWithArgmaxParamCheck(CpuKernelContext &ctx) {
  auto input_x_info = ctx.Input(0);
  auto input_grads_info = ctx.Input(1);
  auto input_argmax_info = ctx.Input(2);
  auto output_y_info = ctx.Output(0);
  DataType input_x_type = input_x_info->GetDataType();
  DataType input_grads_type = input_grads_info->GetDataType();
  DataType out_type = output_y_info->GetDataType();
  KERNEL_CHECK_FALSE((input_x_type == input_grads_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input x [%s] need be same with "
                     "input grads [%s].",
                     DTypeStr(input_x_type).c_str(), DTypeStr(input_grads_type).c_str())
  KERNEL_CHECK_FALSE((input_x_type == out_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input x [%s] need be same with "
                     "output [%s].",
                     DTypeStr(input_x_type).c_str(), DTypeStr(out_type).c_str())
  DataType input_argmax_type = input_argmax_info->GetDataType();
  KERNEL_CHECK_FALSE((input_argmax_type == DT_INT32) || (input_argmax_type == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of output argmax:[%s] should be a int32 or int64. ",
                     DTypeStr(input_argmax_type).c_str())

  std::vector<int64_t> dim_vec = input_x_info->GetTensorShape()->GetDimSizes();
  int64_t dimsize = dim_vec.size();
  KERNEL_CHECK_FALSE(dimsize == 5, KERNEL_STATUS_PARAM_INVALID, "The dim of input:[%d] should be 5.", dimsize)

  const size_t DIM_SIZE1 = 1;
  const size_t DIM_SIZE3 = 3;
  const size_t DIM_SIZE5 = 5;
  AttrValue *attr_ksize = ctx.GetAttr("ksize");
  std::vector<int64_t> ksizeList = attr_ksize->GetListInt();
  KERNEL_CHECK_FALSE(ksizeList.size() == DIM_SIZE1 || ksizeList.size() == DIM_SIZE3, KERNEL_STATUS_PARAM_INVALID,
                     "The size of ksize:[%d] should be 1 or 3.", ksizeList.size())
  AttrValue *attr_strides = ctx.GetAttr("strides");
  std::vector<int64_t> stridesList = attr_strides->GetListInt();
  KERNEL_CHECK_FALSE(stridesList.size() == DIM_SIZE1 || stridesList.size() == DIM_SIZE3, KERNEL_STATUS_PARAM_INVALID,
                     "The size of strides:[%d] should be 1 or 3.", stridesList.size())
  AttrValue *attr_pads = ctx.GetAttr("pads");
  std::vector<int64_t> padsList = attr_pads->GetListInt();
  KERNEL_CHECK_FALSE(padsList.size() == DIM_SIZE1 || padsList.size() == DIM_SIZE3, KERNEL_STATUS_PARAM_INVALID,
                     "The size of pads:[%d] should be 1 or 3.", padsList.size())
  AttrValue *attr_dilation = ctx.GetAttr("dilation");
  std::vector<int64_t> dilationList = attr_dilation->GetListInt();
  KERNEL_CHECK_FALSE(
    dilationList.size() == DIM_SIZE1 || dilationList.size() == DIM_SIZE3 || dilationList.size() == DIM_SIZE5,
    KERNEL_STATUS_PARAM_INVALID, "The size of dilation:[%d] should be 1, 3 or 5.", dilationList.size())
  KERNEL_LOG_DEBUG(
    "MaxPool3DGradWithArgmaxCpuKernel[%s], input x: size[%llu];"
    "input grads: size[%llu], input argmax: size[%llu], output y: "
    "size[%lld].",
    ctx.GetOpType().c_str(), input_x_info->GetDataSize(), input_grads_info->GetDataSize(),
    input_argmax_info->GetDataSize(), output_y_info->GetDataSize());

  return KERNEL_STATUS_OK;
}

template <typename T, typename S>
void MaxPool3DGradWithArgmaxCpuKernel::MaxPool3DGradWithArgmaxSingleCompute(
  T *input_grad, S *input_argmax, T *output_y, int64_t iD, int64_t iH, int64_t iW, int64_t oD, int64_t oH, int64_t oW,
  int64_t kD, int64_t kH, int64_t kW, int64_t sD, int64_t sH, int64_t sW, int64_t pD, int64_t pH, int64_t pW,
  int64_t dD, int64_t dH, int64_t dW) {
  T *in_grad = input_grad;
  T *out_y = output_y;
  S *argmax = input_argmax;

  /* calculate max points */
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t ti, i, j;
  for (ti = 0; ti < oD; ti++) {
    for (i = 0; i < oH; i++) {
      for (j = 0; j < oW; j++) {
        /* retrieve position of max */
        int64_t index = ti * oH * oW + i * oW + j;
        int64_t maxp = argmax[index];

        if (maxp != -1) {
          /* update gradient */
          out_y[maxp] += in_grad[index];
        }
      }
    }
  }
}

template <typename T, typename S>
uint32_t MaxPool3DGradWithArgmaxCpuKernel::MaxPool3DGradWithArgmaxCompute(CpuKernelContext &ctx) {
  auto input_x_info = ctx.Input(0);
  auto input_grads_info = ctx.Input(1);
  auto input_argmax_info = ctx.Input(2);
  auto output_y_info = ctx.Output(0);
  auto input_grads = reinterpret_cast<T *>(input_grads_info->GetData());
  auto input_argmax = reinterpret_cast<S *>(input_argmax_info->GetData());
  auto output_y = reinterpret_cast<T *>(output_y_info->GetData());
  AttrValue *attr_ksize = ctx.GetAttr("ksize");
  std::vector<int64_t> ksizeList = attr_ksize->GetListInt();
  AttrValue *attr_strides = ctx.GetAttr("strides");
  std::vector<int64_t> stridesList = attr_strides->GetListInt();
  AttrValue *attr_pads = ctx.GetAttr("pads");
  std::vector<int64_t> padsList = attr_pads->GetListInt();
  AttrValue *attr_dilation = ctx.GetAttr("dilation");
  std::vector<int64_t> initList = {1, 1, 1, 1, 1};
  std::vector<int64_t> dilationList = (attr_dilation == nullptr) ? initList : attr_dilation->GetListInt();

  auto input_shape_vec = input_x_info->GetTensorShape()->GetDimSizes();
  auto output_shape_vec = input_grads_info->GetTensorShape()->GetDimSizes();
  const int64_t in_width = input_shape_vec[4];
  const int64_t in_height = input_shape_vec[3];
  const int64_t in_depth = input_shape_vec[2];
  const int64_t in_channel = input_shape_vec[1];
  const int64_t in_batch = input_shape_vec[0];
  const int64_t out_width = output_shape_vec[4];
  const int64_t out_height = output_shape_vec[3];
  const int64_t out_depth = output_shape_vec[2];
  const size_t DIM_SIZE1 = 1;
  const size_t DIM_SIZE5 = 5;
  std::vector<int64_t> ksizeTempList;
  if (ksizeList.size() == DIM_SIZE1) {
    ksizeTempList.push_back(ksizeList[0]);
    ksizeTempList.push_back(ksizeList[0]);
    ksizeTempList.push_back(ksizeList[0]);
  } else {
    ksizeTempList.push_back(ksizeList[0]);
    ksizeTempList.push_back(ksizeList[1]);
    ksizeTempList.push_back(ksizeList[2]);
  }
  std::vector<int64_t> stridesTempList;
  if (stridesList.size() == DIM_SIZE1) {
    stridesTempList.push_back(stridesList[0]);
    stridesTempList.push_back(stridesList[0]);
    stridesTempList.push_back(stridesList[0]);
  } else {
    stridesTempList.push_back(stridesList[0]);
    stridesTempList.push_back(stridesList[1]);
    stridesTempList.push_back(stridesList[2]);
  }
  std::vector<int64_t> padsTempList;
  if (padsList.size() == DIM_SIZE1) {
    padsTempList.push_back(padsList[0]);
    padsTempList.push_back(padsList[0]);
    padsTempList.push_back(padsList[0]);
  } else {
    padsTempList.push_back(padsList[0]);
    padsTempList.push_back(padsList[1]);
    padsTempList.push_back(padsList[2]);
  }
  std::vector<int64_t> dilationTempList;
  if (dilationList.size() == DIM_SIZE1) {
    dilationTempList.push_back(dilationList[0]);
    dilationTempList.push_back(dilationList[0]);
    dilationTempList.push_back(dilationList[0]);
  } else if (dilationList.size() == DIM_SIZE5) {
    dilationTempList.push_back(dilationList[2]);
    dilationTempList.push_back(dilationList[3]);
    dilationTempList.push_back(dilationList[4]);
  } else {
    dilationTempList.push_back(dilationList[1]);
    dilationTempList.push_back(dilationList[2]);
    dilationTempList.push_back(dilationList[3]);
  }
  const int64_t k_width = ksizeTempList[2];
  const int64_t k_height = ksizeTempList[1];
  const int64_t k_depth = ksizeTempList[0];
  const int64_t s_width = stridesTempList[2];
  const int64_t s_height = stridesTempList[1];
  const int64_t s_depth = stridesTempList[0];
  const int64_t p_width = padsTempList[2];
  const int64_t p_height = padsTempList[1];
  const int64_t p_depth = padsTempList[0];
  const int64_t d_width = dilationTempList[2];
  const int64_t d_height = dilationTempList[1];
  const int64_t d_depth = dilationTempList[0];
  KERNEL_CHECK_FALSE(k_width / 2 >= p_width && k_height / 2 >= p_height && k_depth / 2 >= p_depth,
                     KERNEL_STATUS_PARAM_INVALID, "pads should be smaller than or equal to half of kernel size.");

  int64_t data_num = ctx.Input(0)->NumElements();
  const int64_t batch = in_batch * in_channel;
  const int64_t in_stride = in_width * in_height * in_depth;
  const int64_t out_stride = out_width * out_height * out_depth;
  const int64_t kParallelDataNum = 16 * in_width * in_height * in_depth;
  const int64_t kParallelDataNumMid = 72 * in_width * in_height * in_depth;
  const float ZERO = 0.f;
  int64_t output_num = ctx.Output(0)->NumElements();
  for (int64_t i = 0; i < output_num; i++) {
    output_y[i] = static_cast<T>(ZERO);
  }
  if (data_num >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

    if (data_num <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    auto sharder_max_pool3d_grad_with_argmax = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        MaxPool3DGradWithArgmaxSingleCompute(input_grads + i * out_stride, input_argmax + i * out_stride,
                                             output_y + i * in_stride, in_depth, in_height, in_width, out_depth,
                                             out_height, out_width, k_depth, k_height, k_width, s_depth, s_height,
                                             s_width, p_depth, p_height, p_width, d_depth, d_height, d_width);
      }
    };

    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }
    KERNEL_HANDLE_ERROR(
      CpuKernelUtils::ParallelFor(ctx, batch, batch / max_core_num, sharder_max_pool3d_grad_with_argmax),
      "MaxPool3DGradWithArgmax Compute failed.");

  } else {
    for (int64_t i = 0; i < batch; i++) {
      MaxPool3DGradWithArgmaxSingleCompute(input_grads + i * out_stride, input_argmax + i * out_stride,
                                           output_y + i * in_stride, in_depth, in_height, in_width, out_depth,
                                           out_height, out_width, k_depth, k_height, k_width, s_depth, s_height,
                                           s_width, p_depth, p_height, p_width, d_depth, d_height, d_width);
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kMaxPool3DGradWithArgmax, MaxPool3DGradWithArgmaxCpuKernel);
}  // namespace aicpu
