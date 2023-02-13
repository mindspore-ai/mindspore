#ifndef AICPU_KERNELS_DEPTHTOSPACE_CC_
#define AICPU_KERNELS_DEPTHTOSPACE_CC_

#include "depth_to_space.h"

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
const char *kDepthToSpace = "DepthToSpace";

#define DEPTHTOSPACE_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                              \
    uint32_t result = DoCompute<TYPE>(CTX);                    \
    if (result != KERNEL_STATUS_OK) {                          \
      KERNEL_LOG_ERROR("DepthToSpace kernel compute failed."); \
      return result;                                           \
    }                                                          \
    break;                                                     \
  }
}  // namespace

namespace aicpu {
template <typename T>
uint32_t DepthToSpaceCpuKernel::DoCompute(CpuKernelContext &ctx) {
  auto input_shape = ctx.Input(0)->GetTensorShape();
  auto output_shape = ctx.Output(0)->GetTensorShape();
  auto input_dims = input_shape->GetDimSizes();
  std::vector<std::string> attr_name1 = {"data_format"};
  AttrValue *attr_data_format = ctx.GetAttr("data_format");
  std::vector<std::string> attr_name2 = {"block_size"};
  data_format_ = (attr_data_format == nullptr) ? "NHWC" : (attr_data_format->GetString());
  int64_t block_size = ctx.GetAttr("block_size")->GetInt();
  int64_t zero = 0;
  int64_t two = 2;
  int64_t n_nhwc = 0;
  int64_t h_nhwc = 1;
  int64_t w_nhwc = 2;
  int64_t c_nhwc = 3;
  int64_t n_nchw = 0;
  int64_t h_nchw = 1;
  int64_t w_nchw = 2;
  int64_t c_nchw = 3;
  if (block_size == zero && block_size * block_size == zero) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  KERNEL_CHECK_FALSE((block_size >= two), KERNEL_STATUS_PARAM_INVALID,
                     "The value of block_size must be greater than 2");

  std::vector<int64_t> output_dims;
  if (data_format_ == "NHWC") {
    KERNEL_CHECK_FALSE((input_dims[c_nhwc] % block_size * block_size == zero), KERNEL_STATUS_PARAM_INVALID,
                       "Channels must can be divided by block_size * block_size.");
    output_dims = {input_dims[n_nhwc], input_dims[h_nhwc] * block_size, input_dims[w_nhwc] * block_size,
                   input_dims[c_nhwc] / (block_size * block_size)};
    output_shape->SetDimSizes(output_dims);
    input_dims = {input_dims[n_nhwc], input_dims[c_nhwc], input_dims[h_nhwc], input_dims[w_nhwc]};
    output_dims = {output_dims[n_nhwc], output_dims[c_nhwc], output_dims[h_nhwc], output_dims[w_nhwc]};
  } else if (data_format_ == "NCHW") {
    KERNEL_CHECK_FALSE((input_dims[h_nchw] % block_size * block_size == zero), KERNEL_STATUS_PARAM_INVALID,
                       "Channels must can be divided by block_size * block_size.");
    output_dims = {input_dims[n_nchw], input_dims[h_nchw] / (block_size * block_size), input_dims[w_nchw] * block_size,
                   input_dims[c_nchw] * block_size};
    output_shape->SetDimSizes(output_dims);
  }

  auto input = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t x = 0;
  const size_t data_num = (size_t)ctx.Input(0)->NumElements();

  for (size_t i = 0; i < data_num; i = i + block_size) {
    for (size_t j = i; j < block_size + i; ++j) {
      if (j % (input_dims[h_nhwc] * input_dims[c_nhwc]) == 0) {
        x = -1;
      }
      if (j % output_dims[h_nhwc] == 0) {
        ++x;
      }
      size_t number = 0, output_pos = 0;
      size_t loc = j / output_dims[h_nhwc];
      number += (loc / (output_dims[w_nhwc] * output_dims[c_nhwc])) * output_dims[w_nhwc] * output_dims[c_nhwc];
      // Mark the position of this segment of the vector in the entire segment.
      number += (input_dims[h_nhwc] * input_dims[c_nhwc] / output_dims[h_nhwc]) *
                (loc / (input_dims[h_nhwc] * input_dims[c_nhwc] / output_dims[h_nhwc]));
      // Label the position of the block within a segment of the vector.
      number += ((loc % input_dims[h_nhwc]) / block_size) * block_size * input_dims[c_nhwc];
      // Mark the relative position within the small block.
      number += loc % block_size + (x / input_dims[h_nhwc]) * block_size;
      output_pos = j % output_dims[h_nhwc] + number * output_dims[h_nhwc];

      output[output_pos] = input[j];
    }
  }

  return KERNEL_STATUS_OK;
}  // DoCompute

uint32_t DepthToSpaceCpuKernel::STDParamCheck(CpuKernelContext &ctx) {
  // check params
  auto input = ctx.Input(0);
  auto output = ctx.Output(0);

  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "DepthToSpace check input and output number failed.");

  KERNEL_LOG_DEBUG(
    "DepthToSpaceCpuKernel[%s], input0: size[%llu];"
    "output: size[%llu].",
    ctx.GetOpType().c_str(), input->GetDataSize(), output->GetDataSize());

  // check data_format
  std::vector<std::string> attr_name1 = {"data_format"};
  AttrValue *attr_data_format = ctx.GetAttr("data_format");
  data_format_ = (attr_data_format == nullptr) ? "NHWC" : (attr_data_format->GetString());
  KERNEL_CHECK_FALSE((data_format_ == "NHWC" || data_format_ == "NCHW"), KERNEL_STATUS_PARAM_INVALID,
                     "The data_format must be NCHW, NHWC or NCHW_VECT_C, but got: [%s]", data_format_);

  return KERNEL_STATUS_OK;
}

uint32_t DepthToSpaceCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(STDParamCheck(ctx), "DepthToSpace check params failed.");
  Tensor *input0_tensor = ctx.Input(0);
  auto input_data_type = input0_tensor->GetDataType();

  switch (input_data_type) {
    DEPTHTOSPACE_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    DEPTHTOSPACE_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    DEPTHTOSPACE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    DEPTHTOSPACE_COMPUTE_CASE(DT_FLOAT, float, ctx)
    DEPTHTOSPACE_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    DEPTHTOSPACE_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    DEPTHTOSPACE_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    DEPTHTOSPACE_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    DEPTHTOSPACE_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    DEPTHTOSPACE_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    DEPTHTOSPACE_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    DEPTHTOSPACE_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    DEPTHTOSPACE_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    DEPTHTOSPACE_COMPUTE_CASE(DT_QINT8, int8_t, ctx)
    DEPTHTOSPACE_COMPUTE_CASE(DT_QINT16, int16_t, ctx)
    DEPTHTOSPACE_COMPUTE_CASE(DT_QINT32, int32_t, ctx)
    default:
      KERNEL_LOG_ERROR("DepthToSpace kernel data type[%s] not support.", DTypeStr(input_data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kDepthToSpace, DepthToSpaceCpuKernel);
}  // namespace aicpu
#endif  // AICPU_KERNELS_SPACETODEPTH_CC_
