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

#include "cpu_kernel/ms_kernel/concat.h"

#include <algorithm>
#include <complex>
#include <utility>

#include "utils/kernel_util.h"
#include "common/kernel_log.h"

namespace {
const char *const Concat = "Concat";
}

namespace aicpu {
uint32_t ConcatCpuKernel::CheckAndInitParams(const CpuKernelContext &ctx) {
  if (ctx.GetAttr("N") == nullptr) {
    n_ = 1;
  } else {
    AttrValue *n_ptr = ctx.GetAttr("N");
    n_ = n_ptr->GetInt();
  }
  // "x" is a list of at least 2 "tensor" objects of the same type
  KERNEL_CHECK_FALSE((n_ >= 2), KERNEL_STATUS_PARAM_INVALID, "Attr N must >= 2, but got attr N[%lld]", n_);

  uint32_t input_num = ctx.GetInputsSize();

  // input_num is n_(concat tensor num) + 1(concat_dim)
  KERNEL_CHECK_FALSE((static_cast<int64_t>(input_num) - 1 == n_), KERNEL_STATUS_PARAM_INVALID,
                     "Input num must equal attr N[%lld + 1],"
                     "but got input num[%u]",
                     n_, input_num);

  Tensor *concat_dim_ptr = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(concat_dim_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input concat_dim failed.");
  auto concat_dim_shape_ptr = concat_dim_ptr->GetTensorShape();
  KERNEL_CHECK_NULLPTR(concat_dim_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input concat_dim shape failed.");
  int32_t concat_dim_dims = concat_dim_shape_ptr->GetDims();
  KERNEL_CHECK_FALSE((concat_dim_dims == 0) || ((concat_dim_dims == 1) && (concat_dim_shape_ptr->NumElements() == 1)),
                     KERNEL_STATUS_PARAM_INVALID, "Input concat_dim should be a scalar integer, but got rank[%d].",
                     concat_dim_dims);
  int32_t concat_dim = 0;
  DataType concat_dim_data_type = concat_dim_ptr->GetDataType();
  KERNEL_CHECK_FALSE((concat_dim_data_type == DT_INT32 || concat_dim_data_type == DT_INT64),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Input concat_dim data type must DT_INT32 or DT_INT64,"
                     "but got data type[%d].",
                     DTypeStr(concat_dim_data_type).c_str());
  auto concat_dim_data_ptr = concat_dim_ptr->GetData();
  KERNEL_CHECK_NULLPTR(concat_dim_data_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input concat_dim data failed.");
  if (concat_dim_data_type == DT_INT32) {
    concat_dim = static_cast<int64_t>(*reinterpret_cast<int32_t *>(concat_dim_data_ptr));
  } else {
    concat_dim = *reinterpret_cast<int64_t *>(concat_dim_data_ptr);
  }

  Tensor *input0_ptr = ctx.Input(1);
  auto input0_type_ptr = input0_ptr->GetDataType();
  for (int64_t i = 1; i < n_; i++) {
    Tensor *inputi_ptr = ctx.Input(i);
    KERNEL_CHECK_NULLPTR(inputi_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input xi failed.");
    auto inputi_shape_ptr = inputi_ptr->GetTensorShape();
    auto inputi_type_ptr = inputi_ptr->GetDataType();
    KERNEL_CHECK_NULLPTR(inputi_shape_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input xi shape failed.");
    KERNEL_CHECK_FALSE((input0_type_ptr == inputi_type_ptr), KERNEL_STATUS_PARAM_INVALID,
                       "Input tensor should have same type, but got %d and %d.", DTypeStr(input0_type_ptr).c_str(),
                       DTypeStr(inputi_type_ptr).c_str());
  }
  auto input0_shape_ptr = input0_ptr->GetTensorShape();
  input_dims_ = input0_shape_ptr->GetDims();
  data_type_ = input0_ptr->GetDataType();
  axis_ = concat_dim < 0 ? concat_dim + input_dims_ : concat_dim;
  KERNEL_CHECK_FALSE((0 <= axis_ && axis_ < input_dims_), KERNEL_STATUS_PARAM_INVALID,
                     "Input concat_dim need in the "
                     "range[%d, %d), but got %lld.",
                     -input_dims_, input_dims_, concat_dim);
  inputs_flat_dim0_ = 1;
  for (uint32_t d = 0; d < axis_; ++d) {
    inputs_flat_dim0_ *= input0_shape_ptr->GetDimSize(d);
  }
  return KERNEL_STATUS_OK;
}

uint32_t ConcatCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_CHECK_FALSE((CheckAndInitParams(ctx) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                     "CheckAndInitParams failed.");
  switch (data_type_) {
    case DT_FLOAT16:
      return DoCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return DoCompute<float>(ctx);
    case DT_DOUBLE:
      return DoCompute<double>(ctx);
    case DT_INT8:
      return DoCompute<int8_t>(ctx);
    case DT_INT16:
      return DoCompute<int16_t>(ctx);
    case DT_INT32:
      return DoCompute<int32_t>(ctx);
    case DT_INT64:
      return DoCompute<int64_t>(ctx);
    case DT_UINT8:
      return DoCompute<uint8_t>(ctx);
    case DT_UINT16:
      return DoCompute<uint16_t>(ctx);
    case DT_UINT32:
      return DoCompute<uint32_t>(ctx);
    case DT_UINT64:
      return DoCompute<uint64_t>(ctx);
    case DT_COMPLEX64:
      return DoCompute<std::complex<float>>(ctx);
    case DT_COMPLEX128:
      return DoCompute<std::complex<double>>(ctx);
    default:
      KERNEL_LOG_ERROR("unsupported datatype[%d]", DTypeStr(data_type_).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T>
uint32_t ConcatCpuKernel::PrepareInput(const CpuKernelContext &ctx,
                                       std::vector<std::shared_ptr<typename TTypes<T>::ConstMatrix>> &inputs) {
  inputs.reserve(n_);
  output_concat_dim_ = 0;
  auto input0_shape_ptr = ctx.Input(1)->GetTensorShape();
  for (uint32_t i = 0; i < n_; ++i) {
    Tensor *input_i_ptr = ctx.Input(i + 1);
    int64_t input_i_num = input_i_ptr->NumElements();
    if (input_i_num == 0) {
      continue;
    }
    auto input_i_shape_ptr = input_i_ptr->GetTensorShape();
    int32_t input_i_dims = input_i_shape_ptr->GetDims();
    KERNEL_CHECK_FALSE((input_i_dims == input_dims_), KERNEL_STATUS_PARAM_INVALID,
                       "Ranks of inputs should match: shape[0]=%d vs. shape[%u]=%d", input_dims_, i, input_i_dims);
    for (int32_t j = 0; j < input_dims_; ++j) {
      int64_t dim_ij = input_i_shape_ptr->GetDimSize(j);
      if (j == axis_) {
        output_concat_dim_ += input_i_dims > 0 ? dim_ij : 1;
        continue;
      }
      int64_t dim_0j = input0_shape_ptr->GetDimSize(j);
      KERNEL_CHECK_FALSE((dim_0j == dim_ij), KERNEL_STATUS_PARAM_INVALID,
                         "Dimensions of inputs should match: shape[0][%d]=%lld vs."
                         "shape[%u][%d]=%lld",
                         j, dim_0j, i, j, dim_ij);
    }

    int64_t inputs_flat_dim1 = input_i_num / inputs_flat_dim0_;
    auto input_i_data_ptr = input_i_ptr->GetData();
    KERNEL_CHECK_NULLPTR(input_i_data_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input x%u data failed.", i);
    auto input_i = std::make_shared<typename TTypes<T>::ConstMatrix>(reinterpret_cast<T *>(input_i_data_ptr),
                                                                     inputs_flat_dim0_, inputs_flat_dim1);
    KERNEL_CHECK_NULLPTR(input_i, KERNEL_STATUS_PARAM_INVALID, "Create input x%u failed!", i);
    inputs.emplace_back(std::move(input_i));
  }

  if (input_dims_ == 0) {
    output_concat_dim_ = n_;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ConcatCpuKernel::PrepareOutput(const CpuKernelContext &ctx,
                                        std::shared_ptr<typename TTypes<T>::Matrix> &output) {
  Tensor *output_ptr = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output_ptr, KERNEL_STATUS_PARAM_INVALID, "Get output failed.");
  auto output_data_ptr = output_ptr->GetData();
  KERNEL_CHECK_NULLPTR(output_data_ptr, KERNEL_STATUS_PARAM_INVALID, "Get output data failed.");
  int64_t output_num = output_ptr->NumElements();
  int64_t output_dim1 = output_num / inputs_flat_dim0_;
  output = std::make_shared<typename TTypes<T>::Matrix>(reinterpret_cast<T *>(output_data_ptr), inputs_flat_dim0_,
                                                        output_dim1);
  KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_PARAM_INVALID, "Create output matrix failed.");
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ConcatCpuKernel::DoCompute(const CpuKernelContext &ctx) {
  std::vector<std::shared_ptr<typename TTypes<T>::ConstMatrix>> inputs;
  KERNEL_CHECK_FALSE((PrepareInput<T>(ctx, inputs) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                     "PrepareInput failed.");
  std::shared_ptr<typename TTypes<T>::Matrix> output = nullptr;
  KERNEL_CHECK_FALSE((PrepareOutput<T>(ctx, output) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                     "PrepareOutput failed.");
  if (inputs.size() > 0) {
    return ConcatCompute<T>(ctx, inputs, output);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ConcatCpuKernel::ConcatCompute(const CpuKernelContext &ctx,
                                        const std::vector<std::shared_ptr<typename TTypes<T>::ConstMatrix>> &inputs,
                                        std::shared_ptr<typename TTypes<T>::Matrix> &output) {
  size_t num_inputs = inputs.size();
  std::vector<ptrdiff_t> sizes;
  sizes.reserve(num_inputs);
  int64_t row_size = 0;
  for (const auto &input : inputs) {
    sizes.push_back(input->dimension(1));
    row_size += sizes.back();
  }
  uint32_t ret = KERNEL_STATUS_OK;
  auto work = [&row_size, &sizes, &inputs, &output, &num_inputs, &ret](int64_t start, int64_t end) {
    if (row_size == 0) {
      ret = KERNEL_STATUS_PARAM_INVALID;
      return;
    }
    int64_t skipped_rows = start / row_size;
    T *out = output->data() + skipped_rows * row_size;
    T *out_start = output->data() + start;
    T *out_end = output->data() + end;

    // Handle partial row at start
    if (out < out_start) {
      for (size_t j = 0; j < num_inputs; ++j) {
        ptrdiff_t size = sizes[j];
        ptrdiff_t offset = out_start - out;
        if (size <= offset) {
          out += size;
          continue;
        }
        const T *inp = &(*inputs[j])(skipped_rows, 0);
        if (offset > 0) {
          out += offset;
          inp += offset;
          size -= offset;
        }
        size = std::min(size, out_end - out);
        KERNEL_CHECK_FALSE_EXEC((size > 0), break)
        size_t copy_size = size * sizeof(T);
        error_t ret = memcpy_s(out, copy_size, inp, copy_size);
        if (ret != EOK) {
          KERNEL_LOG_ERROR("Memcpy failed.");
          ret = KERNEL_STATUS_INNER_ERROR;
          return;
        }
        out += size;
      }
      ++skipped_rows;
    }
    if (out < out_start || out > out_end) {
      KERNEL_LOG_ERROR("Out[%llx] not in range[%llx, %llx)", out, out_start, out_end);
      ret = KERNEL_STATUS_INNER_ERROR;
      return;
    }
    // Copy remaining data.
    std::vector<const T *> inp;
    inp.reserve(num_inputs);
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(inp),
                   [&skipped_rows](auto input) { return &(*input)(skipped_rows, 0); });
    const int64_t dim0 = output->dimension(0);
    for (int64_t i = skipped_rows; i < dim0; ++i) {
      for (int64_t j = 0; j < static_cast<int64_t>(num_inputs); ++j) {
        ptrdiff_t size = std::min(sizes[j], out_end - out);
        size_t copy_size = size * sizeof(T);
        auto ret = memcpy_s(out, copy_size, inp[j], copy_size);
        if (ret != EOK) {
          KERNEL_LOG_ERROR("Memcpy size[%zu] from inp[%llx] to out[%llx] failed.", copy_size, inp[j], out);
          ret = KERNEL_STATUS_INNER_ERROR;
          return;
        }
        out += size;
        inp[j] += size;
        if (!(out != out_end)) {
          return;
        }
      }
    }
  };
  const int64_t kParallelDataNumSameShapeBig = 255 * 1024;
  int64_t data_num = output->size();
  uint32_t min_core_num = 1;
  uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

  if (data_num >= kParallelDataNumSameShapeBig) {
    max_core_num = std::min(max_core_num, 6U);
  } else {
    max_core_num = std::min(max_core_num, 1U);
  }
  if (max_core_num == 0) {
    KERNEL_LOG_ERROR("max_core_num could not be 0.");
  }
  CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, work);
  KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), KERNEL_STATUS_INNER_ERROR, "ConcatCpuKernel failed.");
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(Concat, ConcatCpuKernel);
}  // namespace aicpu
