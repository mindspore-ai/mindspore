/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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
#include "ms_kernel/ragged_tensor_to_sparse.h"
#include <algorithm>

namespace {
const std::uint32_t kInputNum{aicpu::kDynamicInput};
const std::uint32_t kOutputNum{3u};
const char *kRaggedTensorToSparse = "RaggedTensorToSparse";
}  // namespace

namespace aicpu {
uint32_t RaggedTensorToSparseCpuKernel::CheckAndInitParams(const CpuKernelContext &ctx) {
  n_ = ctx.GetInputsSize() - 1;
  KERNEL_CHECK_FALSE((n_ >= 1), KERNEL_STATUS_PARAM_INVALID,
                     "Input num must great equal 1,"
                     "but got input num[%u]",
                     n_);
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "RaggedTensorToSparse check input and output number failed.");
  Tensor *rt_dense_values_ptr = ctx.Input(n_);
  KERNEL_CHECK_NULLPTR(rt_dense_values_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input rt_dense_values failed.");
  auto rt_dense_values_shape_ptr = rt_dense_values_ptr->GetTensorShape();
  KERNEL_CHECK_NULLPTR(rt_dense_values_shape_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "Get input rt_dense_values shape failed.");
  DataType rt_dense_values_data_type = rt_dense_values_ptr->GetDataType();
  KERNEL_CHECK_FALSE((rt_dense_values_data_type == DT_INT32 || rt_dense_values_data_type == DT_INT64 ||
                      rt_dense_values_data_type == DT_BOOL || rt_dense_values_data_type == DT_INT8 ||
                      rt_dense_values_data_type == DT_UINT8 || rt_dense_values_data_type == DT_INT16 ||
                      rt_dense_values_data_type == DT_UINT16 || rt_dense_values_data_type == DT_DOUBLE ||
                      rt_dense_values_data_type == DT_FLOAT || rt_dense_values_data_type == DT_FLOAT16),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Input rt_dense_values data type must {DT_BOOL, DT_INT8, "
                     "DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, "
                     "DT_DOUBLE, DT_FLOAT, DT_FLOAT16},"
                     "but got data type [%s].",
                     DTypeStr(rt_dense_values_data_type).c_str());
  auto rt_dense_values_data_ptr = rt_dense_values_ptr->GetData();
  KERNEL_CHECK_NULLPTR(rt_dense_values_data_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input rt_dense_values data failed.");
  return KERNEL_STATUS_OK;
}

// Validate `rt_nested_splits`
template <typename T1>
uint32_t RaggedTensorToSparseCpuKernel::ValidateInputs(std::vector<typename TTypes<T1>::Flat> rt_nested_splits,
                                                       const Tensor *rt_dense_values_in) {
  for (uint32_t i = 0; i < rt_nested_splits.size(); ++i) {
    if (rt_nested_splits[i].size() == 0) {
      KERNEL_LOG_ERROR("ragged splits may not be empty.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    if (rt_nested_splits[i](0) != 0) {
      KERNEL_LOG_ERROR("First value of ragged splits must be 0.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    for (uint32_t j = 1; j < rt_nested_splits[i].size(); ++j) {
      if (rt_nested_splits[i](j) < rt_nested_splits[i](j - 1)) {
        KERNEL_LOG_ERROR("Ragged splits should be non decreasing.");
        return KERNEL_STATUS_PARAM_INVALID;
      }
    }
    if (i > 0) {
      T1 last_split = rt_nested_splits[i - 1](rt_nested_splits[i - 1].size() - 1);
      if (rt_nested_splits[i].size() != last_split + 1) {
        KERNEL_LOG_ERROR(
          "Final value of ragged splits must match the length "
          "the corresponding ragged values.");
        return KERNEL_STATUS_PARAM_INVALID;
      }
    }
  }
  if (rt_dense_values_in->GetTensorShape()->GetDimSizes()[0] !=
      rt_nested_splits.back()(rt_nested_splits.back().size() - 1)) {
    KERNEL_LOG_ERROR(
      "Final value of ragged splits must match the length "
      "the corresponding ragged values.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

std::vector<std::vector<int64_t>> RaggedTensorToSparseCpuKernel::MakeIndexSuffixes(const TensorShape &values_shape) {
  std::vector<std::vector<int64_t>> suffixes{{}};
  for (int32_t dim = 1; dim < values_shape.GetDims(); ++dim) {
    std::vector<std::vector<int64_t>> new_suffixes;
    for (const auto &suffix : suffixes) {
      for (int64_t i = 0; i < values_shape.GetDimSize(dim); ++i) {
        new_suffixes.push_back(suffix);
        new_suffixes.back().push_back(i);
      }
    }
    suffixes.swap(new_suffixes);
  }
  return suffixes;
}

template <typename T1>
bool RaggedTensorToSparseCpuKernel::IsCompleted(const std::vector<int64_t> &pos, int dim,
                                                const std::vector<typename TTypes<T1>::Flat> &rt_nested_splits) {
  int64_t current_child = pos[dim + 1];
  int64_t limit_child = rt_nested_splits[dim](pos[dim] + 1);
  return current_child >= limit_child;
}

void RaggedTensorToSparseCpuKernel::input_list(CpuKernelContext *ctx, OpInputList *list) {
  if (ctx->Input(0)->NumElements() > 0) {
    static uint32_t stop;
    stop = static_cast<uint32_t>(ctx->Input(0)->NumElements());
    *list = OpInputList(ctx, 0, stop);
  }
}

template <typename T1, typename T2>
uint32_t RaggedTensorToSparseCpuKernel::DoCompute(CpuKernelContext *ctx) {
  // Assemble each value in `sparse_indices` using three parts:
  // - `index_prefix` is the index in dimensions up through the last ragged
  //   dimension.
  // - `index_middle` is the index in the last ragged dimension.
  // - `index_suffix` is the index in the dense value dimensions.
  OpInputList rt_nested_splits_in;
  input_list(ctx, &rt_nested_splits_in);
  const int64_t rt_nested_splits_len = n_;
  std::vector<typename TTypes<T1>::Flat> rt_nested_splits;
  rt_nested_splits.reserve(n_);
  for (int i = 0; i < rt_nested_splits_len; ++i) {
    if (rt_nested_splits_in[i]->NumElements() > 0) {
      EigenTensor indicesET(rt_nested_splits_in[i], rt_nested_splits_in[i]->GetData());
      Eigen::Tensor<T1, 1, Eigen::RowMajor, Eigen::DenseIndex> m = indicesET.flat<T1>();
      rt_nested_splits.push_back(indicesET.flat<T1>());
    }
  }

  const Tensor *rt_dense_values_in = ctx->Input(n_);
  KERNEL_CHECK_FALSE((ValidateInputs<T1>(rt_nested_splits, rt_dense_values_in) == KERNEL_STATUS_OK),
                     KERNEL_STATUS_PARAM_INVALID, "ValidateInputs failed.");
  KERNEL_CHECK_FALSE((Update<T1>(*ctx, rt_nested_splits) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                     "Update failed.");
  OutPutSparseValues<T2>(*ctx);
  OutPutSparseDenseShape<T1>(*ctx, rt_nested_splits_in, rt_nested_splits);
  return KERNEL_STATUS_OK;
}

template <typename T1>
uint32_t RaggedTensorToSparseCpuKernel::Update(const CpuKernelContext &ctx,
                                               std::vector<typename TTypes<T1>::Flat> rt_nested_splits) {
  const Tensor *rt_dense_values_in = ctx.Input(n_);
  const int64_t rt_nested_splits_len = n_;

  std::vector<int64_t> index_prefix(n_);
  std::vector<std::vector<int64_t>> index_suffixes = MakeIndexSuffixes(*rt_dense_values_in->GetTensorShape());

  // Allocate the `sparse_indices` output tensor.
  const int64_t nvals = (rt_nested_splits.back()(rt_nested_splits.back().size() - 1) * index_suffixes.size());
  const int64_t indices_len = rt_nested_splits_len + rt_dense_values_in->GetTensorShape()->GetDims();
  Tensor *sparse_indices = ctx.Output(0);
  KERNEL_CHECK_NULLPTR((sparse_indices), KERNEL_STATUS_PARAM_INVALID, "Get sparse_indices failed.");
  sparse_indices->SetDataType(DT_INT64);
  auto sparse_indices_ptr = reinterpret_cast<int64_t *>(sparse_indices->GetData());
  KERNEL_CHECK_NULLPTR(sparse_indices_ptr, KERNEL_STATUS_PARAM_INVALID, "Get sparse_indices data failed.");
  KERNEL_CHECK_NULLPTR(sparse_indices, KERNEL_STATUS_PARAM_INVALID, "Create sparse_indices Flat failed.");

  // pos[i] is the current position in rt_nested_splits[i].  final_pos is a
  // reference to make it easier to refer to pos[-1].
  std::vector<int64_t> pos(n_);
  int64_t &final_pos = pos[n_ - 1];
  // Each iteration through the loop, we increment pos[-1], and add indices
  // for all the values corresponding to
  // rt_nested_splits[-1][pos[-1]:pos[-1]+1].
  int next_index = 0;
  int64_t num = 0;
  int max_final_pos = rt_nested_splits.back().size() - 1;
  for (; final_pos < max_final_pos; ++final_pos) {
    // Update `pos` to skip over completed elements (i.e., elements where
    // we have already generated indices for all contained values).
    for (int dim = n_ - 2; dim >= 0; --dim) {
      while (IsCompleted<T1>(pos, dim, rt_nested_splits)) {
        pos[dim] += 1;
      }
    }
    // Update index_prefix.
    for (size_t dim = 0; dim < index_prefix.size(); ++dim) {
      int start = dim > 0 ? rt_nested_splits[dim - 1](pos[dim - 1]) : 0;
      index_prefix[dim] = pos[dim] - start;
    }
    // Get length of the final-ragged-dimension slice.
    const auto &final_splits = rt_nested_splits[n_ - 1];
    int64_t slice_len = final_splits(final_pos + 1) - final_splits(final_pos);
    // Add sparse_indices for this slice.
    for (int64_t i = 0; i < slice_len; ++i) {
      for (const auto &index_suffix : index_suffixes) {
        int dim = 0;
        for (int64_t index : index_prefix) {  // index_prefix
          sparse_indices_ptr[num++] = index;
          dim++;
        }
        dim++;
        sparse_indices_ptr[num++] = i;
        for (int64_t index : index_suffix) {  // index_suffix
          sparse_indices_ptr[num++] = index;
          dim++;
        }
        KERNEL_CHECK_FALSE((dim == indices_len), KERNEL_STATUS_PARAM_INVALID,
                           "dim should be equal to indices_len,but get %d.", dim);
        ++next_index;
      }
    }
  }
  KERNEL_CHECK_FALSE((next_index == nvals), KERNEL_STATUS_PARAM_INVALID,
                     "next_index should be equal to nvals,but get %d.", next_index);
  return KERNEL_STATUS_OK;
}

template <typename T2>
void RaggedTensorToSparseCpuKernel::OutPutSparseValues(const CpuKernelContext &ctx) {
  // Output the `sparse_values` Tensor.
  const Tensor *rt_dense_values_in = ctx.Input(n_);
  Tensor *spares_values_out = ctx.Output(1);
  spares_values_out->SetDataType(rt_dense_values_in->GetDataType());
  auto spares_values_out_ptr = reinterpret_cast<T2 *>(spares_values_out->GetData());
  auto rt_dense_values_in_ptr = reinterpret_cast<T2 *>(rt_dense_values_in->GetData());
  for (int64_t i = 0; i < rt_dense_values_in->NumElements(); i++) {
    spares_values_out_ptr[i] = rt_dense_values_in_ptr[i];
  }
}

template <typename T1>
void RaggedTensorToSparseCpuKernel::OutPutSparseDenseShape(const CpuKernelContext &ctx, OpInputList rt_nested_splits_in,
                                                           std::vector<typename TTypes<T1>::Flat> rt_nested_splits) {
  // Output the `sparse_dense_shape` Tensor.
  const Tensor *rt_dense_values_in = ctx.Input(n_);
  Tensor *sparse_dense_shape_out = ctx.Output(2);
  int64_t *sparse_dense_shape = static_cast<int64_t *>(sparse_dense_shape_out->GetData());
  sparse_dense_shape[0] = rt_nested_splits_in[0]->GetTensorShape()->GetDimSizes()[0] - 1;
  for (int dim = 0; dim < n_; ++dim) {
    const auto &splits = rt_nested_splits[dim];
    T1 max_width = 0;
    for (int i = 1; i < splits.size(); ++i) {
      max_width = std::max(max_width, splits(i) - splits(i - 1));
    }
    sparse_dense_shape[dim + 1] = max_width;
  }
  for (int dim = 1; dim < rt_dense_values_in->GetTensorShape()->GetDims(); ++dim) {
    sparse_dense_shape[dim + n_] = rt_dense_values_in->GetTensorShape()->GetDimSizes()[dim];
  }
}

uint32_t RaggedTensorToSparseCpuKernel::ComputeWithSplitTypeInt32(CpuKernelContext *ctx) {
  type1 = ctx->Input(n_)->GetDataType();
  switch (type1) {
    case DT_DOUBLE:
      return DoCompute<int32_t, double>(ctx);
    case DT_FLOAT16:
      return DoCompute<int32_t, Eigen::half>(ctx);
    case DT_FLOAT:
      return DoCompute<int32_t, float>(ctx);
    case DT_INT8:
      return DoCompute<int32_t, int8_t>(ctx);
    case DT_INT16:
      return DoCompute<int32_t, int16_t>(ctx);
    case DT_INT32:
      return DoCompute<int32_t, int32_t>(ctx);
    case DT_INT64:
      return DoCompute<int32_t, int64_t>(ctx);
    case DT_UINT8:
      return DoCompute<int32_t, uint8_t>(ctx);
    case DT_UINT16:
      return DoCompute<int32_t, uint16_t>(ctx);
    case DT_BOOL:
      return DoCompute<int32_t, bool>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported datatype [%s]", DTypeStr(type1).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

uint32_t RaggedTensorToSparseCpuKernel::ComputeWithSplitTypeInt64(CpuKernelContext *ctx) {
  type1 = ctx->Input(n_)->GetDataType();
  switch (type1) {
    case DT_DOUBLE:
      return DoCompute<int64_t, double>(ctx);
    case DT_FLOAT16:
      return DoCompute<int64_t, Eigen::half>(ctx);
    case DT_FLOAT:
      return DoCompute<int64_t, float>(ctx);
    case DT_INT8:
      return DoCompute<int64_t, int8_t>(ctx);
    case DT_INT16:
      return DoCompute<int64_t, int16_t>(ctx);
    case DT_INT32:
      return DoCompute<int64_t, int32_t>(ctx);
    case DT_INT64:
      return DoCompute<int64_t, int64_t>(ctx);
    case DT_UINT8:
      return DoCompute<int64_t, uint8_t>(ctx);
    case DT_UINT16:
      return DoCompute<int64_t, uint16_t>(ctx);
    case DT_BOOL:
      return DoCompute<int64_t, bool>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported datatype [%s]", DTypeStr(type1).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

uint32_t RaggedTensorToSparseCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_CHECK_FALSE((CheckAndInitParams(ctx) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                     "CheckAndInitParams failed.");
  DataType SplitType = ctx.Input(0)->GetDataType();
  switch (SplitType) {
    case DT_INT32:
      return ComputeWithSplitTypeInt32(&ctx);
    case DT_INT64:
      return ComputeWithSplitTypeInt64(&ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported datatype [%s]", DTypeStr(SplitType).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
REGISTER_CPU_KERNEL(kRaggedTensorToSparse, RaggedTensorToSparseCpuKernel);
}  // namespace aicpu
