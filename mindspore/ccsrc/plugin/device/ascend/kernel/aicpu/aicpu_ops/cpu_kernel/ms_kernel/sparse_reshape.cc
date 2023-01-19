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
#include "sparse_reshape.h"
#include <vector>
#include "cpu_kernel_utils.h"
#include "securec.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kSparseReshapeInputNum = 3;
constexpr uint32_t kSparseReshapeOutputNum = 2;
const char *kSparseReshape = "SparseReshape";
// when input data size is more than kParallelDataNum, use Parallel func
const int64_t kParallelDataNumSameShape = 24 * 1024;
const int64_t kParallelDataNumSameShapeMid = 35 * 1024;
}  // namespace

namespace aicpu {
void SparseReshapeCpuKernel::SpecialCompute(int64_t start, int64_t end, const int64_t *in0, int64_t *out0,
                                            const int64_t *input_strides, const int64_t *output_strides,
                                            const int64_t input_rank, const int64_t output_rank) {
  for (int i = start; i < end; i++) {
    int64_t id = 0;
    for (int j = 0; j < input_rank; j++) {
      id += *(in0 + i * input_rank + j) * input_strides[j];
    }
    for (int j = 0; j < output_rank; j++) {
      *(out0 + i * output_rank + j) = id / output_strides[j];
      id %= output_strides[j];
    }
  }
}

uint32_t SparseReshapeCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kSparseReshapeInputNum, kSparseReshapeOutputNum), "[%s] check params failed.",
                      kSparseReshape);

  Tensor *input_0 = ctx.Input(0);
  Tensor *input_1 = ctx.Input(1);
  Tensor *input_2 = ctx.Input(2);
  Tensor *output_0 = ctx.Output(0);
  Tensor *output_1 = ctx.Output(1);

  KERNEL_CHECK_FALSE(
    (input_0->GetDataType() == DT_INT64 && input_1->GetDataType() == DT_INT64 && input_2->GetDataType() == DT_INT64 &&
     output_0->GetDataType() == DT_INT64 && output_1->GetDataType() == DT_INT64),
    KERNEL_STATUS_INNER_ERROR, "the data of SparseReshape kernel must be DT_INT64.");
  KERNEL_CHECK_FALSE((input_0->GetTensorShape()->GetDimSize(1) == input_1->GetTensorShape()->GetDimSize(0)),
                     KERNEL_STATUS_INNER_ERROR, "Input tensor rank must match input shape length.");

  int64_t *in0 = reinterpret_cast<int64_t *>(input_0->GetData());
  int64_t *in1 = reinterpret_cast<int64_t *>(input_1->GetData());
  int64_t *in2 = reinterpret_cast<int64_t *>(input_2->GetData());
  int64_t *out0 = reinterpret_cast<int64_t *>(output_0->GetData());
  int64_t *out1 = reinterpret_cast<int64_t *>(output_1->GetData());

  const int64_t input_rank = input_1->NumElements();
  const int64_t output_rank = input_2->NumElements();
  const int64_t nnz = input_0->GetTensorShape()->GetDimSize(0);
  int64_t dense_size = 1;
  int64_t product = 1;
  int64_t out_num = 1;
  int unknown_index = -1;

  for (int i = 0; i < input_rank; i++) {
    dense_size *= *(in1 + i);
  }
  for (int d = 0; d < output_rank; d++) {
    const int64_t size = *(in2 + d);
    if (size == -1) {
      KERNEL_CHECK_FALSE((unknown_index == -1), KERNEL_STATUS_INNER_ERROR,
                         "only one output dimension may be -1, "
                         "not both [%d] and [%d]",
                         unknown_index, d);
      unknown_index = d;
    } else {
      KERNEL_CHECK_FALSE((size >= 0), KERNEL_STATUS_INNER_ERROR, "size [%d] must be non-negative, not [%ld]", d, size);
      product *= size;
      *(out1 + d) = size;
      out_num *= size;
    }
  }

  if (unknown_index != -1) {
    KERNEL_CHECK_FALSE((product >= 0), KERNEL_STATUS_INNER_ERROR,
                       "reshape cannot infer the missing "
                       "input size for an empty tensor unless all "
                       "specified input sizes are non-zero");
    const int64_t missing = dense_size / product;
    KERNEL_CHECK_FALSE((product * missing == dense_size), KERNEL_STATUS_INNER_ERROR,
                       "Input to reshape is a SparseTensor with [%ld]"
                       " dense values, but the requested shape requires"
                       " a multiple of [%ld].",
                       dense_size, product);
    out_num *= missing;
    *(out1 + unknown_index) = missing;
  }

  KERNEL_CHECK_FALSE((out_num == dense_size), KERNEL_STATUS_INNER_ERROR,
                     "Input to reshape is a tensor with [%ld]"
                     " dense values, but the requested shape has [%ld].",
                     dense_size, out_num);

  int64_t input_size = input_0->GetDataSize();
  int64_t output_size = output_0->GetDataSize();
  if (input_size == output_size && input_rank == output_rank) {
    bool flag = true;
    for (int64_t i = 0; i < input_rank; ++i) {
      if (*(in1 + i) != *(out1 + i)) {
        flag = false;
        break;
      }
    }
    if (flag) {
      auto mem_ret = memcpy_s(out0, output_size, in0, input_size);
      KERNEL_CHECK_FALSE(mem_ret == EOK, KERNEL_STATUS_INNER_ERROR,
                         "[%s] memcpy_s to output failed, destMax [%ld], count [%ld].", kSparseReshape, output_size,
                         input_size);
      return KERNEL_STATUS_OK;
    }
  }

  if (nnz <= 0) return KERNEL_STATUS_OK;
  int64_t *input_strides = new int64_t[input_rank];
  int64_t *output_strides = new int64_t[output_rank];

  if (input_rank > 0) {
    input_strides[input_rank - 1] = 1;
    for (int d = input_rank - 2; d >= 0; d--) {
      input_strides[d] = input_strides[d + 1] * *(in1 + d + 1);
    }
  }
  if (output_rank > 0) {
    output_strides[output_rank - 1] = 1;
    for (int d = output_rank - 2; d >= 0; d--) {
      output_strides[d] = output_strides[d + 1] * *(out1 + d + 1);
    }
  }
  if (nnz * input_rank >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    KERNEL_CHECK_FALSE(max_core_num != 0, KERNEL_STATUS_INNER_ERROR, "core num should not be 0.");
    if (nnz * input_rank <= kParallelDataNumSameShapeMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }
    if (max_core_num > nnz) {
      max_core_num = nnz;
    }
    auto sharder_sparse_reshape = [&](int64_t start, int64_t end) {
      SpecialCompute(start, end, in0, out0, input_strides, output_strides, input_rank, output_rank);
    };

    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, nnz, nnz / max_core_num, sharder_sparse_reshape),
                        "SparseReshape Compute failed.");
  } else {
    SpecialCompute(0, nnz, in0, out0, input_strides, output_strides, input_rank, output_rank);
  }

  delete[] input_strides;
  delete[] output_strides;
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kSparseReshape, SparseReshapeCpuKernel);
}  // namespace aicpu