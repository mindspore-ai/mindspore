/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/sparse_reshape_cpu_kernel.h"
#include <algorithm>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kIndicesShapeSize = 2;
constexpr size_t kSparseReshapeInputsNum = 3;
constexpr size_t kSparseReshapeOutputsNum = 2;
}  // namespace

bool SparseReshapeCpuKernelMod::SameConvert(int64_t input_size, int64_t output_size, int64_t input_rank,
                                            int64_t output_rank, int64_t *in0, const int64_t *in1, int64_t *out0,
                                            const int64_t *out1) {
  if (input_size == output_size && input_rank == output_rank) {
    bool flag = true;
    for (int64_t i = 0; i < input_rank; ++i) {
      if (*(in1 + i) != *(out1 + i)) {
        flag = false;
        break;
      }
    }
    if (flag) {
      auto ret = memcpy_s(out0, output_size, in0, input_size);
      if (ret != 0) {
        MS_LOG(EXCEPTION) << "Memcpy_s error, errorno" << ret;
      }
    }
    return true;
  }
  return false;
}

void SparseReshapeCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  indices_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (indices_shape_.size() != kIndicesShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it requires 'indices' should be a " << kIndicesShapeSize
                      << "-D Tensor, but got " << indices_shape_.size() << "-D";
  }
  auto shape_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  if (indices_shape_[1] != shape_shape[0]) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the rank of input tensor must match input shape length, but got input tensor rank = "
                      << indices_shape_[1] << ", and input shape length = " << shape_shape[0] << ".";
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it requires int64 data type, but got " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename I, typename T>
bool SparseReshapeCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                             const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseReshapeInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseReshapeOutputsNum, kernel_name_);
  auto in0 = reinterpret_cast<int64_t *>(inputs[0]->addr);
  auto in1 = reinterpret_cast<int64_t *>(inputs[1]->addr);
  auto in2 = reinterpret_cast<int64_t *>(inputs[2]->addr);
  auto out0 = reinterpret_cast<int64_t *>(outputs[0]->addr);
  auto out1 = reinterpret_cast<int64_t *>(outputs[1]->addr);
  const int64_t input_rank = SizeToLong(inputs[1]->size / sizeof(int64_t));
  const int64_t output_rank = SizeToLong(inputs[2]->size / sizeof(int64_t));
  const int64_t nnz = indices_shape_[0];

  int64_t dense_size = 1;
  int64_t dividend = 1;
  int64_t out_num = 1;
  int64_t ui = -1;
  for (int64_t i = 0; i < input_rank; i++) {
    dense_size *= *(in1 + i);
  }

  for (int64_t d = 0; d < output_rank; d++) {
    const int64_t size = *(in2 + d);
    if (size == -1) {
      if (ui != -1) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_
                          << "', there should be at most one '-1' dimension in 'newshape' tensor, but got two or more.";
      }
      ui = d;
    } else {
      if (size < 0) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the size of newshape rank-" << d
                          << " should be a non-negative number, but got " << size << ".";
      }
      dividend *= size;
      *(out1 + d) = size;
      out_num *= size;
    }
  }
  if (ui != -1) {
    (void)CheckAndConvertUtils::CheckInteger("divident", dividend, kGreaterThan, 0, kernel_name_);
    const int64_t missing = dense_size / dividend;
    if (dividend * missing != dense_size) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the requested shape should be a multiple of " << dividend
                        << " and " << missing << ", but got a SparseTensor with " << dense_size << " dense values.";
    }
    out_num *= missing;
    *(out1 + ui) = missing;
  }

  if (out_num != dense_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the requested shape has the dense shape of " << out_num
                      << ", but got the input newshape is a tensor with " << dense_size;
  }

  int64_t input_size = SizeToLong(inputs[0]->size);
  int64_t output_size = SizeToLong(outputs[0]->size);
  bool same = SameConvert(input_size, output_size, input_rank, output_rank, in0, in1, out0, out1);
  if (same) {
    return true;
  }

  std::vector<int64_t> input_strides(input_rank);
  std::vector<int64_t> output_strides(output_rank);
  input_strides[input_rank - 1] = 1;
  for (int64_t d = input_rank - 2; d >= 0; d--) {
    input_strides[d] = input_strides[d + 1] * *(in1 + d + 1);
  }
  output_strides[output_rank - 1] = 1;
  for (int64_t d = output_rank - 2; d >= 0; d--) {
    output_strides[d] = output_strides[d + 1] * *(out1 + d + 1);
  }

  auto task = [&input_strides, &output_strides, &input_rank, &output_rank, &in0, &out0](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      int64_t id = 0;
      for (int64_t j = 0; j < input_rank; j++) {
        id += *(in0 + SizeToLong(i) * input_rank + j) * input_strides[j];
      }
      for (int64_t j = 0; j < output_rank; j++) {
        *(out0 + SizeToLong(i) * output_rank + j) = id / output_strides[j];
        id %= output_strides[j];
      }
    }
  };

  ParallelLaunchAutoSearch(task, nnz, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, SparseReshapeCpuKernelMod::SparseReshapeFunc>> SparseReshapeCpuKernelMod::func_list_ =
  {{KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64),
    &SparseReshapeCpuKernelMod::LaunchKernel<int64_t, int64_t>}};

std::vector<KernelAttr> SparseReshapeCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseReshapeFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseReshape, SparseReshapeCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
