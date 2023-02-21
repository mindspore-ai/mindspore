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

#include "plugin/device/cpu/kernel/ragged_tensor_to_sparse_cpu_kernel.h"
#include <algorithm>
#include <cstdio>
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
namespace mindspore {
namespace kernel {
namespace {
constexpr int64_t kRaggedTensorToSparseOutputsNum = 3;
constexpr int64_t kRaggedTensorToSparseInputsNum = 2;
constexpr int64_t kRttsInputStart = 0;
constexpr int64_t kInputValue = 1;
constexpr int64_t kOutputindecs = 0;
constexpr int64_t kOutputValue = 1;
constexpr int64_t kOutputShape = 2;
#define ALL_CASE_CHOOSE(TYPE)                                          \
  do {                                                                 \
    case kNumberTypeInt8:                                              \
      return LaunchKernel<TYPE, int8_t>(inputs, workspace, outputs);   \
    case kNumberTypeUInt8:                                             \
      return LaunchKernel<TYPE, uint8_t>(inputs, workspace, outputs);  \
    case kNumberTypeInt16:                                             \
      return LaunchKernel<TYPE, int16_t>(inputs, workspace, outputs);  \
    case kNumberTypeUInt16:                                            \
      return LaunchKernel<TYPE, uint16_t>(inputs, workspace, outputs); \
    case kNumberTypeUInt32:                                            \
      return LaunchKernel<TYPE, uint32_t>(inputs, workspace, outputs); \
    case kNumberTypeInt32:                                             \
      return LaunchKernel<TYPE, int32_t>(inputs, workspace, outputs);  \
    case kNumberTypeInt64:                                             \
      return LaunchKernel<TYPE, int64_t>(inputs, workspace, outputs);  \
    case kNumberTypeUInt64:                                            \
      return LaunchKernel<TYPE, uint64_t>(inputs, workspace, outputs); \
    case kNumberTypeFloat16:                                           \
      return LaunchKernel<TYPE, float16>(inputs, workspace, outputs);  \
    case kNumberTypeFloat32:                                           \
      return LaunchKernel<TYPE, float>(inputs, workspace, outputs);    \
    case kNumberTypeBool:                                              \
      return LaunchKernel<TYPE, bool>(inputs, workspace, outputs);     \
    case kNumberTypeFloat64:                                           \
      return LaunchKernel<TYPE, double>(inputs, workspace, outputs);   \
  } while (0);
}  // namespace

bool RaggedTensorToSparseCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  kernel_name_ = base_operator->name();
  auto input_num = inputs.size();
  n_ = input_num - 1;
  splits_type_ = inputs[0]->GetDtype();
  values_type_ = inputs[n_]->GetDtype();
  size_t min_input_num = 2;
  if (input_num < min_input_num) {
    MS_LOG(EXCEPTION) << "For op " << kernel_name_ << ", the input number must be bigger than " << min_input_num
                      << ", but got " << input_num << ".";
  }
  size_t output_num = outputs.size();
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kRaggedTensorToSparseOutputsNum, kernel_name_);

  return true;
}

bool RaggedTensorToSparseCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &workspace,
                                              const std::vector<AddressPtr> &outputs) {
  switch (splits_type_) {
    case kNumberTypeInt32:
      switch (values_type_) {
        ALL_CASE_CHOOSE(int32_t)
        default:
          MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of input_values "
                            << TypeIdToType(values_type_)->ToString() << " not support.";
      }
      break;
    case kNumberTypeInt64:
      switch (values_type_) {
        ALL_CASE_CHOOSE(int64_t)
        default:
          MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of input_values "
                            << TypeIdToType(values_type_)->ToString() << " not support.";
      }
      break;
    default:
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of input_splits "
                        << TypeIdToType(splits_type_)->ToString() << " not support.";
  }
  return true;
}

int RaggedTensorToSparseCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs,
                                             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  input2_shape_ = inputs[n_]->GetShapeVector();
  output1_shape_ = outputs[1]->GetShapeVector();
  return ret;
}

template <typename T1>
void RaggedTensorToSparseCpuKernelMod::ValidateInputs(const std::vector<std::vector<T1>> &input1) {
  int64_t input1_sizes = input1.size();
  for (int64_t i = 0; i < input1_sizes; ++i) {
    if (input1[i].size() == 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the size of ragged splits can not be 0.";
    }
    if (input1[i][0] != 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the first value of ragged splits must be 0, but got "
                        << input1[i][0] << ".";
    }
    for (unsigned int j = 1; j < input1[i].size(); ++j) {
      if (input1[i][j] < input1[i][j - 1]) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the ragged splits should be non decreasing, but got "
                          << input1[i][j - 1] << " followed by " << input1[i][j] << ".";
      }
    }
    if (i > 0) {
      int64_t length = input1[i].size();
      int64_t last_split = input1[i - 1][input1[i - 1].size() - 1];
      int64_t input1_s = last_split + 1;
      if (length != input1_s) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_
                          << "', the final value of ragged splits must equal with the length " << length << ", but got "
                          << input1_s << ".";
      }
    }
  }
}

template <typename T1>
void RaggedTensorToSparseCpuKernelMod::Update(const std::vector<std::vector<T1>> &input1, int64_t *output1_ptr,
                                              const std::vector<std::vector<int64_t>> &index_suffixes,
                                              std::vector<int64_t> index_prefix) {
  int64_t nvals = input1.back()[input1.back().size() - 1] * index_suffixes.size();
  int64_t indices_len = n_ + input2_shape_.size();
  output1_shape_[0] = nvals;
  output1_shape_[1] = indices_len;

  std::vector<int64_t> pos(n_);
  int64_t &final_pos = pos[n_ - 1];
  int64_t next_index = 0;
  int64_t num = 0;
  int64_t max_final_pos = input1.back().size() - 1;

  for (; final_pos < max_final_pos; ++final_pos) {
    for (int64_t dim = n_ - 2; dim >= 0; --dim) {
      int64_t current_child = pos[dim + 1];
      int64_t limit_child = input1[dim][pos[dim] + 1];
      while (current_child >= limit_child) {
        pos[dim] += 1;
        current_child = pos[dim + 1];
        limit_child = input1[dim][pos[dim] + 1];
      }
    }
    int64_t index_pre_size = index_prefix.size();
    for (int64_t dim = 0; dim < index_pre_size; ++dim) {
      int64_t start = dim > 0 ? input1[dim - 1][pos[dim - 1]] : 0;
      index_prefix[dim] = pos[dim] - start;
    }

    const auto &final_splits = input1[n_ - 1];
    int64_t slice_len = final_splits[final_pos + 1] - final_splits[final_pos];
    for (int64_t i = 0; i < slice_len; ++i) {
      for (const auto &index_suffix : index_suffixes) {
        int64_t dim = 0;
        for (int64_t index : index_prefix) {
          output1_ptr[num++] = index;
          dim++;
        }
        dim++;
        output1_ptr[num++] = i;
        for (int64_t index : index_suffix) {
          output1_ptr[num++] = index;
          dim++;
        }
        ++next_index;
      }
    }
  }
}

template <typename T2>
void RaggedTensorToSparseCpuKernelMod::OutPutSparseValues(const std::vector<kernel::AddressPtr> &inputs,
                                                          const std::vector<kernel::AddressPtr> &workspace,
                                                          const std::vector<kernel::AddressPtr> &outputs) {
  int64_t input2_value_num = 0;
  auto output2_ptr = reinterpret_cast<T2 *>(outputs[1]->addr);
  auto input2_ptr = reinterpret_cast<T2 *>(inputs[n_]->addr);

  input2_value_num = inputs[n_]->size / sizeof(T2);
  for (int64_t i = 0; i < input2_value_num; i++) {
    output2_ptr[i] = input2_ptr[i];
  }
}

template <typename T1>
void RaggedTensorToSparseCpuKernelMod::OutPutSparseDenseShape(const std::vector<std::vector<T1>> &input1,
                                                              int64_t *output3_ptr) {
  output3_ptr[0] = input1[0].size() - 1;
  for (int64_t dim = 0; dim < n_; ++dim) {
    const auto &splits = input1[dim];
    int64_t max_width = 0;
    int64_t splits_i = splits.size();
    for (int64_t i = 1; i < splits_i; ++i) {
      int64_t splits_i_1 = splits[i] - splits[i - 1];
      max_width = std::max(max_width, splits_i_1);
    }
    output3_ptr[dim + 1] = max_width;
  }
  int64_t input2_shape = input2_shape_.size();
  for (int64_t dim = 1; dim < input2_shape; ++dim) {
    output3_ptr[dim + n_] = input2_shape_[dim];
  }
}

template <typename T1, typename T2>
bool RaggedTensorToSparseCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                    const std::vector<kernel::AddressPtr> &workspace,
                                                    const std::vector<kernel::AddressPtr> &outputs) {
  auto *output1_ptr = static_cast<int64_t *>(outputs[0]->addr);
  auto *output3_ptr = static_cast<int64_t *>(outputs[2]->addr);
  auto input_num = inputs.size();
  n_ = input_num - 1;
  if (n_ <= 0) {
    MS_LOG(EXCEPTION) << "For op " << kernel_name_
                      << ", the length of rt_nested_splits should be bigger than 0, but got " << n_ << ".";
  }
  std::vector<std::vector<T1>> input1(n_);
  for (int64_t i = 0; i < n_; ++i) {
    auto input1_ptr = reinterpret_cast<T1 *>(inputs[kRttsInputStart + i]->addr);
    int64_t inputs_1 = inputs[kRttsInputStart + i]->size / sizeof(T1);
    for (int64_t j = 0; j < inputs_1; j++) {
      input1[i].push_back(*(input1_ptr + j));
    }
  }

  ValidateInputs<T1>(input1);

  int64_t inp2_s = input2_shape_[0];
  int64_t inp1_b = input1.back()[input1.back().size() - 1];
  if (inp2_s != inp1_b) {
    MS_LOG(EXCEPTION) << "For op " << kernel_name_ << ", final value of ragged splits must equal with the length "
                      << inp2_s << ", but got " << inp1_b << ".";
  }

  std::vector<int64_t> index_prefix(n_);
  std::vector<std::vector<int64_t>> index_suffixes;

  std::vector<std::vector<int64_t>> suffixes{{}};
  int64_t input2_sizes = input2_shape_.size();
  for (int64_t dim = 1; dim < input2_sizes; ++dim) {
    std::vector<std::vector<int64_t>> new_suffixes;
    for (const auto &suffix : suffixes) {
      int64_t input2_shapes = input2_shape_[dim];
      for (int64_t i = 0; i < input2_shapes; ++i) {
        new_suffixes.push_back(suffix);
        new_suffixes.back().push_back(i);
      }
    }
    suffixes.swap(new_suffixes);
  }
  index_suffixes = suffixes;

  // Allocate the `sparse_indices` output tensor.
  Update<T1>(input1, output1_ptr, index_suffixes, index_prefix);

  // Output the `sparse_values` Tensor.
  OutPutSparseValues<T2>(inputs, workspace, outputs);

  // Output the `sparse_dense_shape` Tensor.
  OutPutSparseDenseShape<T1>(input1, output3_ptr);

  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RaggedTensorToSparse, RaggedTensorToSparseCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
