/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/fractional_max_pool_grad_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 5;
constexpr size_t kOutputsNum = 1;
constexpr int kInvalidMaxPoolingIndex = -1;
constexpr size_t tensor_in_and_out_dims = 4;
constexpr size_t kInputShapeIndexN = 0;
constexpr size_t kInputShapeIndexH = 1;
constexpr size_t kInputShapeIndexW = 2;
constexpr size_t kInputShapeIndexC = 3;
constexpr size_t kOutputShapeIndexN = 0;
constexpr size_t kOutputShapeIndexH = 1;
constexpr size_t kOutputShapeIndexW = 2;
constexpr size_t kOutputShapeIndexC = 3;
constexpr size_t kInputIndex0 = 0;
constexpr size_t kInputIndex1 = 1;
constexpr size_t kInputIndex2 = 2;
constexpr size_t kInputIndex3 = 3;
constexpr size_t kInputIndex4 = 4;
}  // namespace

bool FractionalMaxPoolGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  constexpr size_t input_num = kInputsNum;
  constexpr size_t output_num = kOutputsNum;
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', does not support this kernel data type: " << kernel_attr;
    return false;
  }
  auto kernel_ptr = std::dynamic_pointer_cast<ops::FractionalMaxPoolGrad>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  overlapping_ = kernel_ptr->get_overlapping();
  kernel_func_ = func_list_[index].second;
  return true;
}

int FractionalMaxPoolGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs,
                                              const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  tensor_in_shape_ = inputs[kInputIndex0]->GetDeviceShapeAdaptively();
  tensor_out_shape_ = inputs[kInputIndex1]->GetDeviceShapeAdaptively();
  return ret;
}

template <typename T>
bool FractionalMaxPoolGradCpuKernelMod::FractionalMaxPoolGradLaunch(const std::vector<AddressPtr> &inputs,
                                                                    const std::vector<AddressPtr> &outputs) {
  typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> ConstEigenMatrixMap;
  typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> EigenMatrixMap;
  typedef Eigen::Map<Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>> EigenIndexMatrixMap;
  T *tensor_in = reinterpret_cast<T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(tensor_in);
  T *tensor_out = reinterpret_cast<T *>(inputs[1]->addr);
  MS_EXCEPTION_IF_NULL(tensor_out);
  T *out_backprop = reinterpret_cast<T *>(inputs[2]->addr);
  MS_EXCEPTION_IF_NULL(out_backprop);
  int64_t *row_seq = reinterpret_cast<int64_t *>(inputs[3]->addr);
  MS_EXCEPTION_IF_NULL(row_seq);
  int64_t *col_seq = reinterpret_cast<int64_t *>(inputs[4]->addr);
  MS_EXCEPTION_IF_NULL(col_seq);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(output);
  size_t tensor_in_num = inputs[0]->size / sizeof(T);
  size_t tensor_out_num = inputs[1]->size / sizeof(T);
  size_t back_in_nums = inputs[kInputIndex2]->size / sizeof(T);
  size_t row_seq_num = inputs[kInputIndex3]->size / sizeof(int64_t);
  size_t col_seq_num = inputs[kInputIndex4]->size / sizeof(int64_t);
  size_t output_nums = outputs[0]->size / sizeof(T);
  std::vector<T> tensor_output(tensor_out_num);
  std::vector<int64_t> tensor_out_index(tensor_out_num);
  for (size_t i = 0; i < tensor_out_num; i++) {
    tensor_output[i] = std::numeric_limits<T>::lowest();
    tensor_out_index[i] = -1;
  }
  // Find arg_max for each tensor_out
  ConstEigenMatrixMap tensor_in_mat(
    reinterpret_cast<T *>(tensor_in), tensor_in_shape_[kInputShapeIndexC],
    tensor_in_shape_[kInputShapeIndexW] * tensor_in_shape_[kInputShapeIndexH] * tensor_in_shape_[kInputShapeIndexN]);
  EigenMatrixMap output_mat(tensor_output.data(), tensor_out_shape_[kOutputShapeIndexC],
                            tensor_out_shape_[kOutputShapeIndexW] * tensor_out_shape_[kOutputShapeIndexH] *
                              tensor_out_shape_[kOutputShapeIndexN]);
  EigenIndexMatrixMap output_index_mat(tensor_out_index.data(), tensor_out_shape_[kOutputShapeIndexC],
                                       tensor_out_shape_[kOutputShapeIndexW] * tensor_out_shape_[kOutputShapeIndexH] *
                                         tensor_out_shape_[kOutputShapeIndexN]);
  /**
   * Now walk through the process of fractional max pooling again.
   * For both input and output,
   * 0: batch
   * 1: height / row
   * 2: width / col
   * 3: depth / channel
   */
  const int64_t height_max = tensor_in_shape_[kInputShapeIndexH] - 1;
  auto task = [this, row_seq_num, row_seq, col_seq_num, col_seq, height_max, tensor_in_mat, output_mat,
               output_index_mat](size_t start, size_t end) {
    for (size_t b = start; b < end; ++b) {
      // height sequence.
      for (size_t hs = 0; hs < row_seq_num - 1; ++hs) {
        // height start and end.
        const int64_t row_start = *(row_seq + hs);
        int64_t row_end = overlapping_ ? *(row_seq + hs + 1) : *(row_seq + hs + 1) - 1;
        row_end = std::min(row_end, height_max);
        // width sequence.
        FractionalMaxPoolGradCompute(row_start, row_end, col_seq_num, b, hs, col_seq, tensor_in_mat, output_mat,
                                     output_index_mat);
      }
    }
  };
  CPUKernelUtils::ParallelFor(task, LongToSize(tensor_in_shape_[kInputShapeIndexN]));
  FractionalMaxPoolGradOutput(tensor_in_num, output, out_backprop, back_in_nums, output_nums, tensor_out_index);
  return true;
}

template <typename T>
void FractionalMaxPoolGradCpuKernelMod::FractionalMaxPoolGradCompute(
  const int64_t row_start, int64_t row_end, size_t col_seq_num, size_t b, size_t hs, const int64_t *col_seq,
  const Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> tensor_in_mat,
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> output_mat,
  Eigen::Map<Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>> output_index_mat) {
  // width sequence.
  const int64_t width_max = tensor_in_shape_[kInputShapeIndexW] - 1;
  for (size_t ws = 0; ws < col_seq_num - 1; ++ws) {
    const int64_t output_index =
      (SizeToLong(b) * tensor_out_shape_[kOutputShapeIndexH] + SizeToLong(hs)) * tensor_out_shape_[kOutputShapeIndexW] +
      SizeToLong(ws);
    // width start and end.
    const int64_t col_start = *(col_seq + ws);
    int64_t col_end = overlapping_ ? *(col_seq + ws + 1) : *(col_seq + ws + 1) - 1;
    col_end = std::min(col_end, width_max);
    for (int64_t h = row_start; h <= row_end; ++h) {
      for (int64_t w = col_start; w <= col_end; ++w) {
        const int64_t input_index =
          SizeToLong(b) * tensor_in_shape_[kInputShapeIndexH] + h * tensor_in_shape_[kInputShapeIndexW] + w;
        // Walk through each channel (depth).
        for (size_t d = 0; d < LongToSize(tensor_in_shape_[kInputShapeIndexC]); ++d) {
          const T &input_ref = tensor_in_mat.coeffRef(SizeToLong(d), input_index);
          T &output_ref = output_mat.coeffRef(SizeToLong(d), output_index);
          int64_t &output_index_ref = output_index_mat.coeffRef(SizeToLong(d), output_index);
          if (output_ref < input_ref || output_index_ref == kInvalidMaxPoolingIndex) {
            output_ref = input_ref;
            int64_t input_offset = input_index * tensor_in_shape_[kInputShapeIndexC] + SizeToLong(d);
            output_index_ref = input_offset;
          }
        }
      }
    }
  }
}

template <typename T>
void FractionalMaxPoolGradCpuKernelMod::FractionalMaxPoolGradOutput(size_t tensor_in_num, T *output,
                                                                    const T *out_backprop, size_t back_in_nums,
                                                                    size_t output_nums,
                                                                    std::vector<int64_t> tensor_out_index) {
  for (size_t i = 0; i < tensor_in_num; i++) {
    *(output + i) = 0;
  }
  for (size_t index = 0; index < back_in_nums; ++index) {
    int64_t input_index = tensor_out_index[index];
    if (input_index < 0 || input_index >= SizeToLong(output_nums))
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', invalid input 'out_backprop' index:[" << input_index
                               << "], the maximum number of output is: [" << output_nums << "].";
    *(output + input_index) += *(out_backprop + index);
  }
}

std::vector<std::pair<KernelAttr, FractionalMaxPoolGradCpuKernelMod::FractionalMaxPoolGradFunc>>
  FractionalMaxPoolGradCpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &FractionalMaxPoolGradCpuKernelMod::FractionalMaxPoolGradLaunch<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     &FractionalMaxPoolGradCpuKernelMod::FractionalMaxPoolGradLaunch<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &FractionalMaxPoolGradCpuKernelMod::FractionalMaxPoolGradLaunch<int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32),
     &FractionalMaxPoolGradCpuKernelMod::FractionalMaxPoolGradLaunch<int32_t>}};

std::vector<KernelAttr> FractionalMaxPoolGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, FractionalMaxPoolGradFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, FractionalMaxPoolGrad, FractionalMaxPoolGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
