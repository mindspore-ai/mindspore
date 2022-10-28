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

#include "plugin/device/cpu/kernel/fractional_avg_pool_grad_cpu_kernel.h"
#include <map>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 4;
constexpr size_t kOutputsNum = 1;
constexpr size_t tensor_in_and_out_dims = 4;
constexpr size_t kShapeIndexN = 0;
constexpr size_t kShapeIndexH = 1;
constexpr size_t kShapeIndexW = 2;
constexpr size_t kShapeIndexC = 3;
}  // namespace

bool FractionalAvgPoolGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  overlapping_ = GetValue<bool>(base_operator->GetAttr("overlapping"));
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', does not support this kernel data type: " << kernel_attr;
  }

  kernel_func_ = func_list_[index].second;
  return true;
}

int FractionalAvgPoolGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs,
                                              const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  orig_input_shape_ = inputs[0]->GetDeviceShapeAdaptively();
  out_backprop_shape_ = inputs[1]->GetDeviceShapeAdaptively();
  return KRET_OK;
}

template <typename T>
bool FractionalAvgPoolGradCpuKernelMod::FractionalAvgPoolGradLaunch(const std::vector<AddressPtr> &inputs,
                                                                    const std::vector<AddressPtr> &outputs) {
  typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> ConstEigenMatrixMap;
  typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> EigenDoubleMatrixMap;
  int64_t *orig_input_tensor_shape = reinterpret_cast<int64_t *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(orig_input_tensor_shape);
  T *out_backprop = reinterpret_cast<T *>(inputs[1]->addr);
  MS_EXCEPTION_IF_NULL(out_backprop);
  int64_t *row_seq = reinterpret_cast<int64_t *>(inputs[2]->addr);
  MS_EXCEPTION_IF_NULL(row_seq);
  int64_t *col_seq = reinterpret_cast<int64_t *>(inputs[3]->addr);
  MS_EXCEPTION_IF_NULL(col_seq);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(output);
  size_t orig_input_shape_num = inputs[0]->size / sizeof(int64_t);
  if (orig_input_shape_.size() != 1 || orig_input_shape_num != tensor_in_and_out_dims) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the input 'orig_input_tensor_shape' must be 1-dimensional and 4 elements.";
  }
  const int64_t out_batch = out_backprop_shape_[kShapeIndexN];
  const int64_t out_rows = out_backprop_shape_[kShapeIndexH];
  const int64_t out_cols = out_backprop_shape_[kShapeIndexW];
  const int64_t out_depth = out_backprop_shape_[kShapeIndexC];
  int64_t row_seq_nums = SizeToLong(inputs[2]->size / sizeof(int64_t));
  int64_t col_seq_nums = SizeToLong(inputs[3]->size / sizeof(int64_t));
  if (row_seq_nums <= out_rows) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', given 'out_backprop' shape [" << out_batch << ","
                             << out_rows << "," << out_cols << "," << out_depth
                             << "], 'row_pooling_sequence' must have at least [" << (out_rows + 1)
                             << "elements, but got[" << row_seq_nums << ".";
  }
  if (col_seq_nums <= out_cols) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', given 'out_backprop' shape [" << out_batch << ","
                             << out_rows << "," << out_cols << "," << out_depth
                             << "], 'col_pooling_sequence' must have at least [" << (out_cols + 1)
                             << "elements, but got[" << col_seq_nums << ".";
  }
  const int64_t in_batch = *(orig_input_tensor_shape);
  const int64_t in_rows = *(orig_input_tensor_shape + kShapeIndexH);
  const int64_t in_cols = *(orig_input_tensor_shape + kShapeIndexW);
  const int64_t in_depth = *(orig_input_tensor_shape + kShapeIndexC);

  out_shape_.clear();
  for (size_t i = 0; i < orig_input_shape_num; i++) {
    if (orig_input_tensor_shape[i] <= 0) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                               << "', each dimension of input 'orig_input_tensor_shape' must be greater than 0.";
    }
    out_shape_.push_back(orig_input_tensor_shape[i]);
  }

  size_t output_nums = LongToSize(in_batch * in_rows * in_cols * in_depth);
  std::vector<double> in_backprop_tensor_temp(output_nums);
  for (size_t i = 0; i < output_nums; i++) {
    in_backprop_tensor_temp[i] = 0;
    *(output + i) = 0;
  }
  EigenDoubleMatrixMap in_backprop_tensor_temp_mat(in_backprop_tensor_temp.data(), in_depth,
                                                   in_cols * in_rows * in_batch);
  ConstEigenMatrixMap out_backprop_mat(reinterpret_cast<T *>(out_backprop), out_depth, out_cols * out_rows * out_batch);
  // Loop through each element of out_backprop and evenly distribute theelement to the corresponding pooling cell.
  const int64_t height_max = in_rows - 1;
  auto sharder_fractional_avg_pool_grad = [this, out_rows, row_seq, out_cols, col_seq, height_max, out_depth, in_rows,
                                           in_cols, out_backprop_mat,
                                           in_backprop_tensor_temp_mat](size_t start, size_t end) {
    for (size_t b = start; b < end; ++b) {
      for (int64_t hs = 0; hs < out_rows; ++hs) {
        const int64_t height_start = *(row_seq + hs);
        int64_t height_end = overlapping_ ? *(row_seq + hs + 1) : *(row_seq + hs + 1) - 1;
        height_end = std::min(height_end, height_max);
        FractionalAvgPoolGradCompute(out_cols, col_seq, height_start, height_end, SizeToLong(b), hs, out_rows,
                                     out_depth, in_rows, in_cols, out_backprop_mat, in_backprop_tensor_temp_mat);
      }
    }
  };
  CPUKernelUtils::ParallelFor(sharder_fractional_avg_pool_grad, LongToSize(out_batch));
  // Depending on the type, cast double to type T.
  for (size_t i = 0; i < output_nums; ++i) {
    *(output + i) = static_cast<T>(in_backprop_tensor_temp[i]);
  }
  return true;
}

void FractionalAvgPoolGradCpuKernelMod::SyncData() { outputs_[kIndex0]->SetShapeVector(out_shape_); }

template <typename T>
void FractionalAvgPoolGradCpuKernelMod::FractionalAvgPoolGradCompute(
  const int64_t out_cols, const int64_t *col_seq, const int64_t height_start, int64_t height_end, int64_t b, int64_t hs,
  const int64_t out_rows, const int64_t out_depth, const int64_t in_rows, const int64_t in_cols,
  const Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &out_backprop_mat,
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> in_backprop_tensor_temp_mat) {
  const int64_t width_max = in_cols - 1;
  for (int64_t ws = 0; ws < out_cols; ++ws) {
    const int64_t width_start = *(col_seq + ws);
    int64_t width_end = overlapping_ ? *(col_seq + ws + 1) : *(col_seq + ws + 1) - 1;
    width_end = std::min(width_end, width_max);
    const int64_t num_elements_in_pooling_cell = (height_end - height_start + 1) * (width_end - width_start + 1);
    if (num_elements_in_pooling_cell == 0) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                               << "', input 'orig_input_tensor_shape' error, please check it.";
    }
    const int64_t out_index = (b * out_rows + hs) * out_cols + ws;
    // Now we can evenly distribute out_backprop(b, h, w, *) to in_backprop(b, hs:he, ws:we, *).
    for (int64_t h = height_start; h <= height_end; ++h) {
      for (int64_t w = width_start; w <= width_end; ++w) {
        const int64_t in_index = (b * in_rows + h) * in_cols + w;
        // Walk through each channel (depth).
        for (int64_t d = 0; d < out_depth; ++d) {
          const double out_backprop_element = static_cast<double>(out_backprop_mat.coeffRef(d, out_index));
          double &in_backprop_ref = in_backprop_tensor_temp_mat.coeffRef(d, in_index);
          in_backprop_ref += out_backprop_element / num_elements_in_pooling_cell;
        }
      }
    }
  }
}

std::vector<std::pair<KernelAttr, FractionalAvgPoolGradCpuKernelMod::FractionalAvgPoolGradFunc>>
  FractionalAvgPoolGradCpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &FractionalAvgPoolGradCpuKernelMod::FractionalAvgPoolGradLaunch<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     &FractionalAvgPoolGradCpuKernelMod::FractionalAvgPoolGradLaunch<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &FractionalAvgPoolGradCpuKernelMod::FractionalAvgPoolGradLaunch<int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32),
     &FractionalAvgPoolGradCpuKernelMod::FractionalAvgPoolGradLaunch<int32_t>}};

std::vector<KernelAttr> FractionalAvgPoolGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, FractionalAvgPoolGradFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, FractionalAvgPoolGrad, FractionalAvgPoolGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
