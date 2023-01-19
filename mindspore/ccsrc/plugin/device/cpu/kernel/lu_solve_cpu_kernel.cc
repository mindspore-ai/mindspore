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
#include "plugin/device/cpu/kernel/lu_solve_cpu_kernel.h"
#include <utility>
#include <algorithm>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kDimNum = 2;
}

int64_t get_element_num(const std::vector<int64_t> &shape) { return SizeToLong(SizeOf(shape)); }

bool LuSolveCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();
  size_t input_num = inputs.size();
  size_t output_num = outputs.size();
  CHECK_KERNEL_INPUTS_NUM(input_num, kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kOutputNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LuSolveFunc> &pair) { return pair.first; });
  auto [is_match, index] = MatchKernelAttr(kernel_attr, support_list);
  if (!is_match) {
    MS_LOG(ERROR) << "LuSolve does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int LuSolveCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  input_0_shape_ = inputs[kIndex0]->GetDeviceShapeAdaptively();
  input_1_shape_ = inputs[kIndex1]->GetDeviceShapeAdaptively();
  output_shape_ = outputs[kIndex0]->GetDeviceShapeAdaptively();
  return KRET_OK;
}

template <typename T1, typename T2>
void LuSolveCpuKernelMod::LuSolve(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &outputs, T1 *b_working_ptr, T1 *lu_working_ptr,
                                  int32_t *pivots_working_ptr, size_t b_stride, size_t a) {
  auto output_y = reinterpret_cast<T2 *>(outputs[0]->addr);
  size_t lu_dims = input_1_shape_.size();
  size_t lu_maxtrix_sizes = LongToSize(input_1_shape_[lu_dims - 2]);
  size_t b_dim = input_0_shape_.size();
  size_t b_m = LongToSize(input_0_shape_[b_dim - 1]);
  typedef Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXd;
  MatrixXd matrix_b = Eigen::Map<MatrixXd>(b_working_ptr, lu_maxtrix_sizes, b_m);
  MatrixXd matrix_A = Eigen::Map<MatrixXd>(lu_working_ptr, lu_maxtrix_sizes, lu_maxtrix_sizes);
  for (size_t i = 0; i < LongToSize(input_0_shape_[b_dim - kDimNum]); i++) {
    size_t pivots_i = *(pivots_working_ptr + i) - 1;
    if (pivots_i > LongToSize(input_0_shape_[b_dim - kDimNum])) {
      MS_EXCEPTION(ValueError) << "lu_pivots values out of index of lu_data. ";
    }
    matrix_b.row(i).swap(matrix_b.row(pivots_i));
  }
  MatrixXd result = matrix_A.template triangularView<Eigen::UnitLower>().solve(matrix_b);
  result.noalias() = matrix_A.template triangularView<Eigen::Upper>().solve(result);
  for (size_t m = 0; m < b_stride; m++) {
    *(output_y + a * b_stride + m) = (T2) * (result.data() + m);
  }
}

template <typename T1, typename T2>
bool LuSolveCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  auto input_x0 = reinterpret_cast<T2 *>(inputs[0]->addr);
  auto input_x1 = reinterpret_cast<T2 *>(inputs[1]->addr);
  auto input_x2 = reinterpret_cast<int32_t *>(inputs[2]->addr);
  auto input0_element_num = SizeOf(input_0_shape_);
  auto input1_element_num = SizeOf(input_1_shape_);
  auto output_element_num = SizeOf(output_shape_);
  std::vector<T1> input_0(input_x0, input_x0 + input0_element_num);
  std::vector<T1> input_1(input_x1, input_x1 + input1_element_num);
  size_t b_dims = input_0_shape_.size();
  std::vector<int64_t> b_dims_vector = input_0_shape_;
  size_t lu_dims = input_1_shape_.size();
  std::vector<int64_t> lu_dims_vector = input_1_shape_;
  size_t b_stride = static_cast<size_t>(input_0_shape_[b_dims - 1] * input_0_shape_[b_dims - 2]);
  size_t lu_stride = static_cast<size_t>(input_1_shape_[lu_dims - 1] * input_1_shape_[lu_dims - 2]);
  size_t pivots_stride = static_cast<size_t>(input_1_shape_[lu_dims - 1]);
  MS_EXCEPTION_IF_ZERO("b_stride", b_stride);
  size_t batch_num = output_element_num / b_stride;
  if (b_dims == lu_dims) {
    for (size_t i = 0; i < batch_num; i++) {
      T1 *b_working_ptr = input_0.data() + i * b_stride;
      T1 *lu_working_ptr = input_1.data() + i * lu_stride;
      int32_t *pivots_working_ptr = &input_x2[i * pivots_stride];
      LuSolve<T1, T2>(inputs, outputs, b_working_ptr, lu_working_ptr, pivots_working_ptr, b_stride, i);
    }
  } else {
    std::vector<int64_t> b_shape = b_dims_vector;
    std::vector<int64_t> lu_shape = lu_dims_vector;
    for (size_t i = 0; i < kDimNum; i++) {
      b_shape.pop_back();
      lu_shape.pop_back();
    }
    auto output_shape = CPUKernelUtils::GetBroadcastShape(b_shape, lu_shape);
    BroadcastIterator iter(b_shape, lu_shape, output_shape);
    iter.SetPos(0);
    for (size_t i = 0; i < batch_num; i++) {
      T1 *b_working_ptr = input_0.data() + iter.GetInputPosA() * b_stride;
      T1 *lu_working_ptr = input_1.data() + iter.GetInputPosB() * lu_stride;
      int32_t *pivots_working_ptr = &input_x2[iter.GetInputPosB() * pivots_stride];
      LuSolve<T1, T2>(inputs, outputs, b_working_ptr, lu_working_ptr, pivots_working_ptr, b_stride, i);
      iter.GenNextPos();
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, LuSolveCpuKernelMod::LuSolveFunc>> LuSolveCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat16),
   &LuSolveCpuKernelMod::LaunchKernel<float, float16>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LuSolveCpuKernelMod::LaunchKernel<float, float>}};

std::vector<KernelAttr> LuSolveCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LuSolveFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LuSolve, LuSolveCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
