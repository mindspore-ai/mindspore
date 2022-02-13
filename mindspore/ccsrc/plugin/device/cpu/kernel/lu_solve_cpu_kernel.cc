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
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kDimNum = 2;
}

size_t get_element_num(const std::vector<size_t> &shape) {
  size_t size = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }
  return size;
}

template <typename T1, typename T2>
void LuSolveCpuKernelMod<T1, T2>::InitKernel(const CNodePtr &kernel_node) {
  node_wpt_ = kernel_node;
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kOutputNum, kernel_name_);
  auto x_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  auto lu_data_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  auto lu_pivots_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 2);
  if (lu_data_shape.size() < kDimNum) {
    MS_EXCEPTION(ValueError) << "For LuSolveCPUKercel lu_data's dimensions should be greater than or equal to 2.";
  }
  if (x_shape.size() < kDimNum) {
    MS_EXCEPTION(ValueError) << "For LuSolveCPUKercel x's dimensions should be greater than or equal to 2.";
  }
  if (lu_pivots_shape.size() < 1) {
    MS_EXCEPTION(ValueError) << "For LuSolveCPUKercel lu_pivots's dimensions should be greater than or equal to 1.";
  }
  if (lu_data_shape[lu_data_shape.size() - 1] != lu_data_shape[lu_data_shape.size() - kDimNum])
    MS_EXCEPTION(ValueError) << "For LuSolveCPUKercel "
                             << " input lu_data should be square matrix "
                             << "while row is " << lu_data_shape[lu_data_shape.size() - kDimNum] << ", col is "
                             << lu_data_shape[lu_data_shape.size() - 1] << ".";

  if (x_shape.size() == lu_data_shape.size()) {
    for (size_t i = 0; i <= x_shape.size() - kDimNum; i++) {
      if (x_shape[i] != lu_data_shape[i]) {
        MS_EXCEPTION(ValueError) << "For LuSolveCPUKercel "
                                 << " shapes in dim[" << i << "] are not the same "
                                 << "while x is " << x_shape[i] << ", lu_data is " << lu_data_shape[i] << ".";
      }
    }
  } else if (lu_data_shape.size() > x_shape.size()) {
    for (size_t i = 0; i < x_shape.size() - kDimNum; i++) {
      if (x_shape[i] != lu_data_shape[lu_data_shape.size() - x_shape.size() + i]) {
        MS_EXCEPTION(ValueError) << "For LuSolveCPUKercel"
                                 << " shapes in dim[" << i << "] are not same as lu_data's dim["
                                 << lu_data_shape.size() - x_shape.size() + i << "]"
                                 << "while x is " << x_shape[i] << ", lu_data is " << lu_data_shape[i] << ".";
      }
    }
  } else {
    for (size_t i = 0; i < lu_data_shape.size() - kDimNum; i++) {
      if (lu_data_shape[i] != x_shape[x_shape.size() - lu_data_shape.size() + i]) {
        MS_EXCEPTION(ValueError) << "For LuSolveCPUKercel "
                                 << " shapes in lu_data's dim[" << i << "] are not same as x's dim["
                                 << x_shape.size() - lu_data_shape.size() + i << "]"
                                 << "while x is " << x_shape[x_shape.size() - lu_data_shape.size() + i]
                                 << ", lu_data is " << lu_data_shape[i] << ".";
      }
    }
  }
  if (lu_pivots_shape[lu_pivots_shape.size() - 1] != lu_data_shape[lu_data_shape.size() - 1]) {
    MS_EXCEPTION(ValueError) << "For LuSolveCPUKercel "
                             << " Number of pivots per batch should be same as the dimension of the matrix.";
  }
  for (size_t i = 0; i < lu_pivots_shape.size(); i++) {
    if (lu_data_shape[i] != lu_pivots_shape[i]) {
      MS_EXCEPTION(ValueError) << "For LuSolveCPUKercel "
                               << "batch dimension of LU_pivots should match batch dimension of LU_data.";
    }
  }
}

template <typename T1, typename T2>
void LuSolveCpuKernelMod<T1, T2>::LuSolve(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &outputs, T1 *b_working_ptr,
                                          T1 *lu_working_ptr, int32_t *pivots_working_ptr, size_t b_stride, size_t a) {
  auto input_0_Shape = AnfAlgo::GetInputDeviceShape(node_wpt_, 0);
  auto input_1_Shape = AnfAlgo::GetInputDeviceShape(node_wpt_, 1);
  auto output_y = reinterpret_cast<T2 *>(outputs[0]->addr);
  size_t lu_dims = input_1_Shape.size();
  size_t lu_maxtrix_sizes = input_1_Shape[lu_dims - 2];
  size_t b_dim = input_0_Shape.size();
  size_t b_m = input_0_Shape[b_dim - 1];
  typedef Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXd;
  MatrixXd matrix_b = Eigen::Map<MatrixXd>(b_working_ptr, lu_maxtrix_sizes, b_m);
  MatrixXd matrix_A = Eigen::Map<MatrixXd>(lu_working_ptr, lu_maxtrix_sizes, lu_maxtrix_sizes);
  for (size_t i = 0; i < input_0_Shape[b_dim - kDimNum]; i++) {
    matrix_b.row(i).swap(matrix_b.row(*(pivots_working_ptr + i) - 1));
  }
  MatrixXd L = matrix_A.template triangularView<Eigen::UnitLower>();
  MatrixXd U = matrix_A.template triangularView<Eigen::Upper>();
  MatrixXd x = L * U;
  MatrixXd result = x.lu().solve(matrix_b);
  for (size_t m = 0; m < b_stride; m++) {
    *(output_y + a * b_stride + m) = (T2) * (result.data() + m);
  }
}

template <typename T1, typename T2>
bool LuSolveCpuKernelMod<T1, T2>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  auto input_x0 = reinterpret_cast<T2 *>(inputs[0]->addr);
  auto input_x1 = reinterpret_cast<T2 *>(inputs[1]->addr);
  auto input_x2 = reinterpret_cast<int32_t *>(inputs[2]->addr);
  auto input_0_Shape = AnfAlgo::GetInputDeviceShape(node_wpt_, 0);
  auto input_1_Shape = AnfAlgo::GetInputDeviceShape(node_wpt_, 1);
  auto output_Shape = AnfAlgo::GetOutputDeviceShape(node_wpt_, 0);
  size_t input0_element_num = get_element_num(input_0_Shape);
  size_t input1_element_num = get_element_num(input_1_Shape);
  size_t output_element_num = get_element_num(output_Shape);
  std::vector<T1> input_0(input_x0, input_x0 + input0_element_num);
  std::vector<T1> input_1(input_x1, input_x1 + input1_element_num);
  size_t b_dims = input_0_Shape.size();
  std::vector<size_t> b_dims_vector = input_0_Shape;
  size_t lu_dims = input_1_Shape.size();
  std::vector<size_t> lu_dims_vector = input_1_Shape;
  size_t b_stride = input_0_Shape[b_dims - 1] * input_0_Shape[b_dims - 2];
  size_t lu_stride = input_1_Shape[lu_dims - 1] * input_1_Shape[lu_dims - 2];
  size_t pivots_stride = input_1_Shape[lu_dims - 1];
  MS_EXCEPTION_IF_ZERO("b_stride", b_stride);
  size_t batch_num = output_element_num / b_stride;
  if (b_dims == lu_dims) {
    for (size_t i = 0; i < batch_num; i++) {
      T1 *b_working_ptr = input_0.data() + i * b_stride;
      T1 *lu_working_ptr = input_1.data() + i * lu_stride;
      int32_t *pivots_working_ptr = &input_x2[i * pivots_stride];
      LuSolve(inputs, outputs, b_working_ptr, lu_working_ptr, pivots_working_ptr, b_stride, i);
    }
  } else {
    std::vector<size_t> b_shape = b_dims_vector;
    std::vector<size_t> lu_shape = lu_dims_vector;
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
      LuSolve(inputs, outputs, b_working_ptr, lu_working_ptr, pivots_working_ptr, b_stride, i);
      iter.GenNextPos();
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
