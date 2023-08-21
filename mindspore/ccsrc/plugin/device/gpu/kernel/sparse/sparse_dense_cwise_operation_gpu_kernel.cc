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

#include <functional>
#include <vector>

#include "mindspore/core/abstract/utils.h"
#include "mindspore/core/ops/sparse_ops.h"
#include "mindspore/core/ops/base_operator.h"
#include "mindspore/core/ops/sparse_dense_cwise_add.h"
#include "mindspore/core/ops/sparse_dense_cwise_div.h"
#include "mindspore/core/ops/sparse_dense_cwise_mul.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/sparse/sparse_dense_cwise_operation_gpu_kernel.h"

namespace mindspore {
namespace kernel {
#define SPARSE_DENSE_CWISE_OPERATION_GPU_REGISTER(T_DT, T) \
  KernelAttr()                                             \
    .AddInputAttr(kNumberTypeInt64)                        \
    .AddInputAttr(T_DT)                                    \
    .AddInputAttr(kNumberTypeInt64)                        \
    .AddInputAttr(T_DT)                                    \
    .AddOutputAttr(T_DT),                                  \
    &SparseDenseCwiseOperationGpuKernelMod::LaunchKernel<T>

constexpr int64_t INPUT_DIMS_1 = 1;
constexpr int64_t INPUT_DIMS_2 = 2;

bool SparseDenseCwiseOperationGpuKernelMod::GetOpType(const BaseOperatorPtr &base_operator) {
  static const std::map<std::string, SparseDenseCwiseOperationFunctionType> sparse_dense_cwise_op_map = {
    {prim::kPrimSparseDenseCwiseAdd->name(), SPARSE_DENSE_CWISE_OPERATION_FUNC_ADD},
    {prim::kPrimSparseDenseCwiseMul->name(), SPARSE_DENSE_CWISE_OPERATION_FUNC_MUL},
    {prim::kPrimSparseDenseCwiseDiv->name(), SPARSE_DENSE_CWISE_OPERATION_FUNC_DIV}};
  auto op_type_iter = sparse_dense_cwise_op_map.find(kernel_name_);
  if (op_type_iter == sparse_dense_cwise_op_map.end()) {
    MS_LOG(ERROR) << "Only support these sparse_dense_cwise function: SparseDenseCwiseAdd, SparseDenseCwiseMul, "
                     "or SparseDenseCwiseDiv currently, but got "
                  << kernel_name_;
    return false;
  }
  op_func_type_ = op_type_iter->second;
  return true;
}

bool SparseDenseCwiseOperationGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                 const std::vector<KernelTensorPtr> &inputs,
                                                 const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (!GetOpType(base_operator)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' it got op type and function type failed.";
    return false;
  }
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

int SparseDenseCwiseOperationGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                  const std::vector<KernelTensorPtr> &inputs,
                                                  const std::vector<KernelTensorPtr> &outputs,
                                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  for (size_t i = 0; i < MAX_DIMS; i++) {
    i_[i] = 1;
    o_[i] = 1;
  }

  std::vector<int64_t> indice_shape = std::vector<int64_t>(inputs[kIndex0]->GetDeviceShapeAdaptively().begin(),
                                                           inputs[kIndex0]->GetDeviceShapeAdaptively().end());
  is_null_input_ = CHECK_SHAPE_NULL(indice_shape, kernel_name_, "input");
  if (!is_null_input_) {
    value_num_ = indice_shape[0];
    dimension_ = indice_shape[1];
    dense_shape_ = std::vector<int64_t>(inputs[kIndex3]->GetDeviceShapeAdaptively().begin(),
                                        inputs[kIndex3]->GetDeviceShapeAdaptively().end());

    dense_num_ =
      std::accumulate(dense_shape_.begin(), dense_shape_.end(), 1,
                      std::multiplies<int64_t>());  // dense_dims_ = static_cast<int64_t>(dense_shape_.size());
  }

  data_unit_size_ = abstract::TypeIdSize(inputs.at(kIndex1)->GetDtype());

  std::vector<int64_t> indices_shape = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                            inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> values_shape = std::vector<int64_t>(inputs.at(kIndex1)->GetDeviceShapeAdaptively().begin(),
                                                           inputs.at(kIndex1)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> shape_shape = std::vector<int64_t>(inputs.at(kIndex2)->GetDeviceShapeAdaptively().begin(),
                                                          inputs.at(kIndex2)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> dense_shape = std::vector<int64_t>(inputs.at(kIndex3)->GetDeviceShapeAdaptively().begin(),
                                                          inputs.at(kIndex3)->GetDeviceShapeAdaptively().end());

  if (indices_shape.size() != INPUT_DIMS_2) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "',  the dim of indices must be 2, but got " << indices_shape.size()
                  << ".";
    return KRET_RESIZE_FAILED;
  }
  if (values_shape.size() != INPUT_DIMS_1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "',  the dim of values must be 1, but got " << values_shape.size()
                  << ".";
    return KRET_RESIZE_FAILED;
  }
  if (shape_shape.size() != INPUT_DIMS_1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "',  the dim of shape must be 1, but got " << shape_shape.size() << ".";
    return KRET_RESIZE_FAILED;
  }
  if (indices_shape[0] != values_shape[0]) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "',  the num of indices  must be equal to the number of value, but got "
                  << indices_shape[0] << " vs " << values_shape[0] << ".";
    return KRET_RESIZE_FAILED;
  }
  if (indices_shape[1] != shape_shape[0]) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "',  shape[1] of `x1_indices` must be equal to shape[0] of `x1_shape`, but got "
                  << indices_shape[1] << " vs " << shape_shape[0] << ".";
    return KRET_RESIZE_FAILED;
  }
  if (dense_shape.size() > size_t(shape_shape[0])) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "',  the dims of `x2` should be less or equal to the shape[0] of `x1_shape`, but got "
                  << dense_shape.size() << " vs " << shape_shape[0] << ".";
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

template <typename T>
bool SparseDenseCwiseOperationGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                         const std::vector<AddressPtr> &workspace,
                                                         const std::vector<AddressPtr> &outputs) {
  if (is_null_input_) {
    return true;
  }
  int64_t *x1_indices = GetDeviceAddress<int64_t>(inputs, kIndex0);
  T *x1_values = GetDeviceAddress<T>(inputs, kIndex1);
  int64_t *x1_shape = GetDeviceAddress<int64_t>(inputs, kIndex2);
  T *x2 = GetDeviceAddress<T>(inputs, kIndex3);
  T *y = GetDeviceAddress<T>(outputs, kIndex0);

  int64_t dense_dims_ = static_cast<int64_t>(dense_shape_.size());
  std::vector<int64_t> x1_shape_host(dimension_);
  std::vector<int64_t> x1_indices_host(value_num_ * dimension_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(x1_shape_host.data(), x1_shape, dimension_ * sizeof(int64_t), cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "For 'SparseDenseCwiseOperation', cudaMemcpy value variable failed.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(x1_indices_host.data(), x1_indices, value_num_ * dimension_ * sizeof(int64_t),
                    cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "For 'SparseDenseCwiseOperation', cudaMemcpy value variable failed.");
  if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(stream_ptr_)) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr_)),
                                       "For 'SparseDenseCwiseOperation', cuda Stream Sync Failed.");
  }

  for (int64_t i = 0; i < value_num_; i++) {
    for (int64_t j = 0; j < dimension_; j++) {
      if (x1_indices_host[static_cast<size_t>(i * dimension_ + j)] >= x1_shape_host[static_cast<size_t>(j)] ||
          x1_indices_host[static_cast<size_t>(i * dimension_ + j)] < 0) {
        MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the indices can't "
                                 << "proceed to cross the border, but got indices[" << static_cast<size_t>(i)
                                 << "] = " << x1_indices_host[static_cast<size_t>(i * dimension_ + j)] << ", while"
                                 << " the border is (0, " << x1_shape_host[static_cast<size_t>(j)] << ").";
        return false;
      }
    }
  }

  std::vector<int64_t> sparse_shape(dimension_);
  for (size_t i = 0; i < static_cast<size_t>(dimension_); i++) {
    sparse_shape[i] = static_cast<int64_t>(x1_shape_host[i]);
  }
  bool isNotNeedBcast_ = (dense_shape_ == sparse_shape || dense_num_ == 1);
  if (isNotNeedBcast_) {
    auto status = CalSparseDenseCwiseOperationNoBcastCompute(op_func_type_, x1_indices, x1_values, x1_shape, x2, y,
                                                             dimension_, value_num_, dense_dims_, device_id_,
                                                             reinterpret_cast<cudaStream_t>(stream_ptr_));
    CHECK_CUDA_STATUS(status, kernel_name_);
  } else if (dense_dims_ <= dimension_) {
    size_t offset_x = sparse_shape.size() - dense_shape_.size();
    for (size_t i = 0; i < dense_shape_.size(); ++i) {
      i_[i + offset_x] = dense_shape_[i];
    }
    for (size_t j = 0; j < sparse_shape.size(); ++j) {
      o_[j] = sparse_shape[j];
    }

    for (int64_t i = dense_dims_ - 1; i >= 0; --i) {
      if ((dense_shape_[static_cast<size_t>(i)] != 1) &&
          (dense_shape_[static_cast<size_t>(i)] != sparse_shape[static_cast<size_t>(i + dimension_ - dense_dims_)])) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape of 'x2' can't broadcast to 'x1_shape'."
                      << "In order to broadcast, the size of the trailing axes for 'x2' and"
                      << "sparse in an operation must either be the same size or size of the"
                      << "trailing axes for 'x2' must be one";
        return false;
      }
    }

    auto status = CalSparseDenseCwiseOperationBcastCompute(op_func_type_, x1_indices, x1_values, x1_shape, x2, y, i_,
                                                           o_, dimension_, value_num_, device_id_,
                                                           reinterpret_cast<cudaStream_t>(stream_ptr_));
    CHECK_CUDA_STATUS(status, kernel_name_);
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', dims of 'x2' should be smaller or equal to Number of"
                  << "elements of 'x1_shape'. Because broadcast direction can only be from dense to sparse."
                  << "but got dims of dense:" << dense_dims_ << "dims of sparse:" << dimension_ << ".";
    return false;
  }

  return true;
}

const SparseDenseCwiseOperationGpuKernelMod::SupportList &SparseDenseCwiseOperationGpuKernelMod::GetFuncList() const {
  static const SparseDenseCwiseOperationGpuKernelMod::SupportList func_list = {
    {SPARSE_DENSE_CWISE_OPERATION_GPU_REGISTER(kNumberTypeInt8, int8_t)},
    {SPARSE_DENSE_CWISE_OPERATION_GPU_REGISTER(kNumberTypeInt16, int16_t)},
    {SPARSE_DENSE_CWISE_OPERATION_GPU_REGISTER(kNumberTypeInt32, int32_t)},
    {SPARSE_DENSE_CWISE_OPERATION_GPU_REGISTER(kNumberTypeInt64, int64_t)},
    {SPARSE_DENSE_CWISE_OPERATION_GPU_REGISTER(kNumberTypeUInt8, uint8_t)},
    {SPARSE_DENSE_CWISE_OPERATION_GPU_REGISTER(kNumberTypeUInt16, uint16_t)},
    {SPARSE_DENSE_CWISE_OPERATION_GPU_REGISTER(kNumberTypeUInt32, uint32_t)},
    {SPARSE_DENSE_CWISE_OPERATION_GPU_REGISTER(kNumberTypeUInt64, uint64_t)},
    {SPARSE_DENSE_CWISE_OPERATION_GPU_REGISTER(kNumberTypeFloat16, half)},
    {SPARSE_DENSE_CWISE_OPERATION_GPU_REGISTER(kNumberTypeFloat32, float)},
    {SPARSE_DENSE_CWISE_OPERATION_GPU_REGISTER(kNumberTypeFloat64, double)},
    {SPARSE_DENSE_CWISE_OPERATION_GPU_REGISTER(kNumberTypeComplex64, utils::Complex<float>)},
    {SPARSE_DENSE_CWISE_OPERATION_GPU_REGISTER(kNumberTypeComplex128, utils::Complex<double>)}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseDenseCwiseAdd, SparseDenseCwiseOperationGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseDenseCwiseMul, SparseDenseCwiseOperationGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseDenseCwiseDiv, SparseDenseCwiseOperationGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
