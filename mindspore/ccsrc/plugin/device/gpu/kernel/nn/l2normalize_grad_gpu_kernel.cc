/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/nn/l2normalize_grad_gpu_kernel.h"
#include "mindspore/core/ops/grad/l2_normalize_grad.h"

namespace mindspore {
namespace kernel {
bool L2NormalizeGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  constexpr size_t input_num = 3;
  constexpr size_t output_num = 1;
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);

  InitResource();
  data_type_ = GetCudnnDataType(TypeIdLabel(inputs.at(kIndex0)->GetDtype()));

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;

  auto l2_normalize_grad_ptr = std::dynamic_pointer_cast<ops::L2NormalizeGrad>(base_operator);
  if (l2_normalize_grad_ptr == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', cast 'L2NormalizeGrad' ops failed!";
    return false;
  }
  epsilon_ = l2_normalize_grad_ptr->get_epsilon();
  axis_origin_ = LongToInt(l2_normalize_grad_ptr->get_axis());
  return true;
}

int L2NormalizeGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  auto input_shape = inputs[0]->GetShapeVector();
  int input_dim_length = SizeToInt(input_shape.size());
  axis_ = axis_origin_ < 0 ? (axis_origin_ + input_dim_length) : axis_origin_;

  input_shape_list_.clear();
  for (size_t i = 0; i < inputs.size(); i++) {
    (void)input_shape_list_.emplace_back(inputs[i]->GetShapeVector());
  }

  auto output_shape = outputs[0]->GetShapeVector();
  if (!CheckInputShape(output_shape)) {
    return KRET_RESIZE_FAILED;
  }

  ShapeVector output_reduce_shape = output_shape;
  if ((size_t)axis_ >= output_shape.size()) {
    MS_LOG(ERROR) << "For 'L2NormalizeGradGpuKernelMod', axis_ must be less than the rank of output "
                  << "but got axis_: " << axis_ << ", rank of output: " << output_shape.size();
    return KRET_RESIZE_FAILED;
  }
  output_reduce_shape[axis_] = 1;

  lhs_shape_.resize(MAX_DIMS, 1);
  rhs_shape_.resize(MAX_DIMS, 1);
  output_shape_.resize(MAX_DIMS, 1);
  all_match_ = true;
  for (size_t i = 0; i < output_shape.size(); i++) {
    output_shape_[i] = LongToSizeClipNeg(output_shape[i]);
    lhs_shape_[i] = LongToSizeClipNeg(output_shape[i]);
    rhs_shape_[i] = LongToSizeClipNeg(output_reduce_shape[i]);
    if (lhs_shape_[i] != rhs_shape_[i]) {
      all_match_ = false;
    }
  }

  InferInAndOutDesc(input_shape_list_[0], output_reduce_shape);
  InferArrayReduceType();
  InitWorkSpaceSizeLists();
  return KRET_OK;
}

template <typename T>
bool L2NormalizeGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                               const std::vector<AddressPtr> &workspace,
                                               const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *x_addr = GetDeviceAddress<T>(inputs, kIndex0);
  T *y_addr = GetDeviceAddress<T>(inputs, kIndex1);
  T *dy_addr = GetDeviceAddress<T>(inputs, kIndex2);
  T *dx_addr = GetDeviceAddress<T>(outputs, kIndex0);
  T *reduce_workspace_addr = GetDeviceAddress<T>(workspace, kIndex0);
  T *reduce_y_dy_workspace_addr = GetDeviceAddress<T>(workspace, kIndex1);
  T *workspace_addr = GetPossiblyNullDeviceAddress<T>(workspace, kIndex2);
  T *workspace_y_dy_addr = GetPossiblyNullDeviceAddress<T>(workspace, kIndex3);

  T alpha = static_cast<T>(1.0f);
  T beta = static_cast<T>(0.0f);

  if (all_match_) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(reduce_workspace_addr, x_addr, input_size_list_[0], cudaMemcpyDeviceToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      kernel_name_ + " cudaMemcpyAsync failed in L2Normalize::Launch.");
  } else {
    if (data_type_ == CUDNN_DATA_DOUBLE) {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnReduceTensor(cudnn_handle_, reduce_tensor_descriptor_, nullptr, 0, workspace_addr,
                          workspace_size_list_[kIndex2], &alpha, inputA_descriptor_, x_addr, &beta, outputC_descriptor_,
                          reduce_workspace_addr),
        kernel_name_ + " cudnnReduceTensor failed.");
    } else {
      const float alphaf = static_cast<float>(alpha);
      const float betaf = static_cast<float>(beta);
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnReduceTensor(cudnn_handle_, reduce_tensor_descriptor_, nullptr, 0, workspace_addr,
                          workspace_size_list_[kIndex2], &alphaf, inputA_descriptor_, x_addr, &betaf,
                          outputC_descriptor_, reduce_workspace_addr),
        kernel_name_ + " cudnnReduceTensor failed.");
    }
  }
  GetMaxWithEpsAndValue(workspace_size_list_[0] / sizeof(T), epsilon_, reduce_workspace_addr,
                        reinterpret_cast<cudaStream_t>(stream_ptr));
  BroadcastArith(output_shape_, output_shape_, output_shape_, BinaryOpType::kMul, y_addr, dy_addr, dx_addr,
                 reinterpret_cast<cudaStream_t>(stream_ptr));
  if (all_match_) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(reduce_y_dy_workspace_addr, dx_addr, output_size_list_[0], cudaMemcpyDeviceToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      kernel_name_ + " cudaMemcpyAsync failed in L2Normalize::Launch.");
  } else {
    if (data_type_ == CUDNN_DATA_DOUBLE) {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnReduceTensor(cudnn_handle_, reduce_sum_tensor_descriptor_, nullptr, 0, workspace_y_dy_addr,
                          workspace_size_list_[kIndex3], &alpha, inputA_descriptor_, dx_addr, &beta,
                          outputC_descriptor_, reduce_y_dy_workspace_addr),
        kernel_name_ + " cudnnReduceTensor failed.");
    } else {
      const float alphaf = static_cast<float>(alpha);
      const float betaf = static_cast<float>(beta);
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnReduceTensor(cudnn_handle_, reduce_sum_tensor_descriptor_, nullptr, 0, workspace_y_dy_addr,
                          workspace_size_list_[kIndex3], &alphaf, inputA_descriptor_, dx_addr, &betaf,
                          outputC_descriptor_, reduce_y_dy_workspace_addr),
        kernel_name_ + " cudnnReduceTensor failed.");
    }
  }
  BroadcastArith(rhs_shape_, lhs_shape_, output_shape_, BinaryOpType::kMul, reduce_y_dy_workspace_addr, y_addr, dx_addr,
                 reinterpret_cast<cudaStream_t>(stream_ptr));
  BroadcastArith(output_shape_, output_shape_, output_shape_, BinaryOpType::kSub, dy_addr, dx_addr, dx_addr,
                 reinterpret_cast<cudaStream_t>(stream_ptr));
  BroadcastArith(output_shape_, rhs_shape_, output_shape_, BinaryOpType::kRealDiv, dx_addr, reduce_workspace_addr,
                 dx_addr, reinterpret_cast<cudaStream_t>(stream_ptr));

  return true;
}

std::vector<std::pair<KernelAttr, L2NormalizeGradGpuKernelMod::L2NormalizeGradGpuLaunchFunc>>
  L2NormalizeGradGpuKernelMod::func_list_ = {{KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddOutputAttr(kNumberTypeFloat16),
                                              &L2NormalizeGradGpuKernelMod::LaunchKernel<half>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddOutputAttr(kNumberTypeFloat32),
                                              &L2NormalizeGradGpuKernelMod::LaunchKernel<float>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddOutputAttr(kNumberTypeFloat64),
                                              &L2NormalizeGradGpuKernelMod::LaunchKernel<double>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeInt32),
                                              &L2NormalizeGradGpuKernelMod::LaunchKernel<int>}};

std::vector<KernelAttr> L2NormalizeGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, L2NormalizeGradGpuKernelMod::L2NormalizeGradGpuLaunchFunc> &pair) {
      return pair.first;
    });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, L2NormalizeGrad, L2NormalizeGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
