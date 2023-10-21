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

#include "plugin/device/gpu/kernel/math/lu_unpack_gpu_kernel.h"
namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kFirstDim = 1;
constexpr size_t kSecondDim = 2;
}  // namespace

void LuUnpackGpuKernelMod::ResetResource() noexcept {
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

template <typename T, typename S>
bool LuUnpackGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                        const std::vector<AddressPtr> &outputs) {
  T *lu_data_ptr = GetDeviceAddress<T>(inputs, kIndex0);
  MS_EXCEPTION_IF_NULL(lu_data_ptr);
  S *lu_pivots_ptr = GetDeviceAddress<S>(inputs, kIndex1);
  MS_EXCEPTION_IF_NULL(lu_pivots_ptr);

  T *pivots_ptr = GetDeviceAddress<T>(outputs, kIndex0);
  T *l_ptr = GetDeviceAddress<T>(outputs, kIndex1);
  T *u_ptr = GetDeviceAddress<T>(outputs, kIndex2);
  S *final_order = GetDeviceAddress<S>(workspace, kIndex0);

  // triu and tril
  cudaError_t status = cudaErrorNotReady;
  if (lu_data_dim1_ > lu_data_dim2_) {
    status = CalTriuAux(batch_num_ * u_stride_, lu_data_ptr, lu_data_dim2_, lu_data_dim2_, lu_data_dim1_, lu_data_dim2_,
                        u_ptr, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
    CHECK_CUDA_STATUS(status, kernel_name_);
    status = CalTrilAux(batch_num_ * l_stride_, lu_data_ptr, lu_data_dim1_, lu_data_dim2_, lu_data_dim1_, lu_data_dim2_,
                        l_ptr, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
    CHECK_CUDA_STATUS(status, kernel_name_);
  } else {
    status = CalTriuAux(batch_num_ * u_stride_, lu_data_ptr, lu_data_dim1_, lu_data_dim2_, lu_data_dim1_, lu_data_dim2_,
                        u_ptr, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
    CHECK_CUDA_STATUS(status, kernel_name_);
    status = CalTrilAux(batch_num_ * l_stride_, lu_data_ptr, lu_data_dim1_, lu_data_dim1_, lu_data_dim1_, lu_data_dim2_,
                        l_ptr, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
    CHECK_CUDA_STATUS(status, kernel_name_);
  }

  // swap and index_select
  status = TransposeEyeMatrix(lu_pivots_ptr, pivots_ptr, final_order, batch_num_, lu_data_dim1_, lu_pivots_dim_,
                              device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

bool LuUnpackGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();

  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "LuUnpack does not support this kernel data type: " << kernel_attr << ".";
    return false;
  }

  kernel_func_ = func_list_[index].second;
  unit_size1_ = abstract::TypeIdSize(inputs.at(kIndex0)->GetDtype());
  unit_size2_ = abstract::TypeIdSize(inputs.at(kIndex1)->GetDtype());

  return true;
}

int LuUnpackGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  std::vector<int64_t> lu_data_shape = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                            inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> lu_pivots_shape = std::vector<int64_t>(inputs.at(kIndex1)->GetDeviceShapeAdaptively().begin(),
                                                              inputs.at(kIndex1)->GetDeviceShapeAdaptively().end());
  lu_data_size_ = size_t(std::accumulate(lu_data_shape.begin(), lu_data_shape.end(), 1, std::multiplies<int64_t>()));
  lu_pivots_size_ =
    size_t(std::accumulate(lu_pivots_shape.begin(), lu_pivots_shape.end(), 1, std::multiplies<int64_t>()));

  auto lu_data_shape_vec = inputs[kIndex0]->GetShapeVector();
  auto lu_pivots_shape_vec = inputs[kIndex1]->GetShapeVector();
  size_t lu_data_dims = lu_data_shape_vec.size();
  size_t lu_pivots_dims = lu_pivots_shape_vec.size();
  std::vector<int64_t> lu_data_dims_vector = lu_data_shape_vec;
  std::vector<int64_t> lu_pivots_dims_vector = lu_pivots_shape_vec;

  lu_data_dim1_ = lu_data_dims_vector[lu_data_dims - kSecondDim];
  lu_data_dim2_ = lu_data_dims_vector[lu_data_dims - kFirstDim];
  if (lu_data_dim1_ > lu_data_dim2_) {
    l_stride_ = lu_data_dim1_ * lu_data_dim2_;
    u_stride_ = lu_data_dim2_ * lu_data_dim2_;
  } else {
    l_stride_ = lu_data_dim1_ * lu_data_dim1_;
    u_stride_ = lu_data_dim1_ * lu_data_dim2_;
  }
  lu_pivots_dim_ = lu_pivots_dims_vector[lu_pivots_dims - kFirstDim];
  batch_num_ = lu_data_size_ / (lu_data_dim1_ * lu_data_dim2_);
  workspace_size_list_.push_back(batch_num_ * lu_data_dim1_ * unit_size2_);
  return KRET_OK;
}

bool LuUnpackGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                  const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  cuda_stream_ = stream_ptr;
  return kernel_func_(this, inputs, workspace, outputs);
}

std::vector<KernelAttr> LuUnpackGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LuUnpackFunc> &pair) { return pair.first; });
  return support_list;
}

std::vector<std::pair<KernelAttr, LuUnpackGpuKernelMod::LuUnpackFunc>> LuUnpackGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &LuUnpackGpuKernelMod::LaunchKernel<double, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &LuUnpackGpuKernelMod::LaunchKernel<double, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &LuUnpackGpuKernelMod::LaunchKernel<double, int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &LuUnpackGpuKernelMod::LaunchKernel<double, int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &LuUnpackGpuKernelMod::LaunchKernel<double, uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LuUnpackGpuKernelMod::LaunchKernel<float, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LuUnpackGpuKernelMod::LaunchKernel<float, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LuUnpackGpuKernelMod::LaunchKernel<float, int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LuUnpackGpuKernelMod::LaunchKernel<float, int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LuUnpackGpuKernelMod::LaunchKernel<float, uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &LuUnpackGpuKernelMod::LaunchKernel<half, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &LuUnpackGpuKernelMod::LaunchKernel<half, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &LuUnpackGpuKernelMod::LaunchKernel<half, int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &LuUnpackGpuKernelMod::LaunchKernel<half, int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &LuUnpackGpuKernelMod::LaunchKernel<half, uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &LuUnpackGpuKernelMod::LaunchKernel<int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &LuUnpackGpuKernelMod::LaunchKernel<int64_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &LuUnpackGpuKernelMod::LaunchKernel<int64_t, int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &LuUnpackGpuKernelMod::LaunchKernel<int64_t, int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &LuUnpackGpuKernelMod::LaunchKernel<int64_t, uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &LuUnpackGpuKernelMod::LaunchKernel<int32_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &LuUnpackGpuKernelMod::LaunchKernel<int32_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &LuUnpackGpuKernelMod::LaunchKernel<int32_t, int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &LuUnpackGpuKernelMod::LaunchKernel<int32_t, int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &LuUnpackGpuKernelMod::LaunchKernel<int32_t, uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &LuUnpackGpuKernelMod::LaunchKernel<int32_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &LuUnpackGpuKernelMod::LaunchKernel<int32_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &LuUnpackGpuKernelMod::LaunchKernel<int32_t, int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &LuUnpackGpuKernelMod::LaunchKernel<int32_t, int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &LuUnpackGpuKernelMod::LaunchKernel<int32_t, uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &LuUnpackGpuKernelMod::LaunchKernel<int16_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &LuUnpackGpuKernelMod::LaunchKernel<int16_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &LuUnpackGpuKernelMod::LaunchKernel<int16_t, int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &LuUnpackGpuKernelMod::LaunchKernel<int16_t, int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &LuUnpackGpuKernelMod::LaunchKernel<int16_t, uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &LuUnpackGpuKernelMod::LaunchKernel<int8_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &LuUnpackGpuKernelMod::LaunchKernel<int8_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &LuUnpackGpuKernelMod::LaunchKernel<int8_t, int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &LuUnpackGpuKernelMod::LaunchKernel<int8_t, int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &LuUnpackGpuKernelMod::LaunchKernel<int8_t, uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &LuUnpackGpuKernelMod::LaunchKernel<int8_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &LuUnpackGpuKernelMod::LaunchKernel<int8_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &LuUnpackGpuKernelMod::LaunchKernel<int8_t, int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &LuUnpackGpuKernelMod::LaunchKernel<int8_t, int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &LuUnpackGpuKernelMod::LaunchKernel<int8_t, uint8_t>}};

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, LuUnpack, LuUnpackGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
