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

#include "plugin/device/gpu/kernel/arrays/index_fill_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kIndexFillInputsNum = 4;
constexpr size_t kIndexFillOutputsNum = 1;
}  // namespace

bool IndexFillGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kIndexFillInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kIndexFillOutputsNum, kernel_name_);
  kernel_name_ = base_operator->GetPrim()->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

void IndexFillGpuKernelMod::UpdateSize(const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &) {
  x_shape_ = inputs.at(kIndex0)->GetShapeVector();
  auto index_shape = inputs.at(kIndex2)->GetShapeVector();
  int64_t init = 1;
  x_num_ = std::accumulate(x_shape_.begin(), x_shape_.end(), init, std::multiplies{});
  index_num_ = std::accumulate(index_shape.begin(), index_shape.end(), init, std::multiplies{});
}

int IndexFillGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  UpdateSize(inputs, outputs);
  workspace_size_list_.push_back(sizeof(bool));  // Place out_bound.
  return KRET_OK;
}

template <typename DataType, typename IndexType>
bool IndexFillGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto x_ptr = GetDeviceAddress<DataType>(inputs, kIndex0);
  auto dim_ptr = inputs[kIndex1]->addr;
  auto index_ptr = GetDeviceAddress<IndexType>(inputs, kIndex2);
  auto value_ptr = GetDeviceAddress<DataType>(inputs, kIndex3);
  auto y_ptr = GetDeviceAddress<DataType>(outputs, kIndex0);
  auto out_bound_ptr = GetDeviceAddress<bool>(workspace, kIndex0);
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto any = [](auto &&... args) -> bool { return ((args == nullptr) || ...); };
  if (any(x_ptr, dim_ptr, index_ptr, value_ptr, y_ptr, out_bound_ptr, cuda_stream)) {
    return false;
  }

  // Copy from 'x' into 'y'.
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(y_ptr, x_ptr, x_num_ * sizeof(DataType), cudaMemcpyDeviceToDevice, cuda_stream),
    "In IndexFill kernel, cudaMemcpyAsync output 'y' from 'x' failed.");
  if (index_num_ == 0) {
    return true;
  }
  // Initialize and check 'dim'.
  int rank = static_cast<int>(x_shape_.size());
  int dim;
  if (inputs[kIndex1]->size == sizeof(int)) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpy(&dim, dim_ptr, inputs[kIndex1]->size, cudaMemcpyDeviceToHost),
                                       "In IndexFill kernel, cudaMemcpy input 'dim' device to host failed.");
  } else {
    int64_t dim_tmp;
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpy(&dim_tmp, dim_ptr, inputs[kIndex1]->size, cudaMemcpyDeviceToHost),
                                       "In IndexFill kernel, cudaMemcpy input 'dim' device to host failed.");
    dim = static_cast<int>(dim_tmp);
  }
  if (dim < -rank || dim >= rank) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'dim' must be in the range [-" << rank << "," << rank
                  << "), but got " << dim;
    return false;
  } else if (dim < 0) {
    dim = dim + rank;
  }
  // Initialize out_bound_ptr.
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemsetAsync(out_bound_ptr, 0, sizeof(bool), cuda_stream),
                                     "In IndexFill kernel, cudaMemsetAsync out_bound variable failed.");
  // Prepare index_num, dim_size, outer_size, inner_size
  int64_t dim_size = 1;
  int64_t outer_size = 1;
  int64_t inner_size = 1;
  for (size_t i = 0; i < x_shape_.size(); i++) {
    int idx = static_cast<int>(i);
    if (idx < dim) {
      outer_size *= x_shape_.at(i);
    } else if (idx > dim) {
      inner_size *= x_shape_.at(i);
    } else {
      dim_size = x_shape_.at(i);
    }
  }
  IndexFill(y_ptr, index_ptr, index_num_, outer_size, dim_size, inner_size, value_ptr, out_bound_ptr, device_id_,
            cuda_stream);

  bool out_bound = false;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&out_bound, out_bound_ptr, sizeof(bool), cudaMemcpyDeviceToHost, cuda_stream),
    "In IndexFill kernel, cudaMemcpyAsync out_bound_ variable failed.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream),
                                     "In IndexFill kernel, cudaStreamSynchronized failed");
  if (out_bound) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the input 'index' is out of bound.";
    return false;
  }
  return true;
}

std::vector<std::pair<KernelAttr, IndexFillGpuKernelMod::IndexFillLaunchFunc>> IndexFillGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &IndexFillGpuKernelMod::LaunchKernel<uint8_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt16)
     .AddOutputAttr(kNumberTypeUInt16),
   &IndexFillGpuKernelMod::LaunchKernel<uint16_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddOutputAttr(kNumberTypeUInt32),
   &IndexFillGpuKernelMod::LaunchKernel<uint32_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt64)
     .AddOutputAttr(kNumberTypeUInt64),
   &IndexFillGpuKernelMod::LaunchKernel<uint64_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &IndexFillGpuKernelMod::LaunchKernel<int8_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &IndexFillGpuKernelMod::LaunchKernel<int16_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &IndexFillGpuKernelMod::LaunchKernel<int32_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &IndexFillGpuKernelMod::LaunchKernel<int64_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &IndexFillGpuKernelMod::LaunchKernel<half, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &IndexFillGpuKernelMod::LaunchKernel<float, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &IndexFillGpuKernelMod::LaunchKernel<double, int>},
};

std::vector<KernelAttr> IndexFillGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, IndexFillGpuKernelMod::IndexFillLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

class IndexFillVmapGpuKernelMod : public IndexFillGpuKernelMod {
 public:
  IndexFillVmapGpuKernelMod() = default;
  ~IndexFillVmapGpuKernelMod() override = default;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &others) override {
    batch_rank_ = base_operator->get_batch_rank();
    if (batch_rank_ <= 0) {
      return IndexFillGpuKernelMod::Resize(base_operator, inputs, outputs, others);
    } else {
      auto input_shape = inputs.at(kIndex0)->GetShapeVector();
      batch_size_ = std::accumulate(input_shape.begin(), input_shape.begin() + batch_rank_,
                                    decltype(input_shape)::value_type(1), std::multiplies{});
      int ret = IndexFillGpuKernelMod::Resize(base_operator, inputs, outputs, others);
      auto new_inputs = inputs;
      for (auto &input : new_inputs) {
        auto shape = input->GetShapeVector();
        std::vector<int64_t> new_shape(shape.begin() + batch_rank_, shape.end());
        input->SetShapeVector(new_shape);
      }
      auto new_outputs = outputs;
      for (auto &output : new_outputs) {
        auto shape = output->GetShapeVector();
        std::vector<int64_t> new_shape(shape.begin() + batch_rank_, shape.end());
        output->SetShapeVector(new_shape);
      }
      UpdateSize(new_inputs, new_outputs);
      return ret;
    }
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (batch_rank_ <= 0) {
      return IndexFillGpuKernelMod::Launch(inputs, workspace, outputs, stream_ptr);
    } else {
      // Initialize address list of inputs and outputs.
      std::vector<AddressPtr> new_inputs;
      std::vector<AddressPtr> new_outputs;
      (void)std::transform(
        inputs.begin(), inputs.end(), std::back_inserter(new_inputs),
        [batch_size = batch_size_](auto ptr) { return std::make_shared<Address>(ptr->addr, ptr->size / batch_size); });
      (void)std::transform(
        outputs.begin(), outputs.end(), std::back_inserter(new_outputs),
        [batch_size = batch_size_](auto ptr) { return std::make_shared<Address>(ptr->addr, ptr->size / batch_size); });
      for (int64_t i = 0; i < batch_size_; i++) {
        if (!IndexFillGpuKernelMod::Launch(new_inputs, workspace, new_outputs, stream_ptr)) {
          return false;
        }
        (void)std::for_each(new_inputs.begin(), new_inputs.end(), [](auto &ptr) {
          ptr->addr = reinterpret_cast<void *>(reinterpret_cast<char *>(ptr->addr) + (ptr->size));
        });
        (void)std::for_each(new_outputs.begin(), new_outputs.end(), [](auto &ptr) {
          ptr->addr = reinterpret_cast<void *>(reinterpret_cast<char *>(ptr->addr) + (ptr->size));
        });
      }
      return true;
    }
  }

 private:
  int64_t batch_rank_{0};
  int64_t batch_size_{1};
};
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, IndexFill, IndexFillVmapGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
