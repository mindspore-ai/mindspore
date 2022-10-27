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

#include "mindspore/ccsrc/plugin/device/gpu/kernel/sparse/sparse_split_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
constexpr size_t InputsNum = 4;
constexpr int64_t Kindex2 = 2;
constexpr int64_t Kindex3 = 3;
template <typename T>
using Complex = mindspore::utils::Complex<T>;
bool SparseSplitGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::SparseSplit>(base_operator);
  num_split = kernel_ptr->get_num_split();

  input_dtype_ = inputs[kIndex2]->GetDtype();
  size_t outputs_num = Kindex3 * num_split;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), outputs_num, kernel_name_);
  std::map<TypeId, SparseSplitGpuKernelMod::SparseSplitLaunchFunc> kernel_list = {
    {kNumberTypeUInt8, &SparseSplitGpuKernelMod::LaunchKernel<uint8_t, int64_t>},
    {kNumberTypeUInt16, &SparseSplitGpuKernelMod::LaunchKernel<uint16_t, int64_t>},
    {kNumberTypeInt64, &SparseSplitGpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {kNumberTypeInt32, &SparseSplitGpuKernelMod::LaunchKernel<int32_t, int64_t>},
    {kNumberTypeInt16, &SparseSplitGpuKernelMod::LaunchKernel<int16_t, int64_t>},
    {kNumberTypeInt8, &SparseSplitGpuKernelMod::LaunchKernel<int8_t, int64_t>},
    {kNumberTypeFloat64, &SparseSplitGpuKernelMod::LaunchKernel<double, int64_t>},
    {kNumberTypeFloat, &SparseSplitGpuKernelMod::LaunchKernel<float, int64_t>},
    {kNumberTypeFloat16, &SparseSplitGpuKernelMod::LaunchKernel<half, int64_t>},
    {kNumberTypeBool, &SparseSplitGpuKernelMod::LaunchKernel<bool, int64_t>},
  };

  kernel_func_ = kernel_list[input_dtype_];
  is_need_retrieve_output_shape_ = true;
  return true;
}

int SparseSplitGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  auto kernel_ptr = std::dynamic_pointer_cast<ops::SparseSplit>(base_operator);
  num_split = kernel_ptr->get_num_split();
  if (ret == KRET_UNKNOWN_OUT_SHAPE) {
    outputs_ = outputs;
    auto input_indices_shape = inputs[kIndex1]->GetShapeVector();
    auto out_shape = outputs.at(kIndex2)->GetShapeVector();
    auto out_size = std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<int64_t>());
    out_size_ = out_size;

    input_nnz_ = input_indices_shape[0];
    num_dim_ = input_indices_shape[1];
    input_dtype_ = inputs[kIndex2]->GetDtype();

    output_size_list_.clear();
    for (size_t i = 0; i < num_split; i++) {
      (void)output_size_list_.emplace_back(input_nnz_ * num_dim_ * GetTypeByte(TypeIdToType(inputs[1]->GetDtype())));
    }
    for (size_t i = 0; i < num_split; i++) {
      (void)output_size_list_.emplace_back(input_nnz_ * GetTypeByte(TypeIdToType(inputs[Kindex2]->GetDtype())));
    }
    for (size_t i = 0; i < num_split; i++) {
      (void)output_size_list_.emplace_back(num_dim_ * GetTypeByte(TypeIdToType(inputs[Kindex3]->GetDtype())));
    }

    workspace_size_list_.clear();
    workspace_size_list_.push_back(num_split * sizeof(void *));
    workspace_size_list_.push_back(num_split * sizeof(void *));
    workspace_size_list_.push_back(num_split * sizeof(void *));
    workspace_size_list_.push_back(num_split * sizeof(int));
    workspace_size_list_.push_back((num_split + 1) * GetTypeByte(TypeIdToType(inputs[1]->GetDtype())));
  }
  return ret;
}

template <typename DataType, typename IndexType>
bool SparseSplitGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &workspace,
                                           const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  MS_EXCEPTION_IF_NULL(cuda_stream);
  auto split_dim_ptr = GetDeviceAddress<IndexType>(inputs, kIndex0);
  auto indices_ptr = GetDeviceAddress<IndexType>(inputs, kIndex1);
  auto values_ptr = GetDeviceAddress<DataType>(inputs, kIndex2);
  auto shape_ptr = GetDeviceAddress<IndexType>(inputs, kIndex3);
  std::vector<IndexType *> y_indices_vec;
  std::vector<DataType *> y_values_ptr;
  std::vector<IndexType *> out_shape_ptr;
  std::vector<IndexType> out_shape_value(num_split * Kindex2, 0);
  for (size_t i = 0; i < num_split; i++) {
    y_indices_vec.push_back(GetDeviceAddress<IndexType>(outputs, i));
    y_values_ptr.push_back(GetDeviceAddress<DataType>(outputs, num_split + i));
    out_shape_ptr.push_back(GetDeviceAddress<IndexType>(outputs, num_split * Kindex2 + i));
  }
  auto d_y_indices_vec = GetDeviceAddress<IndexType *>(workspace, kIndex0);
  auto d_y_values_ptr = GetDeviceAddress<DataType *>(workspace, kIndex1);
  auto d_out_shape_ptr = GetDeviceAddress<IndexType>(workspace, kIndex2);

  std::vector<IndexType> h_shape(Kindex2);

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(h_shape.data(), shape_ptr, sizeof(IndexType) * h_shape.size(), cudaMemcpyDeviceToHost, cuda_stream),
    "For SparseSplit, cudaMemcpyAsync failed.");

  // std::vector<IndexType> h_block(num_split + 1);
  h_block.resize(num_split + 1);
  // IndexType h_split_dim;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&h_split_dim, split_dim_ptr, sizeof(IndexType), cudaMemcpyDeviceToHost, cuda_stream),
    "For SparseSplit, cudaMemcpyAsync failed.");
  h_block[0] = 0;
  int base_range = h_shape[h_split_dim] / num_split;
  size_t res = h_shape[h_split_dim] - base_range * num_split;
  for (size_t i = 1; i < h_block.size(); i++) {
    if (i > 1) {
      h_block[i] = h_block[i - 1] + base_range;
    } else {
      h_block[i] = base_range;
    }
    if (i <= res) {
      h_block[i] += 1;
    }
  }

  for (size_t i = 0; i < num_split; i++) {
    if (i == 0) {
      out_shape_value[i * Kindex2 + h_split_dim] = (IndexType)h_block[i + 1];
    } else {
      out_shape_value[i * Kindex2 + h_split_dim] = (IndexType)(h_block[i + 1] - h_block[i]);
    }
    out_shape_value[i * Kindex2 + 1 - h_split_dim] = h_shape[1 - h_split_dim];
  }

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(d_y_indices_vec, y_indices_vec.data(), sizeof(IndexType *) * num_split, cudaMemcpyHostToDevice,
                    cuda_stream),
    "For SparseSplit, cudaMemcpyAsync failed.");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(d_y_values_ptr, y_values_ptr.data(), sizeof(DataType *) * num_split, cudaMemcpyHostToDevice,
                    cuda_stream),
    "For SparseSplit, cudaMemcpyAsync failed.");

  for (size_t i = 0; i < num_split; i++) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(out_shape_ptr[i], out_shape_value.data() + i * Kindex2, sizeof(IndexType) * Kindex2,
                      cudaMemcpyHostToDevice, cuda_stream),
      "For SparseSplit out_shape_ptr, cudaMemcpyAsync failed.");
  }
  auto d_block_ptr = GetDeviceAddress<int64_t>(workspace, kIndex4);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(d_block_ptr, h_block.data(), sizeof(IndexType) * h_block.size(),
                                                     cudaMemcpyHostToDevice, cuda_stream),
                                     "For SparseSplit, cudaMemcpyAsync failed.");

  auto sum_count_ptr = GetDeviceAddress<int>(workspace, kIndex3);

  SparseSplit<DataType, IndexType>(split_dim_ptr, indices_ptr, values_ptr, shape_ptr, num_split, d_y_indices_vec,
                                   d_y_values_ptr, d_out_shape_ptr, sum_count_ptr, input_nnz_, num_dim_, d_block_ptr,
                                   cuda_stream);
  h_blocks.resize(num_split);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(h_blocks.data(), sum_count_ptr, sizeof(int) * num_split, cudaMemcpyDeviceToHost, cuda_stream),
    "For SparseSplit, cudaMemcpyAsync failed.");

  return true;
}

void SparseSplitGpuKernelMod::SyncData() {
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream), "SparseSplit cudaStreamSynchronized failed");
  for (size_t i = 0; i < num_split; i++) {
    outputs_[i]->SetShapeVector(ShapeVector({h_blocks[i], Kindex2}));                                  // indices
    outputs_[i + num_split]->SetShapeVector(ShapeVector({h_blocks[i]}));                               // value
    outputs_[i + num_split * Kindex2]->SetShapeVector(ShapeVector({static_cast<int64_t>(num_dim_)}));  // shape
  }
}

std::vector<KernelAttr> SparseSplitGpuKernelMod::GetOpSupport() { return {KernelAttr().AddSkipCheckAttr(true)}; }

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseSplit, SparseSplitGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
