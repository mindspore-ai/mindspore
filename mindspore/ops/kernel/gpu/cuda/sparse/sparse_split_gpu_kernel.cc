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

#include "mindspore/ops/kernel/gpu/cuda/sparse/sparse_split_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
constexpr size_t InputsNum = 4;
constexpr int64_t Kindex2 = 2;
constexpr int64_t Kindex3 = 3;
template <typename T>
using Complex = mindspore::utils::Complex<T>;
bool SparseSplitGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  num_split = static_cast<size_t>(GetValue<int64_t>(primitive_->GetAttr("num_split")));

  input_dtype_ = inputs[kIndex2]->dtype_id();
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
    {kNumberTypeFloat32, &SparseSplitGpuKernelMod::LaunchKernel<float, int64_t>},
    {kNumberTypeFloat16, &SparseSplitGpuKernelMod::LaunchKernel<half, int64_t>},
    {kNumberTypeBool, &SparseSplitGpuKernelMod::LaunchKernel<bool, int64_t>},
  };
  if (kernel_list.find(input_dtype_) == kernel_list.end()) {
    MS_LOG(ERROR) << "SparseSplit does not support this data type.";
    return false;
  }
  kernel_func_ = kernel_list[input_dtype_];
  return true;
}

int SparseSplitGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto input_indices_shape = inputs[kIndex1]->GetShapeVector();
  input_nnz_ = input_indices_shape[0];
  num_dim_ = input_indices_shape[1];

  num_split = static_cast<size_t>(GetValue<int64_t>(primitive_->GetAttr("num_split")));
  workspace_size_list_.push_back(num_split * sizeof(void *));
  workspace_size_list_.push_back(num_split * sizeof(void *));
  workspace_size_list_.push_back(num_split * sizeof(void *));
  workspace_size_list_.push_back(num_split * sizeof(int));
  workspace_size_list_.push_back((num_split + 1) * GetTypeByte(TypeIdToType(inputs[1]->dtype_id())));

  return KRET_OK;
}

template <typename DataType, typename IndexType>
bool SparseSplitGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &workspace,
                                           const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
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
    "For SparseSplit, cudaMemcpyAsync shape failed.");
  if (cudaStreamQuery(cuda_stream) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream),
                                       "For 'SparseSplit', cuda Stream Sync Failed.");
  }

  h_block.resize(num_split + 1);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&h_split_dim, split_dim_ptr, sizeof(IndexType), cudaMemcpyDeviceToHost, cuda_stream),
    "For SparseSplit, cudaMemcpyAsync split_dim failed.");
  if (cudaStreamQuery(cuda_stream) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream),
                                       "For 'SparseSplit', cuda Stream Sync Failed.");
  }
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

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemsetAsync(sum_count_ptr, 0, workspace[kIndex3]->size(), cuda_stream),
                                     "For SparseSplit, cudaMemsetAsync failed.");

  SparseSplit<DataType, IndexType>(split_dim_ptr, indices_ptr, values_ptr, shape_ptr, num_split, d_y_indices_vec,
                                   d_y_values_ptr, d_out_shape_ptr, sum_count_ptr, input_nnz_, num_dim_, d_block_ptr,
                                   cuda_stream);
  h_blocks.resize(num_split);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(h_blocks.data(), sum_count_ptr, sizeof(int) * num_split, cudaMemcpyDeviceToHost, cuda_stream),
    "For SparseSplit, cudaMemcpyAsync failed.");

  return true;
}

void SparseSplitGpuKernelMod::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                                       const std::vector<KernelTensor *> &outputs) {
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream), "SparseSplit cudaStreamSynchronized failed");
  for (size_t i = 0; i < num_split; i++) {
    outputs[i]->SetShapeVector(ShapeVector({h_blocks[i], Kindex2}));                                  // indices
    outputs[i + num_split]->SetShapeVector(ShapeVector({h_blocks[i]}));                               // value
    outputs[i + num_split * Kindex2]->SetShapeVector(ShapeVector({static_cast<int64_t>(num_dim_)}));  // shape
    outputs[i]->set_size(LongToSize(h_blocks[i] * Kindex2) * UnitSizeInBytes(outputs[i]->dtype_id()));
    outputs[i + num_split]->set_size(LongToSize(h_blocks[i]) * UnitSizeInBytes(outputs[i + num_split]->dtype_id()));
    outputs[i + num_split * Kindex2]->set_size(num_dim_ *
                                               UnitSizeInBytes(outputs[i + num_split * Kindex2]->dtype_id()));
  }
}

std::vector<KernelAttr> SparseSplitGpuKernelMod::GetOpSupport() { return {KernelAttr().AddSkipCheckAttr(true)}; }

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseSplit, SparseSplitGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
