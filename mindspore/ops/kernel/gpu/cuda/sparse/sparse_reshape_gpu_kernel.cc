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

#include "kernel/gpu/cuda/sparse/sparse_reshape_gpu_kernel.h"

namespace mindspore {
namespace kernel {
constexpr int64_t INDICES_DIMS = 2;
constexpr int64_t SHAPE_DIMS = 1;
constexpr int64_t UNKNOWN_ERR = 1;
constexpr int64_t NEGTIVE_ERR = 2;
constexpr int64_t INFER_ERR = 3;
constexpr int64_t SHAPE_ERR = 4;
bool SparseReshapeGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the kernel type should be int64, but got: " << kernel_attr
                      << ".";
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int SparseReshapeGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  std::vector<int64_t> indices_shape = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeVector().begin(),
                                                            inputs.at(kIndex0)->GetDeviceShapeVector().end());
  std::vector<int64_t> shape_shape = std::vector<int64_t>(inputs.at(kIndex1)->GetDeviceShapeVector().begin(),
                                                          inputs.at(kIndex1)->GetDeviceShapeVector().end());
  std::vector<int64_t> new_shape_shape = std::vector<int64_t>(inputs.at(kIndex2)->GetDeviceShapeVector().begin(),
                                                              inputs.at(kIndex2)->GetDeviceShapeVector().end());
  if (indices_shape.size() != INDICES_DIMS) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'indices' should be 2-D, but got "
                  << indices_shape.size() << "-D.";
    return false;
  }
  if (shape_shape.size() != SHAPE_DIMS) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'shape' should be 1-D, but got "
                  << shape_shape.size() << "-D.";
    return false;
  }
  if (new_shape_shape.size() != SHAPE_DIMS) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'new_shape' should be 1-D, but got "
                  << new_shape_shape.size() << "-D.";
    return false;
  }
  indice_number_ = indices_shape[0];
  indice_dims_ = indices_shape[1];
  shape_elements_ = shape_shape[0];
  new_shape_elements_ = new_shape_shape[0];
  if (indice_dims_ != shape_elements_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of the non-zero element is supposed to be the same as the 'shape' ";
  }
  return KRET_OK;
}

bool SparseReshapeGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &workspace,
                                             const std::vector<KernelTensor *> &outputs) {
  int64_t *indices = GetDeviceAddress<int64_t>(inputs, kIndex0);
  int64_t *shape = GetDeviceAddress<int64_t>(inputs, kIndex1);
  int64_t *new_shape = GetDeviceAddress<int64_t>(inputs, kIndex2);
  int64_t *y_indices = GetDeviceAddress<int64_t>(outputs, kIndex0);
  int64_t *y_shape = GetDeviceAddress<int64_t>(outputs, kIndex1);
  const size_t shape_size = shape_elements_;
  const size_t new_shape_size = new_shape_elements_;
  std::vector<int64_t> h_shape(shape_size);
  std::vector<int64_t> h_new_shape(new_shape_size);
  std::vector<int64_t> h_y_shape(new_shape_size);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(h_shape.data(), shape, sizeof(int64_t) * shape_elements_, cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For 'SparseReshape', cudaMemcpy h_shape variable failed.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(h_new_shape.data(), new_shape, sizeof(int64_t) * new_shape_elements_, cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For 'SparseReshape', cudaMemcpy h_new_shape variable failed.");
  if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(cuda_stream_)) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       "For 'SparseReshape', cuda Stream Sync Failed.");
  }
  int status = CalShape(h_shape.data(), h_new_shape.data(), h_y_shape.data(), shape_elements_, new_shape_elements_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(y_shape, h_y_shape.data(), sizeof(int64_t) * new_shape_elements_, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "cudaMemcpy y_shape variable failed.");
  switch (status) {
    case UNKNOWN_ERR:
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', there should be at most one '-1' dimension in 'newshape' tensor, but got two or more.";
      return false;
    case NEGTIVE_ERR:
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the size of newshape should be a non-negative number.";
      return false;
    case INFER_ERR:
      MS_LOG(ERROR) << "For '" << kernel_name_ << "' The indefinite dimension cannot be inferred.";
      return false;
    case SHAPE_ERR:
      MS_LOG(ERROR) << "For '" << kernel_name_ << "' The number of elements in shape differ from that in new_shape.";
      return false;
  }
  auto ret = CalSparseReshape(indices, shape, y_indices, y_shape, indice_number_, shape_elements_, new_shape_elements_,
                              device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(ret, kernel_name_);
  return true;
}

int SparseReshapeGpuKernelMod::CalShape(const int64_t *shape, const int64_t *new_shape, int64_t *y_shape,
                                        const size_t shape_elements_, const size_t new_shape_elements_) {
  int64_t prod = 1;
  int64_t dense_size = 1;
  int64_t loc = -1;
  int64_t out = 1;
  for (size_t i = 0; i < shape_elements_; ++i) {
    dense_size *= *(shape + i);
  }
  for (size_t i = 0; i < new_shape_elements_; ++i) {
    const int64_t size = *(new_shape + i);
    if (size == -1) {
      if (loc != -1) {
        return UNKNOWN_ERR;
      }
      loc = i;
    } else {
      if (size < 0) {
        return NEGTIVE_ERR;
      }
      prod *= size;
      *(y_shape + i) = size;
      out *= size;
    }
  }
  if (loc != -1) {
    const int64_t missing = dense_size / prod;
    if (missing * prod != dense_size) {
      return INFER_ERR;
    }
    out *= missing;
    *(y_shape + loc) = missing;
  }
  if (out != dense_size) {
    return SHAPE_ERR;
  }
  return 0;
}

std::vector<std::pair<KernelAttr, SparseReshapeGpuKernelMod::SparseReshapeFunc>> SparseReshapeGpuKernelMod::func_list_ =
  {{KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64),
    &SparseReshapeGpuKernelMod::LaunchKernel}};

std::vector<KernelAttr> SparseReshapeGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseReshapeFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseReshape, SparseReshapeGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
