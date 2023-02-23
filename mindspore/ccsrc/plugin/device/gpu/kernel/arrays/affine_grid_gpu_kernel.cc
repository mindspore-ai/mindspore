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

#include "plugin/device/gpu/kernel/arrays/affine_grid_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include <memory>

namespace mindspore {
namespace kernel {
namespace {
constexpr int N_INPUTS = 2;
constexpr int N_OUTPUTS = 1;
constexpr int RANK_THETA = 3;
constexpr int RANK_IMAGE_SIZE = 1;
constexpr int N_ROWS_THETA_4D = 2;
constexpr int N_COLS_THETA_4D = 3;
constexpr int LEN_IMAGE_SIZE_4D = 4;
constexpr int N_ROWS_THETA_5D = 3;
constexpr int N_COLS_THETA_5D = 4;
constexpr int LEN_IMAGE_SIZE_5D = 5;
constexpr int RANK_GRID_4D = 4;
constexpr int RANK_GRID_5D = 5;

bool PreLaunchKernel4D(const std::vector<int64_t> &theta_shape, const std::vector<int64_t> &grid_shape,
                       const int32_t *image_size_d, int32_t *image_size_h, cudaStream_t cuda_stream,
                       const std::string &kernel_name) {
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(image_size_h, image_size_d, sizeof(int32_t) * kDim4, cudaMemcpyDeviceToHost, cuda_stream),
    "For '" << kernel_name << "', "
            << "cudaMemcpy input 'size' to host failed.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaDeviceSynchronize(), "For '" << kernel_name << "', "
                                                                      << "cudaDeviceSyncFailed");
  int32_t N = image_size_h[kIndex0];
  int32_t C = image_size_h[kIndex1];
  int32_t H = image_size_h[kIndex2];
  int32_t W = image_size_h[kIndex3];
  bool is_valid = (N == theta_shape[kIndex0]) && (N == grid_shape[kIndex0]) && (H == grid_shape[kIndex1]) &&
                  (W == grid_shape[kIndex2]);
  if (is_valid) {
    return true;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name << "', "
                  << "the inferred shape of sampling grids (N×H×W×2) must match the value of 'size' (N×C×H×W). "
                  << "The inferred shape is (" << grid_shape[kIndex0] << ", " << grid_shape[kIndex1] << ", "
                  << grid_shape[kIndex2] << ", " << grid_shape[kIndex3] << "), "
                  << "while the value of 'size' is (" << N << ", " << C << ", " << H << ", " << W << ").\n";
    return false;
  }
}

bool PreLaunchKernel5D(const std::vector<int64_t> &theta_shape, const std::vector<int64_t> &grid_shape,
                       const int32_t *image_size_d, int32_t *image_size_h, cudaStream_t cuda_stream,
                       const std::string &kernel_name) {
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(image_size_h, image_size_d, sizeof(int32_t) * kDim5, cudaMemcpyDeviceToHost, cuda_stream),
    "For '" << kernel_name << "', "
            << "cudaMemcpy input 'size' to host failed.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaDeviceSynchronize(), "For '" << kernel_name << "', "
                                                                      << "cudaDeviceSyncFailed");
  int32_t N = image_size_h[kIndex0];
  int32_t C = image_size_h[kIndex1];
  int32_t D = image_size_h[kIndex2];
  int32_t H = image_size_h[kIndex3];
  int32_t W = image_size_h[kIndex4];
  bool is_valid = (N == theta_shape[kIndex0]) && (N == grid_shape[kIndex0]) && (D == grid_shape[kIndex1]) &&
                  (H == grid_shape[kIndex2]) && (W == grid_shape[kIndex3]);
  if (is_valid) {
    return true;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name << "', "
                  << "the inferred shape of sampling grids (N×D×H×W×3) must match the value of 'size' (N×C×D×H×W). "
                  << "The inferred shape is (" << grid_shape[kIndex0] << ", " << grid_shape[kIndex1] << ", "
                  << grid_shape[kIndex2] << ", " << grid_shape[kIndex3] << ", " << grid_shape[kIndex4] << "), "
                  << "while the value of 'size' is (" << N << ", " << C << ", " << D << ", " << H << ", " << W
                  << ").\n";
    return false;
  }
}
}  // namespace

bool AffineGridGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::AffineGrid>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);
  kernel_name_ = kernel_ptr->name();
  align_corners_ = kernel_ptr->get_align_corners();

  CHECK_KERNEL_INPUTS_NUM(inputs.size(), N_INPUTS, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), N_OUTPUTS, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  data_type_bytes_ = GetTypeByte(TypeIdToType(inputs[0]->GetDtype()));
  return true;
}

void AffineGridGpuKernelMod::ResetResource() noexcept {
  grid_dim_ = AffineGridDim::unknown;
  theta_shape_.clear();
  grid_shape_.clear();
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

bool AffineGridGpuKernelMod::CheckShapeOfInputs(const std::vector<KernelTensorPtr> &inputs) {
  // Case spatial: theta(N, 2, 3) & size(4,)
  // Case volumetric: theta(N, 3, 4) & size(5,)
  theta_shape_ = inputs[0]->GetShapeVector();
  auto size_shape = inputs[1]->GetShapeVector();
  grid_dim_ = AffineGridDim::unknown;
  if (theta_shape_.size() == RANK_THETA && size_shape.size() == RANK_IMAGE_SIZE) {
    if (theta_shape_[kIndex0] > 0 && theta_shape_[kIndex1] == N_ROWS_THETA_4D &&
        theta_shape_[kIndex2] == N_COLS_THETA_4D && size_shape[0] == LEN_IMAGE_SIZE_4D) {
      grid_dim_ = AffineGridDim::spatial;
    } else if (theta_shape_[kIndex0] > 0 && theta_shape_[kIndex1] == N_ROWS_THETA_5D &&
               theta_shape_[kIndex2] == N_COLS_THETA_5D && size_shape[0] == LEN_IMAGE_SIZE_5D) {
      grid_dim_ = AffineGridDim::volumetric;
    }
  }
  if (grid_dim_ == AffineGridDim::unknown) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', "
                  << "the input batch of affine matrices must be with shape of (N×2×3) for 2D or (N×3×4) for 3D, "
                  << "and the target output image size must be N×C×H×W for 2D or N×C×D×H×W for 3D.\n";
    return false;
  }
  return true;
}

bool AffineGridGpuKernelMod::CheckShapeOfOutputs(const std::vector<KernelTensorPtr> &outputs) {
  // Case spatial: grid(N, H, W, 2)
  // Case volumetric: grid(N, D, H, W, 3)
  // Note: We only check the first item of shape, the last item of shape and the length of shape here.
  //       (H, W) or (D, H, W) will be checked in LaunchKernel because it is related to the value of input.
  grid_shape_ = outputs[0]->GetShapeVector();
  if (!(grid_dim_ == AffineGridDim::spatial && grid_shape_.size() == RANK_GRID_4D &&
        grid_shape_[0] == theta_shape_[0] && grid_shape_.back() == kDim2) &&
      !(grid_dim_ == AffineGridDim::volumetric && grid_shape_.size() == RANK_GRID_5D &&
        grid_shape_[0] == theta_shape_[0] && grid_shape_.back() == kDim3)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', "
                  << "the output batch of sampling grids must be with shape of "
                  << "(N×H×W×2) for 2D or (N×D×H×W×3) for 3D.\n";
    return false;
  }
  return true;
}

int AffineGridGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  ResetResource();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), N_INPUTS, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), N_OUTPUTS, kernel_name_);
  // set up input_size_list_ & output_size_list_ if <ops::AffineGrid> infer shape successfully.
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    return ret;
  }
  // Check if the shape of inputs is correct.
  if (!CheckShapeOfInputs(inputs)) {
    return KRET_RESIZE_FAILED;
  }
  // Check if <ops::AffineGrid> infer output shape correctly.
  if (!CheckShapeOfOutputs(outputs)) {
    return KRET_RESIZE_FAILED;
  }
  if (grid_dim_ == AffineGridDim::spatial) {
    // We use cudnn here. However, cudnn does not support align_corners, we transform thetas as an alternative.
    workspace_size_list_ = {input_size_list_[0]};
  } else {
    size_t base_grid_size = std::accumulate(grid_shape_.begin() + 1, grid_shape_.end() - 1, 0L);  // H+W or D+H+W
    size_t wrapped_grid_size = grid_shape_.front() * base_grid_size * grid_shape_.back();  // N*(H+W)*2 or N*(D+H+W)*3
    workspace_size_list_ = {(base_grid_size + wrapped_grid_size) * data_type_bytes_};
  }
  return KRET_OK;
}

template <typename T>
bool AffineGridGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &workspace,
                                          const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (!IsValidShape(grid_shape_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', "
                  << "the shape of output is invalid, since all the inputs are not ready.";
    return false;
  }
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto theta_ptr = GetDeviceAddress<T>(inputs, kIndex0);
  auto image_size_ptr = GetDeviceAddress<int32_t>(inputs, kIndex1);
  auto workspace_ptr = GetDeviceAddress<T>(workspace, kIndex0);
  auto grid_ptr = GetDeviceAddress<T>(outputs, kIndex0);
  if (theta_ptr == nullptr || image_size_ptr == nullptr || workspace_ptr == nullptr || grid_ptr == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the address of output or input is nullptr.";
    return false;
  }
  // Get the target output image size as [N, C, H, W] for 2d or [N, C, D, H, W] for 3d from device memory.
  // Note: We also check output shape here because it is related to the value of input(target output image size).
  if (grid_dim_ == AffineGridDim::spatial) {
    int32_t image_size[kDim4];
    if (!PreLaunchKernel4D(theta_shape_, grid_shape_, image_size_ptr, image_size, cuda_stream, kernel_name_)) {
      return false;
    }
    auto status =
      CalculateAffineGrid4D(theta_ptr, workspace_ptr, grid_ptr, image_size[kIndex0], image_size[kIndex1],
                            image_size[kIndex2], image_size[kIndex3], align_corners_, device_id_, cuda_stream);
    if (status != CUDNN_STATUS_SUCCESS) {
      return false;
    }
  } else if (grid_dim_ == AffineGridDim::volumetric) {
    int32_t image_size[kDim5];
    if (!PreLaunchKernel5D(theta_shape_, grid_shape_, image_size_ptr, image_size, cuda_stream, kernel_name_)) {
      return false;
    }
    auto status = CalculateAffineGrid5D(theta_ptr, workspace_ptr, grid_ptr, image_size[kIndex0], image_size[kIndex1],
                                        image_size[kIndex2], image_size[kIndex3], image_size[kIndex4], align_corners_,
                                        device_id_, cuda_stream);
    CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', grid_dim_ is not properly set in KernelMod::Resize";
    return false;
  }
  return true;
}

std::vector<std::pair<KernelAttr, AffineGridGpuKernelMod::AffineGridFunc>> AffineGridGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
   &AffineGridGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
   &AffineGridGpuKernelMod::LaunchKernel<float>}};

std::vector<KernelAttr> AffineGridGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, AffineGridFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, AffineGrid, AffineGridGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
