/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/stft_aclnn_kernel.h"
#include <vector>
#include "ir/tensor.h"
#include "abstract/ops/primitive_infer_map.h"
#include "infer/ops_func_impl/stft.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"

namespace mindspore {
namespace kernel {
namespace stft {
namespace {
constexpr size_t kSTFTExtInputIdx = 0;
constexpr size_t kSTFTExtWindowIdx = 1;
constexpr size_t kSTFTExtNFftIdx = 2;
constexpr size_t kSTFTExtHopLengthIdx = 3;
constexpr size_t kSTFTExtWinLengthIdx = 4;
constexpr size_t kSTFTExtNormalizedIdx = 5;
constexpr size_t kSTFTExtOnesidedIdx = 6;
constexpr size_t kSTFTExtReturnComplexIdx = 7;
}  // namespace

void STFTAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  ClearOpsWorkSpaceList();
  n_fft_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kSTFTExtNFftIdx]);
  hop_length_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kSTFTExtHopLengthIdx]);
  win_length_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kSTFTExtWinLengthIdx]);
  normalized_ = device::ascend::ConvertKernelTensor<bool>(inputs[kSTFTExtNormalizedIdx]);
  onesided_ = device::ascend::ConvertKernelTensor<bool>(inputs[kSTFTExtOnesidedIdx]);
  return_complex_ = device::ascend::ConvertKernelTensor<bool>(inputs[kSTFTExtReturnComplexIdx]);

  common_dtype_ = ops::GetSTFTCommonDtype(inputs[kSTFTExtInputIdx]->dtype_id(), inputs[kSTFTExtWindowIdx]->dtype_id());
  cast_input_ = inputs[kSTFTExtInputIdx]->dtype_id() != common_dtype_;
  cast_window_ = inputs[kSTFTExtWindowIdx]->dtype_id() != common_dtype_;

  auto input = inputs[kSTFTExtInputIdx];
  if (cast_input_) {
    input_cast_.SetType(std::make_shared<TensorType>(TypeIdToType(common_dtype_)));
    input_cast_.SetShape(std::make_shared<abstract::TensorShape>(input->GetShapeVector()));
    GetWorkspaceForResizeCastInput(input, common_dtype_, &input_cast_);
    input = &input_cast_;
  }

  auto window = inputs[kSTFTExtWindowIdx];
  if (cast_window_) {
    window_cast_.SetType(std::make_shared<TensorType>(TypeIdToType(common_dtype_)));
    window_cast_.SetShape(std::make_shared<abstract::TensorShape>(window->GetShapeVector()));
    GetWorkspaceForResizeCastWindow(window, common_dtype_, &window_cast_);
    window = &window_cast_;
  }

  GetWorkspaceForResizeSTFT(input, window, outputs[kIndex0], n_fft_, hop_length_, win_length_, normalized_, onesided_,
                            return_complex_);
  if (cast_input_) {
    workspace_size_list_.emplace_back(
      ops::CalOutputSize(input_cast_.GetShapeVector(), abstract::TypeIdSize(common_dtype_)));
  }
  if (cast_window_) {
    workspace_size_list_.emplace_back(
      ops::CalOutputSize(window_cast_.GetShapeVector(), abstract::TypeIdSize(common_dtype_)));
  }
}

bool STFTAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  const size_t cast_count = static_cast<size_t>(cast_input_) + static_cast<size_t>(cast_window_);
  size_t workspace_offset = workspace.size() - cast_count;
  auto input = inputs[kSTFTExtInputIdx];
  if (cast_input_) {
    input_cast_.set_device_ptr(workspace[workspace_offset++]->device_ptr());
    RunOpCastInput(stream_ptr, workspace, input, common_dtype_, &input_cast_);
    input = &input_cast_;
  }

  auto window = inputs[kSTFTExtWindowIdx];
  if (cast_window_) {
    window_cast_.set_device_ptr(workspace[workspace_offset]->device_ptr());
    RunOpCastWindow(stream_ptr, workspace, window, common_dtype_, &window_cast_);
    window = &window_cast_;
  }

  RunOpSTFT(stream_ptr, workspace, input, window, outputs[kIndex0], n_fft_, hop_length_, win_length_, normalized_,
            onesided_, return_complex_);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(STFT, STFTAscend);
}  // namespace stft
}  // namespace kernel
}  // namespace mindspore
