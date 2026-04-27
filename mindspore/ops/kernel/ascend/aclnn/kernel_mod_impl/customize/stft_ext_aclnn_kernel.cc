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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/stft_ext_aclnn_kernel.h"
#include <vector>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"

namespace mindspore {
namespace kernel {
namespace stft_ext {
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

void STFTExtAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  n_fft_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kSTFTExtNFftIdx]);
  hop_length_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kSTFTExtHopLengthIdx]);
  win_length_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kSTFTExtWinLengthIdx]);
  normalized_ = device::ascend::ConvertKernelTensor<bool>(inputs[kSTFTExtNormalizedIdx]);
  onesided_ = device::ascend::ConvertKernelTensor<bool>(inputs[kSTFTExtOnesidedIdx]);
  return_complex_ = device::ascend::ConvertKernelTensor<bool>(inputs[kSTFTExtReturnComplexIdx]);

  GetWorkspaceForResize(inputs[kSTFTExtInputIdx], inputs[kSTFTExtWindowIdx], outputs[kIndex0], n_fft_, hop_length_,
                        win_length_, normalized_, onesided_, return_complex_);
}

bool STFTExtAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                           const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kSTFTExtInputIdx], inputs[kSTFTExtWindowIdx], outputs[kIndex0], n_fft_,
        hop_length_, win_length_, normalized_, onesided_, return_complex_);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(STFTExt, STFTExtAscend);
}  // namespace stft_ext
}  // namespace kernel
}  // namespace mindspore
