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

#ifndef MINDSPORE_OPS_KERNEL_ASCEND_ACLNN_PYBOOST_IMPL_CUSTOMIZE_STFT_H_
#define MINDSPORE_OPS_KERNEL_ASCEND_ACLNN_PYBOOST_IMPL_CUSTOMIZE_STFT_H_

#include "ir/tensor.h"
#include "ir/value.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
TensorPtr STFTAscendCustomize(const TensorPtr &input_tensor, const TensorPtr &window_tensor, const Int64ImmPtr &n_fft,
                              const Int64ImmPtr &hop_length, const Int64ImmPtr &win_length,
                              const BoolImmPtr &normalized, const BoolImmPtr &onesided,
                              const BoolImmPtr &return_complex);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_OPS_KERNEL_ASCEND_ACLNN_PYBOOST_IMPL_CUSTOMIZE_STFT_H_
