/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "extendrt/delegate/ascend_native/ascend_native_vsl_kernel.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/vector_core/encoder_vector_kernels.h"
#include "extendrt/delegate/ascend_native/ascend_native_kernel_registry.h"
#include "extendrt/delegate/ascend_native/ops/ascend_native_vsl.h"
#include "extendrt/delegate/plugin/tensorrt_executor_plugin.h"
#include "extendrt/kernel/ascend_native/ascend_native_copy_kernel.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/vector_core/copy_cast.h"

namespace mindspore::kernel {
using mindspore::ops::kNameAscendNativeVsl;

int AscendNativeVslKernel::InferShape() {
  for (size_t i = 0; i < C6NUM; i++) {
    out_tensors_[i]->set_data_type(TypeId::kNumberTypeInt32);
  }
  std::vector<int> b_dims = {batch_size_};
  std::vector<int> b_s_dims = {batch_size_, seq_};
  out_tensors_[0]->set_shape(b_dims);
  out_tensors_[C1NUM]->set_shape(b_dims);
  out_tensors_[C2NUM]->set_shape(b_s_dims);
  out_tensors_[C3NUM]->set_shape(b_s_dims);
  out_tensors_[C4NUM]->set_shape(b_dims);
  out_tensors_[C5NUM]->set_shape({Num2});
  return kSuccess;
}

int AscendNativeVslKernel::Prepare() {
  batch_size_ = in_tensors_[0]->shape()[0];
  seq_ = in_tensors_[C1NUM]->shape()[C1NUM];
  if (out_tensors_[0]->data() == nullptr) {
    out_tensors_[0]->set_data(malloc(batch_size_ * sizeof(int)));
    out_tensors_[C1NUM]->set_data(malloc(batch_size_ * sizeof(int)));
    out_tensors_[C4NUM]->set_data(malloc(batch_size_ * sizeof(int)));
    out_tensors_[C5NUM]->set_data(malloc(Num2 * sizeof(int)));
  }
  return kSuccess;
}
int AscendNativeVslKernel::Run() {
  ascend_native::CreateVSLAscendc(in_tensors_[0]->device_data(), in_tensors_[C1NUM]->device_data(),
                                  out_tensors_[0]->device_data(), out_tensors_[C1NUM]->device_data(),
                                  out_tensors_[C2NUM]->device_data(), out_tensors_[C3NUM]->device_data(),
                                  out_tensors_[C4NUM]->device_data(), out_tensors_[C5NUM]->device_data(), batch_size_,
                                  seq_, const_cast<void *>(stream_));
  ascend_native::CopyDTH(out_tensors_[0]->data(), out_tensors_[0]->device_data(),
                         batch_size_ * sizeof(int));  // q_seq_len
  ascend_native::CopyDTH(out_tensors_[C1NUM]->data(), out_tensors_[C1NUM]->device_data(),
                         batch_size_ * sizeof(int));  // kv_seq_len
  ascend_native::CopyDTH(out_tensors_[C4NUM]->data(), out_tensors_[C4NUM]->device_data(),
                         batch_size_ * sizeof(int));  // mode
  ascend_native::CopyDTH(out_tensors_[C5NUM]->data(), out_tensors_[C5NUM]->device_data(),
                         Num2 * sizeof(int));  // token_num
  return kSuccess;
}

int AscendNativeVslKernel::ReSize() { return lite::RET_OK; }
REGISTER_ASCEND_NATIVE_CREATOR(kNameAscendNativeVsl, AscendNativeVslKernel)
}  // namespace mindspore::kernel
