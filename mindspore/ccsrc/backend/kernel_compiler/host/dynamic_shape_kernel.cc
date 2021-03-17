/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/host/dynamic_shape_kernel.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
void DynamicShapeKernel::Execute() {
  MS_LOG(INFO) << "Execute DynamicShapeKernel Start";
  auto cnode = cnode_ptr_.lock();
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_num = AnfAlgo::GetInputTensorNum(cnode);
  if (input_num != 1) {
    MS_LOG(EXCEPTION) << "Invalid Input Num:" << input_num;
  }

  auto prev_output_shape = AnfAlgo::GetPrevNodeOutputInferShape(cnode, 0);
  std::vector<int64_t> output_shape = {SizeToLong(prev_output_shape.size())};

  auto output_type = TypeId::kNumberTypeInt64;

  auto output_tensor_for_sync = std::make_shared<tensor::Tensor>(output_type, output_shape);
  auto data_ptr = static_cast<int64_t *>(output_tensor_for_sync->data_c());
  for (size_t i = 0; i < prev_output_shape.size(); ++i) {
    MS_LOG(INFO) << "DEBUG prev_output_shape[" << i << "]:" << prev_output_shape[i];
    *(data_ptr + i) = prev_output_shape[i];
  }

  auto output_addr = AnfAlgo::GetOutputAddr(cnode, 0);
  MS_EXCEPTION_IF_NULL(output_addr);
  output_addr->SyncHostToDevice(output_shape, LongToSize(output_tensor_for_sync->data().nbytes()),
                                output_tensor_for_sync->data_type(), output_tensor_for_sync->data_c());
  MS_LOG(INFO) << "Execute DynamicShapeKernel End";
}

device::DynamicKernelPtr DynamicShapeKernelMod::GenDynamicKernel(const CNodePtr &cnode_ptr, void *stream_ptr) {
  return std::make_shared<DynamicShapeKernel>(stream_ptr, cnode_ptr);
}
}  // namespace kernel
}  // namespace mindspore
