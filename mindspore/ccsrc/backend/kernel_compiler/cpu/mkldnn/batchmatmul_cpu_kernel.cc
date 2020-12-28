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
#include "backend/kernel_compiler/cpu/mkldnn/batchmatmul_cpu_kernel.h"
#include <utility>
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "utils/ms_utils.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
bool BatchMatMulCPUKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                  const std::vector<AddressPtr> &outputs) {
  if (inputs.size() < 2 || outputs.empty()) {
    MS_LOG(EXCEPTION) << "batchmatmul error input output size!";
  }

  if (batch_ == 0) {
    MS_LOG(EXCEPTION) << "batchmatmul error batch size!";
  }

  LaunchKernel<float>(inputs, outputs);

  return true;
}

template <typename T>
void BatchMatMulCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  T *input_a = reinterpret_cast<T *>(inputs[0]->addr);
  T *input_b = reinterpret_cast<T *>(inputs[1]->addr);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);

  const int lda = (trans_a_ == TRANSPOSE_YES) ? SizeToInt(dim_m_) : SizeToInt(dim_k_);
  const int ldb = (trans_b_ == TRANSPOSE_YES) ? SizeToInt(dim_k_) : SizeToInt(dim_n_);
  const int ldc = dim_n_;

  const float alpha = 1;
  const float beta = 0;

  for (unsigned int i = 0; i < batch_; i++) {
    (void)dnnl_sgemm(trans_a_, trans_b_, dim_m_, dim_n_, dim_k_, alpha, input_a + i * size_mat_a_, lda,
                     input_b + i * size_mat_b_, ldb, beta, output + i * size_mat_output_, ldc);
  }
}

void BatchMatMulCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<size_t> src_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  std::vector<size_t> weight_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  std::vector<size_t> dst_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);

  if (src_shape.size() < 3 || weight_shape.size() < 3 || dst_shape.size() < 3) {
    MS_LOG(EXCEPTION) << "batchmatmul invalid input size";
  }

  auto dims = dst_shape.size();

  dim_m_ = static_cast<dnnl_dim_t>(dst_shape[dims - 2]);
  dim_n_ = static_cast<dnnl_dim_t>(dst_shape[dims - 1]);

  size_mat_a_ = src_shape[dims - 2] * src_shape[dims - 1];
  size_mat_b_ = weight_shape[dims - 2] * weight_shape[dims - 1];
  size_mat_output_ = dst_shape[dims - 2] * dst_shape[dims - 1];

  bool trans_a = AnfAlgo::GetNodeAttr<bool>(kernel_node, TRANSPOSE_A);
  bool trans_b = AnfAlgo::GetNodeAttr<bool>(kernel_node, TRANSPOSE_B);

  batch_ = 1;
  for (unsigned int i = 0; i < dst_shape.size() - 2; i++) {
    batch_ *= dst_shape[i];
  }

  auto input1_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  dim_k_ = trans_a ? input1_shape[dims - 2] : input1_shape[dims - 1];

  trans_a_ = trans_a ? TRANSPOSE_YES : TRANSPOSE_NO;
  trans_b_ = trans_b ? TRANSPOSE_YES : TRANSPOSE_NO;
}
}  // namespace kernel
}  // namespace mindspore
