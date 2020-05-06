/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "kernel/cpu/mkldnn/matmul_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "kernel/cpu/mkldnn/mkl_kernel_engine.h"
#include "common/utils.h"
#include "device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void MatMulCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<size_t> src_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  std::vector<size_t> weight_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  std::vector<size_t> dst_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);

  if (src_shape.size() != 2 || weight_shape.size() != 2 || dst_shape.size() != 2) {
    MS_LOG(EXCEPTION) << "matmul invalid input size";
  }
  bool trans_a = AnfAlgo::GetNodeAttr<bool>(kernel_node, TRANSPOSE_A);
  bool trans_b = AnfAlgo::GetNodeAttr<bool>(kernel_node, TRANSPOSE_B);
  if (trans_a) {
    trans_a_ = TRANSPOSE_YES;
    dim_m_ = static_cast<dnnl_dim_t>(src_shape[1]);
    dim_k_ = static_cast<dnnl_dim_t>(src_shape[0]);
  } else {
    dim_m_ = static_cast<dnnl_dim_t>(src_shape[0]);
    dim_k_ = static_cast<dnnl_dim_t>(src_shape[1]);
  }
  if (trans_b) {
    trans_b_ = TRANSPOSE_YES;
  }
  dim_n_ = static_cast<dnnl_dim_t>(dst_shape[1]);
}

bool MatMulCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                             const std::vector<kernel::AddressPtr> & /*workspace*/,
                             const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() < 2 || outputs.empty()) {
    MS_LOG(EXCEPTION) << "matmul error input output size!";
  }
  dnnl_dim_t lda = dim_m_;
  if (trans_a_ == TRANSPOSE_NO) {
    lda = dim_k_;
  }
  dnnl_dim_t ldb = dim_k_;
  if (trans_b_ == TRANSPOSE_NO) {
    ldb = dim_n_;
  }
  auto input_a = reinterpret_cast<float *>(inputs[0]->addr);
  auto input_b = reinterpret_cast<float *>(inputs[1]->addr);
  auto output = reinterpret_cast<float *>(outputs[0]->addr);
  (void)dnnl_sgemm(trans_a_, trans_b_, dim_m_, dim_n_, dim_k_, 1.f, input_a, lda, input_b, ldb, 0.f, output, dim_n_);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
