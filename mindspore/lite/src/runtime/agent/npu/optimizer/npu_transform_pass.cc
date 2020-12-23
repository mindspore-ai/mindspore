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
#include "src/runtime/agent/npu/optimizer/npu_transform_pass.h"
#include <vector>
#include "src/lite_kernel.h"
#include "src/runtime/agent/npu/npu_manager.h"
#include "src/runtime/agent/npu/optimizer/npu_pass_utils.h"
namespace mindspore::lite {
using kernel::KERNEL_ARCH::kCPU;
using kernel::KERNEL_ARCH::kNPU;
int NPUTransformPass::InsertPreNode(const InnerContext *context, kernel::LiteKernel *kernel,
                                    std::vector<kernel::LiteKernel *> *all_kernels,
                                    std::vector<Tensor *> *all_tensors) {
  bool is_input_kernel = kernel->in_kernels().empty();
  if (is_input_kernel || kernel->in_kernels()[0]->desc().arch != kNPU ||
      npu_trans_nodes.find(kernel->in_kernels()[0]->Type()) == npu_trans_nodes.end()) {
    kernel::LiteKernel *before_kernel = nullptr;
    if (!is_input_kernel) {
      before_kernel = kernel->in_kernels()[0];
    }
    // Create pre transform kernel out tensors.
    auto nhwc_shape = kernel->in_tensors()[0]->shape();
    std::vector<int> nchw_shape = {nhwc_shape[0], nhwc_shape[3], nhwc_shape[1], nhwc_shape[2]};
    auto tensor = new Tensor(kernel->in_tensors()[0]->data_type(), nchw_shape, schema::Format_NCHW, Tensor::VAR);
    std::vector<Tensor *> pre_trans_out_tensors = {tensor};
    all_tensors->push_back(pre_trans_out_tensors[0]);
    // Replace the output tensor of the previous node
    auto name = kernel->name() + "_pre_trans" + "_Nhwc2Nchw_" + std::to_string(total++);
    auto *pre_trans_kernel =
      NPUPassUtils::CreateNhwc2NchwKernel({kernel->in_tensors()[0]}, pre_trans_out_tensors, context, name);
    // Insert Nhwc2Nchw into the front of the current queue
    all_kernels->push_back(pre_trans_kernel);
    insert_primitive_.push_back(pre_trans_kernel->GetPrimitive());
    // Replace the output kernel of the previous node
    std::vector<kernel::LiteKernel *> pre_trans_in_kernel;
    if (is_input_kernel) {
      pre_trans_in_kernel = {};
    } else {
      pre_trans_in_kernel = {before_kernel};
    }
    NPUPassUtils::UpdateKernel(pre_trans_kernel, pre_trans_in_kernel, {kernel}, {kernel->in_tensors()[0]},
                               pre_trans_out_tensors);

    if (before_kernel != nullptr) {
      NPUPassUtils::UpdateNH2NCTransNodePreKernel(before_kernel, pre_trans_kernel, kernel);
    }
    NPUPassUtils::UpdateNH2NCTransNodeAfterKernel(kernel, pre_trans_kernel, before_kernel);
  }
  return RET_OK;
}

int NPUTransformPass::InsertPostNode(const InnerContext *context, kernel::LiteKernel *kernel,
                                     std::vector<kernel::LiteKernel *> *all_kernels,
                                     std::vector<Tensor *> *all_tensors) {
  // Model output does not insert operator
  if (kernel->out_kernels().empty()) {
    return RET_OK;
  }
  // Single output multiple references
  for (int i = 0; i < kernel->out_kernels().size(); i++) {
    auto next_kernel = kernel->out_kernels().at(i);
    if (next_kernel->desc().arch == kNPU && npu_trans_nodes.find(next_kernel->Type()) != npu_trans_nodes.end()) {
      continue;
    }
    // Change format the output of the current kernel nhwc->nchw
    auto tensor = new Tensor(kernel->out_tensors()[0]->data_type(), kernel->out_tensors()[0]->shape(),
                             schema::Format_NHWC, Tensor::VAR);
    std::vector<Tensor *> post_trans_out_tensors = {tensor};
    all_tensors->push_back(post_trans_out_tensors[0]);
    // Use the output tensor of the current node as the input tensor of the post-conversion operator
    auto name = kernel->name() + "_post_trans" + "_Nchw2Nhwc" + std::to_string(total++);
    auto *post_trans_kernel =
      NPUPassUtils::CreateNchw2NhwcKernel(kernel->out_tensors(), post_trans_out_tensors, context, name);
    // Replace the input tensor of the next node
    NPUPassUtils::UpdateKernel(post_trans_kernel, {kernel}, {next_kernel}, kernel->out_tensors(),
                               post_trans_out_tensors);
    insert_primitive_.push_back(post_trans_kernel->GetPrimitive());
    // Directly insert in the back, will not affect the topological sort
    all_kernels->push_back(post_trans_kernel);
    NPUPassUtils::UpdateNC2NHTransNodePreKernel(kernel, post_trans_kernel, next_kernel);
    NPUPassUtils::UpdateNC2NHTransNodeAfterKernel(kernel, post_trans_kernel, next_kernel);
  }
  return RET_OK;
}

int NPUTransformPass::Run() {
  if (!context_->IsNpuEnabled()) {
    return RET_OK;
  }
  for (size_t i = 0; i < all_kernels_->size();) {
    auto kernel = (*all_kernels_)[i];
    if (kernel->desc().arch != kNPU || npu_trans_nodes.find(kernel->Type()) == npu_trans_nodes.end()) {
      i++;
      continue;
    }
    std::vector<kernel::LiteKernel *> pre_kernels;
    InsertPreNode(context_, kernel, &pre_kernels, all_tensors_);
    all_kernels_->insert(all_kernels_->begin() + i, pre_kernels.begin(), pre_kernels.end());
    i += (pre_kernels.size() + 1);

    std::vector<kernel::LiteKernel *> post_kernels;
    InsertPostNode(context_, kernel, &post_kernels, all_tensors_);
    all_kernels_->insert(all_kernels_->begin() + i, post_kernels.begin(), post_kernels.end());
    i += post_kernels.size();
  }
  return RET_OK;
}
}  // namespace mindspore::lite
