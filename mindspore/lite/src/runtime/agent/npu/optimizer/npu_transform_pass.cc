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
using kernel::KERNEL_ARCH::kNPU;
int NPUTransformPass::InsertPreNodes(kernel::LiteKernel *kernel, std::vector<kernel::LiteKernel *> *trans_kernels) {
  bool is_input_kernel = kernel->in_kernels().empty();
  // single input
  if (is_input_kernel || kernel->in_kernels()[0]->desc().arch != kNPU ||
      npu_trans_nodes.find(kernel->in_kernels()[0]->Type()) == npu_trans_nodes.end()) {
    kernel::LiteKernel *pre_kernel = nullptr;
    if (!is_input_kernel) {
      pre_kernel = kernel->in_kernels()[0];
    }

    // Create pre transform kernel's out tensor.
    auto nhwc_shape = kernel->in_tensors()[0]->shape();
    std::vector<int> nchw_shape = {nhwc_shape[0], nhwc_shape[3], nhwc_shape[1], nhwc_shape[2]};
    auto tensor =
      new (std::nothrow) Tensor(kernel->in_tensors()[0]->data_type(), nchw_shape, schema::Format_NCHW, Tensor::VAR);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "New nchw tensor failed when inserting pre nhwc2nchw kernel.";
      return RET_ERROR;
    }
    std::vector<Tensor *> pre_trans_out_tensors = {tensor};
    all_tensors_->push_back(pre_trans_out_tensors[0]);

    // Create pre transform kernel: Nhwc2Nchw
    auto name = kernel->name() + "_pre_trans" + "_Nhwc2Nchw_" + std::to_string(total++);
    auto *trans_kernel =
      NPUPassUtils::CreateNhwc2NchwKernel({kernel->in_tensors()[0]}, pre_trans_out_tensors, context_, name);

    trans_kernels->push_back(trans_kernel);
    insert_primitive_.push_back(trans_kernel->GetPrimitive());

    // Set in_kernels, out_kernels, in_tensors, out_tensors for transform kernel
    std::vector<kernel::LiteKernel *> pre_trans_in_kernels;
    if (!is_input_kernel) {
      pre_trans_in_kernels = {pre_kernel};
    }
    NPUPassUtils::UpdateKernel(trans_kernel, pre_trans_in_kernels, {kernel}, {kernel->in_tensors()[0]},
                               pre_trans_out_tensors);

    if (pre_kernel != nullptr) {
      NPUPassUtils::UpdateNH2NCTransNodePreKernel(pre_kernel, trans_kernel, kernel);
    }
    NPUPassUtils::UpdateNH2NCTransNodePostKernel(trans_kernel, kernel);
  }
  return RET_OK;
}

int NPUTransformPass::InsertPostNodes(kernel::LiteKernel *kernel, std::vector<kernel::LiteKernel *> *trans_kernels) {
  bool is_output_kernel = kernel->out_kernels().empty();
  // Get the post kernel that need insert trans kernel.
  // If no need for inserting trans kernel, the post kernel must be npu and in trans_nodes.
  std::vector<kernel::LiteKernel *> post_insert_kernels;
  for (int i = 0; i < kernel->out_kernels().size(); i++) {
    auto post_kernel = kernel->out_kernels()[i];
    if (post_kernel->desc().arch != kNPU || npu_trans_nodes.find(post_kernel->Type()) == npu_trans_nodes.end()) {
      post_insert_kernels.push_back(post_kernel);
    }
  }
  if (is_output_kernel || !post_insert_kernels.empty()) {
    // Create post transform kernel's in tensor.
    auto nhwc_shape = kernel->out_tensors()[0]->shape();
    std::vector<int> nchw_shape = {nhwc_shape[0], nhwc_shape[3], nhwc_shape[1], nhwc_shape[2]};
    auto tensor =
      new (std::nothrow) Tensor(kernel->out_tensors()[0]->data_type(), nchw_shape, schema::Format_NCHW, Tensor::VAR);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "New nchw tensor failed when inserting post nchw2nhwc kernel.";
      return RET_ERROR;
    }
    std::vector<Tensor *> post_trans_in_tensors = {tensor};
    all_tensors_->push_back(tensor);
    auto name = kernel->name() + "_post_trans" + "_Nchw2Nhwc" + std::to_string(total++);
    tensor->set_tensor_name(name + "/input0");

    // Create post transform kernel: Nchw2Nhwc
    auto *post_trans_kernel =
      NPUPassUtils::CreateNchw2NhwcKernel(post_trans_in_tensors, kernel->out_tensors(), context_, name);

    // Set in_kernels, out_kernels, in_tensors, out_tensors for transform kernel
    NPUPassUtils::UpdateKernel(post_trans_kernel, {kernel}, post_insert_kernels, post_trans_in_tensors,
                               kernel->out_tensors());
    insert_primitive_.push_back(post_trans_kernel->GetPrimitive());
    trans_kernels->push_back(post_trans_kernel);

    if (!is_output_kernel) {
      for (int i = 0; i < kernel->out_kernels().size(); i++) {
        auto post_kernel = kernel->out_kernels()[i];
        if (find(post_insert_kernels.begin(), post_insert_kernels.end(), post_kernel) != post_insert_kernels.end()) {
          NPUPassUtils::UpdateNC2NHTransNodePostKernel(kernel, post_trans_kernel, post_kernel);
        } else {
          NPUPassUtils::UpdateNC2NHPostKernelInTensors(kernel, post_trans_kernel, post_kernel);
        }
      }
    }
    NPUPassUtils::UpdateNC2NHTransNodePreKernel(kernel, post_trans_kernel, post_insert_kernels);
  }
  return RET_OK;
}

int NPUTransformPass::Run() {
  for (size_t i = 0; i < all_kernels_->size();) {
    auto kernel = (*all_kernels_)[i];
    if (kernel->desc().arch != kNPU || npu_trans_nodes.find(kernel->Type()) == npu_trans_nodes.end()) {
      i++;
      continue;
    }
    // insert pre_kernels before kernel in vector
    // modify loop index add (pre_kernels.size() + 1) to the post_kernels insert location
    std::vector<kernel::LiteKernel *> pre_kernels;
    auto ret = InsertPreNodes(kernel, &pre_kernels);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert nhwc2nchw kernel before kernel " << kernel->name() << " failed.";
      return RET_ERROR;
    }
    all_kernels_->insert(all_kernels_->begin() + i, pre_kernels.begin(), pre_kernels.end());
    i += (pre_kernels.size() + 1);

    // insert post_kernels after kernel in vector
    // modify loop index add post_kernels.size() to the next kernel in the origin vector
    std::vector<kernel::LiteKernel *> post_kernels;
    ret = InsertPostNodes(kernel, &post_kernels);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert nchw2nhwc kernel after kernel " << kernel->name() << " failed.";
      return RET_ERROR;
    }
    all_kernels_->insert(all_kernels_->begin() + i, post_kernels.begin(), post_kernels.end());
    i += post_kernels.size();
  }
  return RET_OK;
}
}  // namespace mindspore::lite
