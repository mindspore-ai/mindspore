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
#include "src/runtime/agent/npu/optimizer/npu_add_transform_pass.h"
#include "src/runtime/agent/npu/optimizer/npu_pass_utils.h"
namespace mindspore::lite {
using kernel::KERNEL_ARCH::kNPU;
int NPUAddTransformPass::UpdateNH2NCTransNodePreKernel(kernel::LiteKernel *kernel, kernel::LiteKernel *trans_kernel,
                                                       kernel::LiteKernel *after_kernel) {
  std::vector<kernel::LiteKernel *> out_kernels;

  for (auto out_kernel : kernel->out_kernels()) {
    if (out_kernel == after_kernel) {
      out_kernels.push_back(trans_kernel);
    } else {
      out_kernels.push_back(out_kernel);
    }
  }
  NPUPassUtils::UpdateKernel(kernel, kernel->in_kernels(), out_kernels, kernel->in_tensors(), kernel->out_tensors());
  return RET_OK;
}

int NPUAddTransformPass::InsertNode(const InnerContext *context, std::vector<kernel::LiteKernel *>::iterator it,
                                    std::vector<kernel::LiteKernel *> *all_kernels,
                                    std::vector<Tensor *> *all_tensors) {
  auto kernel = *it;
  for (auto out_kernel : kernel->out_kernels()) {
    if (out_kernel->Type() == schema::PrimitiveType_Nhwc2Nchw) {
      continue;
    }

    std::vector<int> nh2nc_shape = {kernel->out_tensors()[0]->shape()[0], kernel->out_tensors()[0]->shape()[3],
                                    kernel->out_tensors()[0]->shape()[1], kernel->out_tensors()[0]->shape()[2]};
    auto nh2nc_tensor =
      new Tensor(kernel->out_tensors()[0]->data_type(), nh2nc_shape, schema::Format_NHWC, Tensor::VAR);
    std::vector<Tensor *> nh2nc_tensors = {nh2nc_tensor};
    all_tensors->push_back(nh2nc_tensors[0]);

    auto nc2nh_shape = {nh2nc_shape[0], nh2nc_shape[2], nh2nc_shape[3], nh2nc_shape[1]};
    auto nc2nh_tensor = new Tensor(nh2nc_tensor->data_type(), nc2nh_shape, schema::Format_NCHW, Tensor::VAR);
    std::vector<Tensor *> nc2nh_tensors = {nc2nh_tensor};
    all_tensors->push_back(nc2nh_tensors[0]);

    auto nh2nc_name = kernel->name() + "_nh2nc_" + std::to_string(total++);
    auto *nh2nc_kernel = NPUPassUtils::CreateNhwc2NchwKernel(kernel->out_tensors(), nh2nc_tensors, context, nh2nc_name);
    all_kernels->push_back(nh2nc_kernel);
    insert_primitive_.push_back(nh2nc_kernel->GetPrimitive());
    auto nc2nh_name = kernel->name() + "_nc2nh_" + std::to_string(total++);
    auto *nc2nh_kernel = NPUPassUtils::CreateNchw2NhwcKernel(nh2nc_tensors, nc2nh_tensors, context, nc2nh_name);
    all_kernels->push_back(nc2nh_kernel);
    insert_primitive_.push_back(nc2nh_kernel->GetPrimitive());
    NPUPassUtils::UpdateKernel(nh2nc_kernel, {kernel}, {nc2nh_kernel}, kernel->out_tensors(), nh2nc_tensors);
    NPUPassUtils::UpdateKernel(nc2nh_kernel, {nh2nc_kernel}, {out_kernel}, nh2nc_tensors, nc2nh_tensors);
    UpdateNH2NCTransNodePreKernel(kernel, nh2nc_kernel, out_kernel);
    UpdateNC2NHTransNodeAfterKernel(kernel, nc2nh_kernel, out_kernel);
  }
  return RET_OK;
}

int NPUAddTransformPass::UpdateNC2NHTransNodeAfterKernel(kernel::LiteKernel *kernel, kernel::LiteKernel *trans_kernel,
                                                         kernel::LiteKernel *next_kernel) {
  std::vector<Tensor *> next_in_tensors;
  for (auto next_in_tensor : next_kernel->in_tensors()) {
    if (next_in_tensor != kernel->out_tensors()[0]) {
      next_in_tensors.push_back(next_in_tensor);
    } else {
      next_in_tensors.push_back(trans_kernel->out_tensors()[0]);
    }
  }
  next_kernel->set_in_tensors(next_in_tensors);
  std::vector<kernel::LiteKernel *> next_in_kernels;
  for (auto in_kernel : next_kernel->in_kernels()) {
    if (in_kernel == kernel) {
      next_in_kernels.push_back(trans_kernel);
    } else {
      next_in_kernels.push_back(in_kernel);
    }
  }
  NPUPassUtils::UpdateKernel(next_kernel, next_in_kernels, next_kernel->out_kernels(), next_in_tensors,
                             next_kernel->out_tensors());
  return RET_OK;
}

int NPUAddTransformPass::Run() {
  if (context_->IsNpuEnabled()) {
    std::vector<kernel::LiteKernel *> new_kernels;

    for (auto it = all_kernels_->begin(); it != all_kernels_->end(); it++) {
      auto kernel = *it;
      new_kernels.push_back(kernel);
      if (kernel->desc().arch != kNPU) {
        continue;
      }
      if (kernel->Type() == schema::PrimitiveType_Add && kernel->out_kernels().size() >= 2) {
        int sum = 0;
        for (auto i : kernel->out_kernels()) {
          if (i->Type() == schema::PrimitiveType_Nhwc2Nchw) {
            sum++;
          }
        }
        if (kernel->out_kernels().size() != sum) {
          InsertNode(context_, it, &new_kernels, all_tensors_);
        }
      }
    }

    all_kernels_->clear();
    for (int i = 0; i < new_kernels.size(); i++) {
      all_kernels_->push_back(new_kernels[i]);
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
