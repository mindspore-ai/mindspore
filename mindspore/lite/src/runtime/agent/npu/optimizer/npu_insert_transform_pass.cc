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
#include "src/runtime/agent/npu/optimizer/npu_insert_transform_pass.h"
#include <set>
#include "src/runtime/agent/npu/optimizer/npu_pass_utils.h"

namespace mindspore::lite {
using kernel::KERNEL_ARCH::kNPU;
enum InsertState { InsertNone, PreInsert, PostInsert };

std::set<mindspore::schema::PrimitiveType> npu_insert_nodes = {schema::PrimitiveType_Concat, schema::PrimitiveType_Add,
                                                               schema::PrimitiveType_Eltwise,
                                                               schema::PrimitiveType_Activation};

int GetInsertState(kernel::LiteKernel *kernel) {
  if (npu_insert_nodes.find(kernel->Type()) == npu_insert_nodes.end()) {
    return InsertNone;
  }
  auto pre_flag = std::all_of(kernel->in_kernels().begin(), kernel->in_kernels().end(),
                              [](const kernel::LiteKernel *kernel) { return NPUPassUtils::IsNchw2Nhwc(kernel); });
  auto post_flag = std::all_of(kernel->out_kernels().begin(), kernel->out_kernels().end(),
                               [](const kernel::LiteKernel *kernel) { return NPUPassUtils::IsNhwc2Nchw(kernel); });
  if (pre_flag && !post_flag) {
    return PostInsert;
  }
  if (!pre_flag && post_flag) {
    return PreInsert;
  }
  return InsertNone;
}

int NPUInsertTransformPass::InsertPreNode(const InnerContext *context, kernel::LiteKernel *kernel,
                                          std::vector<kernel::LiteKernel *> *trans_kernels,
                                          std::vector<Tensor *> *all_tensors) {
  for (auto in_kernel : kernel->in_kernels()) {
    if (NPUPassUtils::IsNchw2Nhwc(in_kernel)) {
      continue;
    }
    auto nhwc_shape = in_kernel->out_tensors()[0]->shape();
    std::vector<int> nchw_shape = {nhwc_shape[0], nhwc_shape[3], nhwc_shape[1], nhwc_shape[2]};

    auto nh2nc_tensor =
      new Tensor(in_kernel->out_tensors()[0]->data_type(), nchw_shape, schema::Format_NHWC, Tensor::VAR);
    std::vector<Tensor *> nh2nc_tensors = {nh2nc_tensor};
    all_tensors->push_back(nh2nc_tensors[0]);

    auto nc2nh_tensor = new Tensor(nh2nc_tensor->data_type(), nhwc_shape, schema::Format_NCHW, Tensor::VAR);
    std::vector<Tensor *> nc2nh_tensors = {nc2nh_tensor};
    all_tensors->push_back(nc2nh_tensors[0]);

    auto nh2nc_name = in_kernel->name() + "_nh2nc_" + std::to_string(total++);
    auto *nh2nc_kernel =
      NPUPassUtils::CreateNhwc2NchwKernel(in_kernel->out_tensors(), nh2nc_tensors, context, nh2nc_name);
    trans_kernels->push_back(nh2nc_kernel);
    insert_primitive_.push_back(nh2nc_kernel->GetPrimitive());

    auto nc2nh_name = in_kernel->name() + "_nc2nh_" + std::to_string(total++);
    auto *nc2nh_kernel = NPUPassUtils::CreateNchw2NhwcKernel(nh2nc_tensors, nc2nh_tensors, context, nc2nh_name);
    trans_kernels->push_back(nc2nh_kernel);
    insert_primitive_.push_back(nc2nh_kernel->GetPrimitive());

    NPUPassUtils::UpdateKernel(nh2nc_kernel, {in_kernel}, {nc2nh_kernel}, in_kernel->out_tensors(), nh2nc_tensors);
    NPUPassUtils::UpdateKernel(nc2nh_kernel, {nh2nc_kernel}, {kernel}, nh2nc_tensors, nc2nh_tensors);
    NPUPassUtils::UpdateNH2NCTransNodePreKernel(in_kernel, nh2nc_kernel, kernel);
    NPUPassUtils::UpdateNC2NHTransNodeAfterKernel(in_kernel, nc2nh_kernel, kernel);
  }
  return RET_OK;
}

int NPUInsertTransformPass::InsertPostNode(const InnerContext *context, kernel::LiteKernel *kernel,
                                           std::vector<kernel::LiteKernel *> *trans_kernels,
                                           std::vector<Tensor *> *all_tensors) {
  for (auto out_kernel : kernel->out_kernels()) {
    if (NPUPassUtils::IsNhwc2Nchw(out_kernel)) {
      continue;
    }
    auto nhwc_shape = kernel->out_tensors()[0]->shape();
    std::vector<int> nchw_shape = {nhwc_shape[0], nhwc_shape[3], nhwc_shape[1], nhwc_shape[2]};

    auto nh2nc_tensor = new Tensor(kernel->out_tensors()[0]->data_type(), nchw_shape, schema::Format_NHWC, Tensor::VAR);
    std::vector<Tensor *> nh2nc_tensors = {nh2nc_tensor};
    all_tensors->push_back(nh2nc_tensors[0]);

    auto nc2nh_tensor = new Tensor(nh2nc_tensor->data_type(), nhwc_shape, schema::Format_NCHW, Tensor::VAR);
    std::vector<Tensor *> nc2nh_tensors = {nc2nh_tensor};
    all_tensors->push_back(nc2nh_tensors[0]);

    auto nh2nc_name = kernel->name() + "_nh2nc_" + std::to_string(total++);
    auto *nh2nc_kernel = NPUPassUtils::CreateNhwc2NchwKernel(kernel->out_tensors(), nh2nc_tensors, context, nh2nc_name);
    trans_kernels->push_back(nh2nc_kernel);
    insert_primitive_.push_back(nh2nc_kernel->GetPrimitive());

    auto nc2nh_name = kernel->name() + "_nc2nh_" + std::to_string(total++);
    auto *nc2nh_kernel = NPUPassUtils::CreateNchw2NhwcKernel(nh2nc_tensors, nc2nh_tensors, context, nc2nh_name);
    trans_kernels->push_back(nc2nh_kernel);
    insert_primitive_.push_back(nc2nh_kernel->GetPrimitive());

    NPUPassUtils::UpdateKernel(nh2nc_kernel, {kernel}, {nc2nh_kernel}, kernel->out_tensors(), nh2nc_tensors);
    NPUPassUtils::UpdateKernel(nc2nh_kernel, {nh2nc_kernel}, {out_kernel}, nh2nc_tensors, nc2nh_tensors);
    NPUPassUtils::UpdateNH2NCTransNodePreKernel(kernel, nh2nc_kernel, out_kernel);
    NPUPassUtils::UpdateNC2NHTransNodeAfterKernel(kernel, nc2nh_kernel, out_kernel);
  }
  return RET_OK;
}

int NPUInsertTransformPass::Run() {
  for (size_t i = 0; i < all_kernels_->size(); i++) {
    auto kernel = (*all_kernels_)[i];
    if (kernel->desc().arch != kNPU) {
      continue;
    }
    auto insert_state = GetInsertState(kernel);
    if (insert_state == PreInsert) {
      std::vector<kernel::LiteKernel *> pre_kernels;
      InsertPreNode(context_, kernel, &pre_kernels, all_tensors_);
      all_kernels_->insert(all_kernels_->begin() + i, pre_kernels.begin(), pre_kernels.end());
      i += pre_kernels.size();
    }
    if (insert_state == PostInsert) {
      std::vector<kernel::LiteKernel *> post_kernels;
      InsertPostNode(context_, kernel, &post_kernels, all_tensors_);
      all_kernels_->insert(all_kernels_->begin() + i + 1, post_kernels.begin(), post_kernels.end());
      i += post_kernels.size();
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
