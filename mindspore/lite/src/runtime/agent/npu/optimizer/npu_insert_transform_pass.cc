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
#include <string>
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

int NPUInsertTransformPass::InsertNode(kernel::LiteKernel *kernel, kernel::LiteKernel *post_kernel,
                                       std::vector<kernel::LiteKernel *> *trans_kernels) {
  // Kernel and post_kernel can't be nullptr at the same time.
  std::string kernel_name;
  Tensor *in_tensor = nullptr;

  std::vector<kernel::LiteKernel *> out_kernels;
  // If post_kernel equals nullptr, kernel is the output of whole graph.
  if (post_kernel != nullptr) {
    out_kernels.push_back(post_kernel);
    kernel_name = post_kernel->name() + "_pre";
    in_tensor = post_kernel->in_tensors()[0];
  }
  std::vector<kernel::LiteKernel *> in_kernels;
  // If kernel equals nullptr, post_kernel is the input of whole graph.
  if (kernel != nullptr) {
    in_kernels.push_back(kernel);
    kernel_name = kernel->name() + "_post";
    in_tensor = kernel->out_tensors()[0];
  }
  std::vector<int> nhwc_shape = in_tensor->shape();
  std::vector<int> nchw_shape = {nhwc_shape[0], nhwc_shape[3], nhwc_shape[1], nhwc_shape[2]};

  auto nh2nc_tensor = new (std::nothrow) Tensor(in_tensor->data_type(), nchw_shape, schema::Format_NHWC, Tensor::VAR);
  if (nh2nc_tensor == nullptr) {
    MS_LOG(ERROR) << "New nchw tensor failed when inserting nchw2nhwc kernel.";
    return RET_ERROR;
  }
  std::vector<Tensor *> nh2nc_tensors = {nh2nc_tensor};
  all_tensors_->push_back(nh2nc_tensors[0]);

  auto nc2nh_tensor = new (std::nothrow) Tensor(in_tensor->data_type(), nhwc_shape, schema::Format_NCHW, Tensor::VAR);
  if (nc2nh_tensor == nullptr) {
    MS_LOG(ERROR) << "New nhwc tensor failed when inserting nhwc2nchw kernel.";
    return RET_ERROR;
  }
  std::vector<Tensor *> nc2nh_tensors = {nc2nh_tensor};
  all_tensors_->push_back(nc2nh_tensors[0]);

  auto nh2nc_name = kernel_name + "_nh2nc_" + std::to_string(total++);
  auto *nh2nc_kernel = NPUPassUtils::CreateNhwc2NchwKernel({in_tensor}, nh2nc_tensors, context_, nh2nc_name);
  trans_kernels->push_back(nh2nc_kernel);
  insert_primitive_.push_back(nh2nc_kernel->GetPrimitive());

  auto nc2nh_name = kernel_name + "_nc2nh_" + std::to_string(total++);
  auto *nc2nh_kernel = NPUPassUtils::CreateNchw2NhwcKernel(nh2nc_tensors, nc2nh_tensors, context_, nc2nh_name);
  trans_kernels->push_back(nc2nh_kernel);
  insert_primitive_.push_back(nc2nh_kernel->GetPrimitive());

  NPUPassUtils::UpdateKernel(nh2nc_kernel, in_kernels, {nc2nh_kernel}, {in_tensor}, nh2nc_tensors);
  NPUPassUtils::UpdateKernel(nc2nh_kernel, {nh2nc_kernel}, out_kernels, nh2nc_tensors, nc2nh_tensors);
  if (kernel != nullptr) {
    NPUPassUtils::UpdateNH2NCTransNodePreKernel(kernel, nh2nc_kernel, post_kernel);
  }
  if (post_kernel != nullptr) {
    NPUPassUtils::UpdateNC2NHTransNodePostKernel(kernel, nc2nh_kernel, post_kernel);
  }
  return RET_OK;
}

int NPUInsertTransformPass::InsertPreNodes(kernel::LiteKernel *kernel,
                                           std::vector<kernel::LiteKernel *> *trans_kernels) {
  if (kernel->in_kernels().size() != kernel->in_tensors().size()) {
    MS_LOG(DEBUG) << "The input tensors of kernel may be the input of whole graph or const tensor.";
    return RET_OK;
  }
  if (kernel->in_kernels().empty()) {
    auto ret = InsertNode(nullptr, kernel, trans_kernels);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert nhwc2nchw kernel and nchw2nhwc kernel before kernel " << kernel->name() << " failed.";
      return RET_ERROR;
    }
  }
  for (auto in_kernel : kernel->in_kernels()) {
    if (NPUPassUtils::IsNchw2Nhwc(in_kernel)) {
      continue;
    }
    auto ret = InsertNode(in_kernel, kernel, trans_kernels);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert nhwc2nchw kernel and nchw2nhwc kernel before kernel " << kernel->name() << " failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int NPUInsertTransformPass::InsertPostNodes(kernel::LiteKernel *kernel,
                                            std::vector<kernel::LiteKernel *> *trans_kernels) {
  if (kernel->out_kernels().empty()) {
    auto ret = InsertNode(kernel, nullptr, trans_kernels);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert nhwc2nchw kernel and nchw2nhwc kernel after kernel " << kernel->name() << " failed.";
      return RET_ERROR;
    }
  }
  for (auto out_kernel : kernel->out_kernels()) {
    if (NPUPassUtils::IsNhwc2Nchw(out_kernel)) {
      continue;
    }
    auto ret = InsertNode(kernel, out_kernel, trans_kernels);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert nhwc2nchw kernel and nchw2nhwc kernel after kernel " << kernel->name() << " failed.";
      return RET_ERROR;
    }
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
    // If the every output kernel is nhwc2nchw, insert
    // modify loop index add post_kernels.size() to the next kernel in the origin vector
    if (insert_state == PreInsert) {
      std::vector<kernel::LiteKernel *> pre_kernels;
      auto ret = InsertPreNodes(kernel, &pre_kernels);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Insert nhwc2nchw kernel and nchw2nhwc kernel before kernel " << kernel->name() << " failed.";
        return RET_ERROR;
      }
      all_kernels_->insert(all_kernels_->begin() + i, pre_kernels.begin(), pre_kernels.end());
      i += pre_kernels.size();
    }

    if (insert_state == PostInsert) {
      std::vector<kernel::LiteKernel *> post_kernels;
      auto ret = InsertPostNodes(kernel, &post_kernels);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Insert nhwc2nchw kernel and nchw2nhwc kernel after kernel " << kernel->name() << " failed.";
        return RET_ERROR;
      }
      all_kernels_->insert(all_kernels_->begin() + i + 1, post_kernels.begin(), post_kernels.end());
      i += post_kernels.size();
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
