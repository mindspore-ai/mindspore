/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include <set>
#include <string>
#include "src/runtime/agent/npu/optimizer/npu_pass_utils.h"

namespace mindspore::lite {
using kernel::KERNEL_ARCH::kNPU;
enum InsertState { InsertNone, PreInsert, PostInsert, BothInsert };
std::set<mindspore::schema::PrimitiveType> npu_insert_nodes = {
  schema::PrimitiveType_Concat,      schema::PrimitiveType_AddFusion, schema::PrimitiveType_Eltwise,
  schema::PrimitiveType_Activation,  schema::PrimitiveType_Split,     schema::PrimitiveType_PadFusion,
  schema::PrimitiveType_StridedSlice};

// this pass goal is to minimize subgraphs generated
// by inserting nchw2nhwc or nhwc2nchw before or after the operator (e.g. concat, add, etc..) together with
// fusion pass. If transpose inserted are more than half of input output, we will insert remaining input
// output with transpose and hopefully do a fusion pass. Otherwise, we don't insert anything.
//
// Typically concat accept output from nchw2nhwc, we fill other input with nh2nc and nc2nh so that inputs to concat are
// format same and then fusion all nchw2nhwc op.
// e.g.
// original     (conv->nchw2nhwc, add(format nhwc)) -> concat-> (nhwc2nchw->conv)
// current pass (conv->nchw2nhwc, add->nhwc2nchw->nchw2nhwc) -> concat -> (nhwc2nchw->conv)
// fusion pass  (conv, add->nhwc2nchw) -> concat -> conv
// original 2 cpusubgraph, after 2 pass, only 1 cpu subgraph
//
// node:
// Such ops require inputs all have same format, could be nchw or nhwc or other format.
// Their inputs outputs may not be 4d, or are already format ok,
// so we won't insert nc2nh or nh2nc when op's in kernels and out kernels contains no nc2nh or nh2nc.
// This pass should be run after npu_transform_pass, which insert transpose for nchw-input-limited op like conv2d.

int NPUInsertTransformPass::GetInsertState(kernel::LiteKernel *kernel) {
  // filter out irrelevant kernel
  if (npu_insert_nodes.find(kernel->Type()) == npu_insert_nodes.end()) {
    return InsertNone;
  }

  // current kernel is target kernel
  // use out kernels to count how many out lines from current kernel
  std::vector<Tensor *> in_tensors = NPUPassUtils::GetNonConstInputs(kernel);
  size_t in_out_tensor_num =
    in_tensors.size() +
    std::max(std::max(kernel->out_kernels().size(), static_cast<size_t>(1)), kernel->out_tensors().size());
  size_t transpose_input_num = 0;
  size_t transpose_output_num = 0;
  bool need_pre_insert = false;
  bool need_post_insert = false;
  // count number of input tensor from nc2nh and output tensor to nh2nc
  for (size_t i = 0; i < in_tensors.size(); ++i) {
    auto in_kernel = NPUPassUtils::KernelInputFromKernel(kernel, in_tensors.at(i));
    if (NPUPassUtils::IsNchw2Nhwc(in_kernel)) {
      transpose_input_num++;
    } else {
      need_pre_insert = true;
    }
  }
  if (kernel->out_kernels().empty()) {
    need_post_insert = true;
  }
  for (const auto out_kernel : kernel->out_kernels()) {
    if (NPUPassUtils::IsNhwc2Nchw(out_kernel)) {
      transpose_output_num++;
    } else {
      need_post_insert = true;
    }
  }

  // won't insert any thing if num of transpose tensor is smaller than half of total input output.
  // won't insert if total input output are all transpose tensor, the fusion pass will handle this.
  size_t transpose_tensor_num = transpose_input_num + transpose_output_num;
  if (transpose_tensor_num == 0 || transpose_tensor_num * 2 < in_out_tensor_num ||
      transpose_tensor_num == in_out_tensor_num) {
    return InsertNone;
  }
  InsertState ret;
  if (need_pre_insert && !need_post_insert) {
    ret = PreInsert;
  } else if (need_pre_insert && need_post_insert) {
    ret = BothInsert;
  } else if (!need_pre_insert && need_post_insert) {
    ret = PostInsert;
  } else {
    ret = InsertNone;
  }

  return ret;
}

int NPUInsertTransformPass::InsertNode(kernel::LiteKernel *kernel, kernel::LiteKernel *post_kernel,
                                       size_t post_input_index, std::vector<kernel::LiteKernel *> *trans_kernels) {
  // Kernel and post_kernel can't be nullptr at the same time.
  std::string kernel_name;
  Tensor *in_tensor = nullptr;

  std::vector<kernel::LiteKernel *> out_kernels;
  // If post_kernel equals nullptr, kernel is the output of whole graph.
  if (post_kernel != nullptr) {
    out_kernels.push_back(post_kernel);
    kernel_name = post_kernel->name() + "_pre";
    in_tensor = post_kernel->in_tensors().at(post_input_index);
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

  auto nh2nc_name = kernel_name + "_nh2nc_" + std::to_string(total++);
  auto nh2nc_tensor = new (std::nothrow) Tensor(in_tensor->data_type(), nchw_shape, schema::Format_NCHW, Tensor::VAR);
  if (nh2nc_tensor == nullptr) {
    MS_LOG(ERROR) << "New nchw tensor failed when inserting nchw2nhwc kernel.";
    return RET_ERROR;
  }
  nh2nc_tensor->set_tensor_name(nh2nc_name + "/output0");
  std::vector<Tensor *> nh2nc_tensors = {nh2nc_tensor};
  all_tensors_->push_back(nh2nc_tensors[0]);

  auto nc2nh_name = kernel_name + "_nc2nh_" + std::to_string(total++);
  auto nc2nh_tensor = new (std::nothrow) Tensor(in_tensor->data_type(), nhwc_shape, schema::Format_NHWC, Tensor::VAR);
  if (nc2nh_tensor == nullptr) {
    MS_LOG(ERROR) << "New nhwc tensor failed when inserting nhwc2nchw kernel.";
    return RET_ERROR;
  }
  nc2nh_tensor->set_tensor_name(nc2nh_name + "/output0");
  std::vector<Tensor *> nc2nh_tensors = {nc2nh_tensor};
  all_tensors_->push_back(nc2nh_tensors[0]);

  auto *nh2nc_kernel = NPUPassUtils::CreateNhwc2NchwKernel({in_tensor}, nh2nc_tensors, context_, nh2nc_name);
  trans_kernels->push_back(nh2nc_kernel);

  auto *nc2nh_kernel = NPUPassUtils::CreateNchw2NhwcKernel(nh2nc_tensors, nc2nh_tensors, context_, nc2nh_name);
  trans_kernels->push_back(nc2nh_kernel);

  auto nh2nc_perm_tensor = new Tensor(kNumberTypeInt32, {4}, schema::Format_NHWC, Tensor::CONST_TENSOR);
  auto nh2nc_data = nh2nc_perm_tensor->MutableData();
  if (nh2nc_data == nullptr) {
    return RET_ERROR;
  }
  std::vector<int> nh2nc_perm_vector = {0, 3, 1, 2};
  memcpy(nh2nc_data, nh2nc_perm_vector.data(), 4 * sizeof(int));
  all_tensors_->push_back(nh2nc_perm_tensor);

  auto nc2nh_perm_tensor = new Tensor(kNumberTypeInt32, {4}, schema::Format_NHWC, Tensor::CONST_TENSOR);
  auto nc2nh_data = nc2nh_perm_tensor->MutableData();
  if (nc2nh_data == nullptr) {
    return RET_ERROR;
  }

  std::vector<int> nc2nh_perm_vector = {0, 2, 3, 1};
  memcpy(nc2nh_data, nc2nh_perm_vector.data(), 4 * sizeof(int));
  all_tensors_->push_back(nc2nh_perm_tensor);

  NPUPassUtils::UpdateKernel(nh2nc_kernel, in_kernels, {nc2nh_kernel}, {in_tensor, nh2nc_perm_tensor}, nh2nc_tensors);
  NPUPassUtils::UpdateKernel(nc2nh_kernel, {nh2nc_kernel}, out_kernels, {nh2nc_tensors[0], nc2nh_perm_tensor},
                             nc2nh_tensors);
  if (kernel != nullptr) {
    NPUPassUtils::UpdateNH2NCTransNodePreKernel(kernel, nh2nc_kernel, post_kernel);
  }
  if (post_kernel != nullptr) {
    NPUPassUtils::UpdateNC2NHTransNodePostKernel(kernel, nc2nh_kernel, post_kernel);
  } else {
    // post_kernel nullptr mean output, we remain graph output tensor name unchanged
    auto graph_output_name = in_tensor->tensor_name();
    in_tensor->set_tensor_name(graph_output_name + "_before_" + name_);
    nc2nh_tensor->set_tensor_name(graph_output_name);
  }
  return RET_OK;
}

int NPUInsertTransformPass::InsertForInputTensor(kernel::LiteKernel *kernel, size_t in_tensor_index,
                                                 kernel::LiteKernel *pre_kernel,
                                                 std::vector<kernel::LiteKernel *> *trans_kernels) {
  // insert transpose nodes before target ops
  return InsertNode(pre_kernel, kernel, in_tensor_index, trans_kernels);
}

int NPUInsertTransformPass::InsertForOutputTensor(kernel::LiteKernel *kernel, kernel::LiteKernel *post_kernel,
                                                  size_t post_in_tensor_index,
                                                  std::vector<kernel::LiteKernel *> *trans_kernels) {
  // insert transpose nodes after target ops
  return InsertNode(kernel, post_kernel, post_in_tensor_index, trans_kernels);
}

int NPUInsertTransformPass::InsertPreNodes(kernel::LiteKernel *kernel,
                                           std::vector<kernel::LiteKernel *> *trans_kernels) {
  int ret = RET_OK;
  auto in_tensors = NPUPassUtils::GetNonConstInputs(kernel);
  for (auto tensor : in_tensors) {
    auto pre_kernel = NPUPassUtils::KernelInputFromKernel(kernel, tensor);
    if (NPUPassUtils::IsNchw2Nhwc(pre_kernel)) {
      continue;
    }
    // if this tensor is input of graph, pre_kernel is nullptr.
    auto it = find(kernel->in_tensors().begin(), kernel->in_tensors().end(), tensor);
    if (it == kernel->in_tensors().end()) {
      MS_LOG(ERROR) << "Find in tensor index error";
      return RET_ERROR;
    }
    size_t index = it - kernel->in_tensors().begin();
    ret = InsertForInputTensor(kernel, index, pre_kernel, trans_kernels);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert nhwc2nchw kernel and nchw2nhwc kernel before kernel " << kernel->name() << " failed.";
      return ret;
    }
  }
  return ret;
}

int NPUInsertTransformPass::InsertPostNodes(kernel::LiteKernel *kernel,
                                            std::vector<kernel::LiteKernel *> *trans_kernels) {
  int ret = RET_OK;

  for (const auto post_kernel : kernel->out_kernels()) {
    if (NPUPassUtils::IsNhwc2Nchw(post_kernel)) {
      continue;
    }
    auto post_kernel_in_tensors = post_kernel->in_tensors();
    // kernel's out tensor is one of post_kernel's input tensor
    auto it = std::find(post_kernel_in_tensors.begin(), post_kernel_in_tensors.end(), kernel->out_tensors().at(0));
    if (it == post_kernel_in_tensors.end()) {
      return RET_ERROR;
    }
    size_t input_index = it - post_kernel_in_tensors.begin();
    ret = InsertForOutputTensor(kernel, post_kernel, input_index, trans_kernels);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert nhwc2nchw kernel and nchw2nhwc kernel after kernel " << kernel->name() << " failed.";
      return ret;
    }
  }
  if (kernel->out_tensors().size() > kernel->out_kernels().size()) {
    // kernel out is graph output
    ret = InsertForOutputTensor(kernel, nullptr, 0, trans_kernels);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert nhwc2nchw kernel and nchw2nhwc kernel after kernel " << kernel->name() << " failed.";
      return ret;
    }
  }
  return ret;
}

int NPUInsertTransformPass::Run() {
  std::vector<kernel::LiteKernel *> insert_kernels;
  for (int j = 0; j < 2; ++j) {
    for (size_t i = 0; i < all_kernels_->size(); i++) {
      auto kernel = (*all_kernels_)[i];
      if (kernel->desc().arch != kNPU) {
        continue;
      }
      auto insert_state = GetInsertState(kernel);
      insert_kernels.clear();
      // If the every output kernel is nhwc2nchw, insert
      // modify loop index add post_kernels.size() to the next kernel in the origin vector
      switch (insert_state) {
        case PreInsert: {
          auto ret = InsertPreNodes(kernel, &insert_kernels);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "Insert nhwc2nchw kernel and nchw2nhwc kernel before kernel " << kernel->name()
                          << " failed.";
            return RET_ERROR;
          }
          all_kernels_->insert(all_kernels_->begin() + i, insert_kernels.begin(), insert_kernels.end());
          i += insert_kernels.size();
          break;
        }
        case PostInsert: {
          auto ret = InsertPostNodes(kernel, &insert_kernels);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "Insert nhwc2nchw kernel and nchw2nhwc kernel after kernel " << kernel->name()
                          << " failed.";
            return RET_ERROR;
          }
          all_kernels_->insert(all_kernels_->begin() + i + 1, insert_kernels.begin(), insert_kernels.end());
          i += insert_kernels.size();
          break;
        }
        case BothInsert: {
          auto ret = InsertPreNodes(kernel, &insert_kernels);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "Insert nhwc2nchw kernel and nchw2nhwc kernel before kernel " << kernel->name()
                          << " failed.";
            return RET_ERROR;
          }
          all_kernels_->insert(all_kernels_->begin() + i, insert_kernels.begin(), insert_kernels.end());
          i += insert_kernels.size();

          insert_kernels.clear();
          ret = InsertPostNodes(kernel, &insert_kernels);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "Insert nhwc2nchw kernel and nchw2nhwc kernel after kernel " << kernel->name()
                          << " failed.";
            return RET_ERROR;
          }
          all_kernels_->insert(all_kernels_->begin() + i + 1, insert_kernels.begin(), insert_kernels.end());
          i += insert_kernels.size();
          break;
        }
        default:
          MS_LOG(DEBUG) << "Insert Nothing on kernel " << kernel->name();
      }
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
