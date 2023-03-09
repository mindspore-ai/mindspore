/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/pass/format_pass/eliminate_transpose.h"
#include "src/litert/kernel_exec_util.h"

namespace mindspore::lite::pass {
int TransFullyFusion(kernel::SubGraphKernel *subgraph, kernel::KernelExec *trans_kernel0,
                     kernel::KernelExec *trans_kernel1) {
  CHECK_NULL_RETURN(trans_kernel0);
  CHECK_NULL_RETURN(trans_kernel1);
  auto in_tensor = trans_kernel0->in_tensors().at(0);

  auto out_tensor = trans_kernel1->out_tensors().at(0);
  auto in_kernel = kernel::KernelExecUtil::FindInKernelForInTensor(trans_kernel0, in_tensor);
  auto out_kernels = kernel::KernelExecUtil::FindOutKernelsForOutTensor(trans_kernel1, out_tensor);
  subgraph->UpdateInOutKernels(in_kernel, out_kernels, trans_kernel0, trans_kernel1);
  auto ret = subgraph->UpdateInOutTensors(in_kernel, out_kernels, in_tensor, out_tensor, true);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Update tensor failed when fusing kernel " << trans_kernel0->name() << " and "
                  << trans_kernel1->name();
    return RET_ERROR;
  }
  subgraph->DropNode(trans_kernel1);
  delete trans_kernel1;
  if (trans_kernel0->out_kernels().empty() && !IsContain(subgraph->out_tensors(), trans_kernel0->out_tensors().at(0))) {
    subgraph->DropNode(trans_kernel0);
    delete trans_kernel0;
  }
  return RET_OK;
}
int TransHeadTailFusion(kernel::SubGraphKernel *subgraph, kernel::KernelExec *trans_kernel0,
                        kernel::KernelExec *trans_kernel1, const TransInfoPair &trans_info) {
  CHECK_NULL_RETURN(trans_kernel0);
  CHECK_NULL_RETURN(trans_kernel1);
  auto ctx = trans_kernel0->Context();
  auto desc = trans_kernel0->desc();
  auto in_tensor = trans_kernel0->in_tensors().at(0);
  auto out_tensor = trans_kernel1->out_tensors().at(0);
  auto in_kernel = kernel::KernelExecUtil::FindInKernelForInTensor(trans_kernel0, in_tensor);
  auto out_kernels = kernel::KernelExecUtil::FindOutKernelsForOutTensor(trans_kernel1, out_tensor);
  subgraph->UpdateInOutKernels(in_kernel, out_kernels, trans_kernel0, trans_kernel1);
  // new trans kernel: src_format -> dst_format
  auto trans_name = trans_kernel0->name() + "_and_" + trans_kernel1->name() + "_fusion";
  auto kernel = CreateFormatTranspose(in_tensor, out_tensor, trans_info, trans_name, ctx, desc);
  CHECK_NULL_RETURN(kernel);
  if (in_kernel != nullptr) {
    in_kernel->AddOutKernel(kernel);
    kernel->AddInKernel(in_kernel);
  }
  for (const auto &out_kernel : out_kernels) {
    if (in_kernel != nullptr) {
      in_kernel->RemoveOutKernel(out_kernel);
      out_kernel->RemoveInKernel(in_kernel);
    }
    out_kernel->AddInKernel(kernel);
    kernel->AddOutKernel(out_kernel);
  }
  subgraph->nodes().push_back(kernel);

  subgraph->DropNode(trans_kernel1);
  delete trans_kernel1;
  if (trans_kernel0->out_kernels().empty() && !IsContain(subgraph->out_tensors(), trans_kernel0->out_tensors().at(0))) {
    subgraph->DropNode(trans_kernel0);
    delete trans_kernel0;
  }
  return RET_OK;
}

int PackConstData(Tensor *tensor, const TransInfoPair &pre_trans) {
  if (tensor->shape().size() != 4) {
    MS_LOG(ERROR) << "Pack const data only valid for 4 dims tensor.";
    return RET_OK;
  }
  auto allocator = tensor->allocator();
  auto original_data = tensor->data();
  auto original_own_data = tensor->own_data();

  auto batch = tensor->Batch();
  auto height = tensor->Height();
  auto width = tensor->Width();
  auto channel = tensor->Channel();
  tensor->set_format(static_cast<mindspore::Format>(pre_trans.dst_format_));
  if (tensor->format() == NHWC) {
    tensor->set_shape({batch, height, width, channel});
  }
  if (tensor->format() == NCHW || tensor->format() == NC4HW4 || tensor->format() == NC8HW8) {
    tensor->set_shape({batch, channel, height, width});
  }
  tensor->set_data(nullptr);

  auto ret = tensor->MallocData();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc new format data failed";
    return ret;
  }

  if (original_own_data) {
    if (allocator != nullptr) {
      allocator->Free(original_data);
    } else {
      free(original_data);
    }
  }
  MS_LOG(ERROR) << "Can't call TransData function.";
  return RET_ERROR;
}

int DoPreFusion(kernel::SubGraphKernel *subgraph, kernel::KernelExec *kernel, std::vector<Tensor *> *all_tensors,
                const TransInfoPair &pre_trans) {
  for (size_t i = 0; i < kernel->in_tensors().size(); i++) {
    auto in_tensor = kernel->in_tensors().at(i);
    auto in_kernel = kernel::KernelExecUtil::FindInKernelForInTensor(kernel, in_tensor);
    if (in_kernel == nullptr) {
      if (in_tensor->data() == nullptr) {
        // graph input, insert an opposite transpose
        auto ret = InsertPreTranspose(subgraph, kernel, all_tensors,
                                      TransInfoPair(pre_trans.dst_format_, pre_trans.src_format_), i);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "Insert previous transpose kernel for op: " << kernel->name() << " input tensor " << i
                        << " failed.";
          return RET_ERROR;
        }
      } else {
        auto ret = PackConstData(in_tensor, pre_trans);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "Pack tensor " << in_tensor->tensor_name() << " data failed.";
          return RET_ERROR;
        }
      }
    } else {
      TransInfoPair in_kernel_trans;
      auto ret = GetTransposeInfo(in_kernel, &in_kernel_trans);
      if (ret != RET_OK || !IsSameTranspose(pre_trans, in_kernel_trans)) {
        // not a transpose kernel that can be fused, insert an opposite transpose
        ret = InsertPreTranspose(subgraph, kernel, all_tensors,
                                 TransInfoPair(pre_trans.dst_format_, pre_trans.src_format_), i);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "Insert previous transpose kernel for op: " << kernel->name() << " input tensor " << i
                        << " failed.";
          return RET_ERROR;
        }
      } else {
        auto pre_in_kernel = kernel::KernelExecUtil::FindInKernelForInTensor(in_kernel, in_kernel->in_tensors().at(0));
        subgraph->UpdateInOutKernels(pre_in_kernel, {kernel}, in_kernel, in_kernel);
        ret = subgraph->UpdateInOutTensors(pre_in_kernel, {kernel}, in_kernel->in_tensors().at(0), in_tensor, true);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "Update tensor failed when removing kernel " << in_kernel->name();
          return RET_ERROR;
        }

        if (in_kernel->out_kernels().empty() && !IsContain(subgraph->out_tensors(), in_kernel->out_tensors().at(0))) {
          subgraph->DropNode(in_kernel);
          delete in_kernel;
        }
      }
    }
  }
  return RET_OK;
}

int DoPostFusion(kernel::SubGraphKernel *subgraph, const kernel::KernelExec *kernel, std::vector<Tensor *> *all_tensors,
                 const TransInfoPair &post_trans) {
  for (size_t i = 0; i < kernel->out_tensors().size(); i++) {
    auto tensor = kernel->out_tensors().at(i);
    auto out_kernels = kernel::KernelExecUtil::FindOutKernelsForOutTensor(kernel, tensor);

    std::vector<kernel::KernelExec *> to_deletes;
    for (const auto &out_kernel : out_kernels) {
      TransInfoPair out_kernel_trans;
      auto ret = GetTransposeInfo(out_kernel, &out_kernel_trans);
      if (ret == RET_OK && IsSameTranspose(post_trans, out_kernel_trans)) {
        (void)to_deletes.emplace_back(out_kernel);
        continue;
      }
      auto in_tensor_of_out_kernel_idx = out_kernel->FindInTensorIndex(tensor);
      ret =
        InsertPreTranspose(subgraph, out_kernel, all_tensors,
                           TransInfoPair(post_trans.dst_format_, post_trans.src_format_), in_tensor_of_out_kernel_idx);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Insert pre transpose kernel for op: " << out_kernel->name() << " input tensor "
                      << in_tensor_of_out_kernel_idx << " failed.";
        return RET_ERROR;
      }
    }
    for (auto &to_delete : to_deletes) {
      auto ret = subgraph->DeleteSingleWayNode(to_delete, false);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Delete kernel: " << to_delete->name() << " failed.";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

int EliminateTranspose::EliminateForSingleKernel(kernel::SubGraphKernel *subgraph, std::vector<Tensor *> *all_tensors) {
  auto kernels = &(subgraph->nodes());
  auto kernel_iter = kernels->begin();
  while (kernel_iter != kernels->end()) {
    auto kernel = *kernel_iter;
    CHECK_NULL_RETURN(kernel);
    TransInfoPair pre_trans;
    TransInfoPair post_trans;
    if (!transpose_strategy_.CheckFusion(kernel, &pre_trans, &post_trans)) {
      (void)kernel_iter++;
      continue;
    }
    auto ret = transpose_strategy_.ChangeKernelAxis(kernel, post_trans);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Change kernel axis " << kernel->name() << " failed.";
      return RET_ERROR;
    }

    graph_changed_ = true;
    ret = DoPreFusion(subgraph, kernel, all_tensors, pre_trans);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Fusion for pre transpose of " << kernel->name() << " failed.";
      return RET_ERROR;
    }
    ret = DoPostFusion(subgraph, kernel, all_tensors, post_trans);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Fusion for post transpose of " << kernel->name() << " failed.";
      return RET_ERROR;
    }
    kernel_iter = find(kernels->begin(), kernels->end(), kernel);
    (void)kernel_iter++;
  }
  return RET_OK;
}

int EliminateTranspose::RepeteTransposeFusionPass(kernel::SubGraphKernel *subgraph,
                                                  std::vector<lite::Tensor *> *all_tensors) {
  std::vector<kernel::KernelExec *> nodes = subgraph->in_nodes();
  std::vector<lite::Tensor *> tensors = subgraph->in_tensors();
  auto iter = tensors.begin();
  while (iter != tensors.end()) {
    auto tensor = *iter;

    std::vector<kernel::KernelExec *> repete_trans;
    for (auto node : nodes) {
      if (node->type() != schema::PrimitiveType_Transpose && node->type() != schema::PrimitiveType_FormatTranspose) {
        iter++;
        continue;
      }
      if (node->in_tensors().size() == 1 && node->in_tensors().at(0) == tensor) {
        repete_trans.push_back(node);
      }
    }

    if (repete_trans.size() <= 1) {
      iter++;
      continue;
    }
    graph_changed_ = true;
    kernel::KernelExec *transpose_kernel = repete_trans.at(0);
    auto reserve_tensor = transpose_kernel->out_tensors().at(0);

    for (size_t i = 1; i < repete_trans.size(); i++) {
      kernel::KernelExec *replace_kernel = repete_trans.at(i);
      lite::Tensor *replace_tensor = replace_kernel->out_tensors().at(0);

      auto post_kernels = kernel::KernelExecUtil::FindOutKernelsForOutTensor(replace_kernel, replace_tensor);
      for (const auto &post : post_kernels) {
        replace_kernel->RemoveOutKernel(post);
        post->RemoveInKernel(replace_kernel);

        post->AddInKernel(transpose_kernel);
        transpose_kernel->AddOutKernel(post);

        auto input_index = post->FindInTensorIndex(replace_kernel->out_tensors().at(0));
        post->set_in_tensor(reserve_tensor, input_index);
      }
      subgraph->DropNode(replace_kernel);
      delete replace_kernel;
    }
    iter++;
  }
  return RET_OK;
}

int EliminateTranspose::HorizontalTransposeFusionPass(kernel::SubGraphKernel *subgraph) {
  auto kernels = &(subgraph->nodes());
  auto kernel_iter = kernels->begin();
  while (kernel_iter != kernels->end()) {
    auto kernel = *kernel_iter;
    CHECK_NULL_RETURN(kernel);

    bool is_fusion = false;
    for (size_t i = 0; i < kernel->out_tensors().size(); i++) {
      auto trans_in_tensor = kernel->out_tensors().at(i);
      auto out_kernels = kernel::KernelExecUtil::FindOutKernelsForOutTensor(kernel, trans_in_tensor);
      TransInfoPair post_trans;
      auto count = transpose_strategy_.GetTransCount(out_kernels, &post_trans);
      if (count <= 1) {
        continue;
      }

      is_fusion = true;
      graph_changed_ = true;

      kernel::KernelExec *reserve_kernel = nullptr;
      std::vector<kernel::KernelExec *> to_deletes;
      for (const auto &out_kernel : out_kernels) {
        TransInfoPair tmp_trans;
        if (GetTransposeInfo(out_kernel, &tmp_trans) != RET_OK || !IsSameTranspose(post_trans, tmp_trans)) {
          continue;
        }
        if (reserve_kernel == nullptr) {
          // firstly set value
          reserve_kernel = out_kernel;
          continue;
        }
        if (IsContain(subgraph->out_tensors(), out_kernel->out_tensors().at(0))) {
          to_deletes.push_back(reserve_kernel);
          reserve_kernel = out_kernel;
        } else {
          to_deletes.push_back(out_kernel);
        }
      }
      auto reserve_tensor = reserve_kernel->out_tensors().at(0);

      for (const auto &to_delete : to_deletes) {
        if (to_delete == reserve_kernel) {
          continue;
        }

        kernel->RemoveOutKernel(to_delete);
        to_delete->RemoveInKernel(kernel);

        auto post_kernels =
          kernel::KernelExecUtil::FindOutKernelsForOutTensor(to_delete, to_delete->out_tensors().at(0));
        for (const auto &post : post_kernels) {
          to_delete->RemoveOutKernel(post);
          post->RemoveInKernel(to_delete);

          post->AddInKernel(reserve_kernel);
          reserve_kernel->AddOutKernel(post);

          auto input_index = post->FindInTensorIndex(to_delete->out_tensors().at(0));
          post->set_in_tensor(reserve_tensor, input_index);
        }
        subgraph->DropNode(to_delete);
        delete to_delete;
      }
    }
    kernel_iter = is_fusion ? find(kernels->begin(), kernels->end(), kernel) : kernel_iter + 1;
  }
  return RET_OK;
}

int EliminateTranspose::DoubleTransposeFusion(kernel::SubGraphKernel *subgraph) {
  auto kernels = &(subgraph->nodes());
  auto kernel_iter = kernels->begin();
  while (kernel_iter != kernels->end()) {
    auto &kernel = *kernel_iter;
    CHECK_NULL_RETURN(kernel);
    (void)kernel_iter++;

    if (kernel->in_kernels().size() != 1) {
      continue;
    }

    auto pre_kernel = kernel->in_kernels().at(0);
    if (IsContain(subgraph->nodes(), kernel->in_kernels().at(0)) == false) {
      continue;
    }

    TransInfoPair post_trans_info;
    if (GetTransposeInfo(kernel, &post_trans_info) != RET_OK) {
      MS_LOG(INFO) << "The kernel " << kernel->name() << " isn't transpose and can't be fused.";
      continue;
    }

    TransInfoPair pre_trans_info;
    if (GetTransposeInfo(pre_kernel, &pre_trans_info) != RET_OK) {
      MS_LOG(INFO) << "The kernel " << pre_kernel->name() << " isn't transpose and can't be fused.";
      continue;
    }

    if (pre_trans_info.dst_format_ != post_trans_info.src_format_) {
      continue;
    }

    graph_changed_ = true;
    // record the next kernel to update iterator
    auto next_kernel = (kernel_iter == kernels->end()) ? nullptr : (*kernel_iter);

    int ret = RET_OK;
    if (pre_trans_info.src_format_ == post_trans_info.dst_format_) {
      // pattern opposite, like: nhwc2nchw & nchw2nhwc -> none
      ret = TransFullyFusion(subgraph, pre_kernel, kernel);
    } else {
      // pattern, the previous dest format and the post source format are same
      // op1: format1 -> format2, op2: format2 -> format3, like: nhwc2nchw & nchw2nc4hw4 -> nhwc2nc4hw4
      TransInfoPair new_trans_info(pre_trans_info.src_format_, post_trans_info.dst_format_);
      ret = TransHeadTailFusion(subgraph, pre_kernel, kernel, new_trans_info);
    }
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Fusion " << pre_kernel->name() << " and " << kernel->name() << " failed";
      return RET_ERROR;
    }

    // The dropped kernel may be in front of the kernel, update kernel iterator.
    kernel_iter = (next_kernel == nullptr) ? kernels->end() : (find(kernels->begin(), kernels->end(), next_kernel));
  }
  return RET_OK;
}

int EliminateTranspose::RunPass(kernel::SubGraphKernel *graph, std::vector<lite::Tensor *> *tensors) {
  int pass_count = 0;
  while (graph_changed_ && pass_count < max_pass_count_) {
    graph_changed_ = false;

    auto ret = DoubleTransposeFusion(graph);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Double transpose fusion failed in runtime pass.";
      return RET_ERROR;
    }

    ret = EliminateForSingleKernel(graph, tensors);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Eliminate for single kernel failed in runtime pass.";
      return RET_ERROR;
    }

    ret = HorizontalTransposeFusionPass(graph);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "HorizontalTransposeFusionPass failed in runtime pass.";
      return RET_ERROR;
    }

    ret = RepeteTransposeFusionPass(graph, tensors);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Graph input fusion pass failed.";
      return RET_ERROR;
    }

    pass_count++;
  }

  auto ret = graph->TopologicalSortNodes();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Topological sort kernels failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::lite::pass
