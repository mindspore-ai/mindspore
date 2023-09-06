/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "src/litert/pass/format_pass/insert_transpose.h"
#include "src/litert/pass/format_pass/format_utils.h"
#include "src/litert/kernel_exec_util.h"
#include "nnacl/base/format_transpose.h"

namespace mindspore::lite::pass {
int InsertTranspose::TransposeConstData(kernel::KernelExec *kernel, size_t index) {
  lite::Tensor *tensor = kernel->in_tensors().at(index);
  Format except_format = kernel->desc().format;
  if (tensor->format() == except_format) {
    return RET_OK;
  }

  if (tensor->allocator() != nullptr) {
    MS_LOG(ERROR) << "Const data allocator invalid.";
    return RET_ERROR;
  }

  void *buffer = malloc(tensor->Size());
  if (buffer == nullptr) {
    MS_LOG(ERROR) << "malloc transpose data failed";
    return RET_ERROR;
  }
  auto ret = TransData(tensor->data(), buffer, (FormatC)(tensor->format()), (FormatC)except_format,
                       static_cast<TypeIdC>(tensor->data_type()), tensor->Batch(), tensor->Channel(),
                       tensor->Height() * tensor->Width());
  if (ret != RET_OK) {
    return ret;
  }

  tensor->FreeData();
  tensor->set_data(buffer, true);
  if (!TransTensorShapeAndFormat(tensor, except_format)) {
    MS_LOG(ERROR) << "unsupported except format: " << except_format;
    return RET_ERROR;
  }
  return RET_OK;
}

int InsertTranspose::RunPass(kernel::SubGraphKernel *graph, std::vector<lite::Tensor *> *tensors) {
  auto kernels = graph->nodes();

  auto origin_kernel_size = kernels.size();
  for (size_t kernel_index = 0; kernel_index < origin_kernel_size; kernel_index++) {
    kernel::KernelExec *kernel = kernels.at(kernel_index);
    CHECK_NULL_RETURN(kernel);
    Format kernel_format = kernel->desc().format;
    if (kernel_format == format_) {
      continue;
    }

    // to be realized: 1. get type from kernel; 2. flag for transpose weight.
    std::string type_name = kernel::TypeName(kernel->type());
    auto find_result = cloud_format_kernel_list.find(type_name);
    if (find_result == cloud_format_kernel_list.end()) {
      MS_LOG(INFO) << "Kernel(" << kernel->name() << ") has different format(" << FormatEnumToString(kernel_format)
                   << ") with graph format(" << FormatEnumToString(format_) << "), but not in insert-transpose white "
                   << "list and will not insert transpose kernel, type name: " << type_name;
      continue;
    }

    auto insert_input_list = find_result->second;
    for (auto index : insert_input_list) {
      if (index >= kernel->in_tensors().size()) {
        continue;
      }

      if (kernel->in_tensors().at(index)->IsConst()) {
        (void)TransposeConstData(kernel, index);
        continue;
      }
      auto ret = InsertPreTranspose(graph, kernel, tensors, TransInfoPair(format_, kernel_format), index,
                                    create_format_transpose_func_);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Insert pre transpose for op: " << kernel->name() << ", index: " << index << ", failed";
        return RET_ERROR;
      }
    }

    for (size_t i = 0; i < kernel->out_kernels().size(); i++) {
      auto ret = InsertPostTranspose(graph, kernel, tensors, TransInfoPair(kernel_format, format_), i,
                                     create_format_transpose_func_);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Insert post transpose for op: " << kernel->name() << ", index: " << i << ", failed";
        return RET_ERROR;
      }
    }
    // graph output node has no output kernels, take care of these nodes
    if (IsContain(graph->out_nodes(), kernel)) {
      for (size_t i = 0; i < kernel->out_tensors().size(); i++) {
        auto ret = InsertPostTranspose(graph, kernel, tensors, TransInfoPair(kernel_format, format_), i,
                                       create_format_transpose_func_);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "Insert post transpose for op: " << kernel->name() << ", index: " << i << ", failed";
          return RET_ERROR;
        }
      }
    }
    MS_LOG(INFO) << "Insert transpose before and after node: " << kernel->name();

    graph->SetInNodes(kernel::KernelExecUtil::SubgraphInputNodes(graph->nodes()));
    graph->SetOutNodes(kernel::KernelExecUtil::SubgraphOutputNodes(graph->nodes()));

    auto ret = graph->TopologicalSortNodes();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Topological sort kernels failed.";
      return RET_ERROR;
    }
  }

  return RET_OK;
}
}  // namespace mindspore::lite::pass
