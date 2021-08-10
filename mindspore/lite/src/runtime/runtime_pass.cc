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

#include "src/runtime/runtime_pass.h"
#include "nnacl/conv_parameter.h"

namespace mindspore::lite {
void Nc4hw4PassReplace(std::vector<kernel::LiteKernel *> *kernels, std::vector<Tensor *> *tensors, size_t index) {
  kernel::LiteKernel *conv_kernel = kernels->at(index);
  kernel::LiteKernel *traspose_kernel = conv_kernel->out_kernels().front();
  kernel::LiteKernel *c4_kernel = traspose_kernel->out_kernels().front();

  /* tensor */
  Tensor *transpose_param_tensor = traspose_kernel->in_tensors().at(1);
  VectorErase(tensors, transpose_param_tensor);
  delete transpose_param_tensor;
  transpose_param_tensor = nullptr;

  Tensor *conv_out_tensor = conv_kernel->out_tensors().front();
  conv_out_tensor->set_format(NC4HW4);
  Tensor *c4_input_tensor = c4_kernel->in_tensors().front();
  c4_kernel->set_in_tensor(conv_out_tensor, 0);
  VectorErase(tensors, c4_input_tensor);
  delete c4_input_tensor;
  c4_input_tensor = nullptr;

  /* kernel */
  VectorErase(kernels, traspose_kernel);
  delete traspose_kernel;
  traspose_kernel = nullptr;
  conv_kernel->set_out_kernels({c4_kernel});
  c4_kernel->set_in_kernels({conv_kernel});

  return;
}

bool Nc4hw4PassMatch(std::vector<kernel::LiteKernel *> *kernels, size_t index) {
  kernel::LiteKernel *start_kernel = kernels->at(index);
  if (IsContain(Nc4hw4FormatOutOpList, start_kernel->type()) == false) {
    return false;
  }
  if (start_kernel->out_kernels().size() != 1) {
    return false;
  }
  if (reinterpret_cast<ConvParameter *>(start_kernel->op_parameter())->group_ != 1) {
    /* conv-depthwise and group-conv */
    return false;
  }

  kernel::LiteKernel *traspose_kernel = start_kernel->out_kernels().front();
  if (start_kernel->type() != Nc4hw4FormatTransposeOp) {
    return false;
  }
  if (traspose_kernel->out_kernels().size() != 1) {
    return false;
  }

  kernel::LiteKernel *end_kernel = traspose_kernel->out_kernels().front();
  if (IsContain(Nc4hw4FormatInOpList, end_kernel->type()) == false) {
    return false;
  }

  /* double check ops topological sorted in kernel-list */
  auto start_iter = find(kernels->begin(), kernels->end(), start_kernel);
  auto start_index = std::distance(kernels->begin(), start_iter);
  auto transpose_iter = find(kernels->begin(), kernels->end(), traspose_kernel);
  auto transpose_index = std::distance(kernels->begin(), transpose_iter);
  auto end_iter = find(kernels->begin(), kernels->end(), end_kernel);
  auto end_index = std::distance(kernels->begin(), end_iter);
  if (start_index > transpose_index || transpose_index > end_index) {
    return false;
  }

  return true;
}

bool Nc4hw4PassValid(const InnerContext *context, std::vector<kernel::LiteKernel *> *kernels) {
  return false;

  if (context->IsGpuEnabled() || context->IsNpuEnabled()) {
    return false;
  }

  for (auto kernel : *kernels) {
    if (kernel->op_parameter()->quant_type_ == schema::QuantType_AwareTraining ||
        kernel->op_parameter()->quant_type_ == schema::QuantType_PostTraining) {
      return false;
    }
  }
  return true;
}

void Nc4hw4Pass(std::vector<kernel::LiteKernel *> *kernels, std::vector<Tensor *> *tensors) {
  size_t kernel_size = kernels->size();
  size_t index = 0;
  for (; index < kernel_size - 2; index++) {
    kernel::LiteKernel *kernel = kernels->at(index);

    if (kernel->subgraph_type() != kernel::kNotSubGraph) {
      kernel::SubGraphKernel *subgraph = reinterpret_cast<kernel::SubGraphKernel *>(kernel);
      std::vector<kernel::LiteKernel *> &particial_nodes = subgraph->nodes();
      Nc4hw4Pass(&particial_nodes, tensors);
    }

    if (Nc4hw4PassMatch(kernels, index)) {
      Nc4hw4PassReplace(kernels, tensors, index);
      index += 1;
    }
    kernel_size = kernels->size();
  }
  return;
}
}  // namespace mindspore::lite
