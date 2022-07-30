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

#include "src/litert/pass/to_nchw_format.h"
#include <unordered_map>
#include "src/litert/kernel_exec_util.h"
#include "src/litert/pass/pass_utils.h"
#include "src/litert/kernel_registry.h"

namespace mindspore::lite::pass {
// Holds the index of the input tensor index that needs to be inserted a transpose kernel.
// Only support lite inference, not support training.
// If the index vec is empty, mean all inputs or all outputs.
static const std::unordered_map<schema::PrimitiveType, std::vector<std::vector<size_t>>> kernel_insert_map = {
  {schema::PrimitiveType_AvgPoolFusion, {{0}, {0}}},
  {schema::PrimitiveType_BatchNorm, {{0}, {0}}},
  {schema::PrimitiveType_BatchToSpace, {{0}, {0}}},
  {schema::PrimitiveType_BiasAdd, {{0}, {0}}},
  {schema::PrimitiveType_Concat, {{}, {0}}},
  {schema::PrimitiveType_Conv2DFusion, {{0}, {0}}},
  {schema::PrimitiveType_Conv2dTransposeFusion, {{0}, {0}}},
  {schema::PrimitiveType_DepthToSpace, {{0}, {0}}},
  {schema::PrimitiveType_FusedBatchNorm, {{0}, {0}}},
  {schema::PrimitiveType_InstanceNorm, {{0}, {0}}},
  {schema::PrimitiveType_LRN, {{0}, {0}}},
  {schema::PrimitiveType_MaxPoolFusion, {{0}, {0}}},
  {schema::PrimitiveType_PadFusion, {{0}, {0}}},
  {schema::PrimitiveType_PReLUFusion, {{0}, {0}}},
  {schema::PrimitiveType_Resize, {{0}, {0}}},
  {schema::PrimitiveType_ROIPooling, {{0}, {0}}},
  {schema::PrimitiveType_SpaceToBatch, {{0}, {0}}},
  {schema::PrimitiveType_SpaceToBatchND, {{0}, {0}}},
  {schema::PrimitiveType_SpaceToDepth, {{0}, {0}}},
};

int ToNCHWFormat::InsertPreTransKernel(kernel::SubGraphKernel *subgraph, kernel::KernelExec *kernel,
                                       std::vector<Tensor *> *all_tensors) {
  if (kernel_insert_map.find(kernel->type()) == kernel_insert_map.end()) {
    MS_LOG(ERROR) << "Get Transpose insert index failed.";
    return RET_ERROR;
  }
  auto input_index = kernel_insert_map.at(kernel->type()).at(0);
  if (input_index.size() == 0) {
    for (size_t i = 0; i < kernel->in_tensors().size(); i++) {
      input_index.push_back(i);
    }
  }

  for (const auto &index : input_index) {
    // new src_format(nhwc/nchw) -> dst_format kernel
    auto ret = InsertPreTranspose(subgraph, kernel, all_tensors,
                                  TransInfoPair((FormatC)src_format_, (FormatC)dst_format_), index);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert previous transpose kernel for op: " << kernel->name() << " input tensor " << index
                    << " failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int ToNCHWFormat::InsertPostTransKernel(kernel::SubGraphKernel *subgraph, kernel::KernelExec *kernel,
                                        std::vector<Tensor *> *all_tensors) {
  if (kernel_insert_map.find(kernel->type()) == kernel_insert_map.end()) {
    MS_LOG(ERROR) << "Get Transpose insert index failed.";
    return RET_ERROR;
  }
  auto output_index = kernel_insert_map.at(kernel->type()).at(1);
  if (output_index.size() == 0) {
    for (size_t i = 0; i < kernel->out_tensors().size(); i++) {
      output_index.push_back(i);
    }
  }

  for (const auto &index : output_index) {
    // new dst_format -> src_format(nhwc/nchw) kernel
    auto ret = InsertPostTranspose(subgraph, kernel, all_tensors,
                                   TransInfoPair((FormatC)dst_format_, (FormatC)src_format_), index);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert post transpose kernel for op: " << kernel->name() << " output tensor " << index
                    << " failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int ToNCHWFormat::Run(kernel::SubGraphKernel *subgraph, std::vector<Tensor *> *tensors) {
  auto kernels = subgraph->nodes();
  auto origin_kernel_size = kernels.size();
  for (size_t i = 0; i < origin_kernel_size; i++) {
    auto kernel = kernels.at(i);
    CHECK_NULL_RETURN(kernel);
    if (to_trans_kernels_.find(kernel->type()) == to_trans_kernels_.end()) {
      continue;
    }

    // replace kernel with specific format
    auto kernel_key = kernel->desc();
    kernel_key.format = dst_format_;
    auto ret = KernelRegistry::GetInstance()->ReplaceKernelExec(kernel, kernel_key);
    if (ret != RET_OK) {
      continue;
    }

    ret = InsertPreTransKernel(subgraph, kernel, tensors);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert previous transpose kernel for op: " << kernel->name() << " failed.";
      return RET_ERROR;
    }
    ret = InsertPostTransKernel(subgraph, kernel, tensors);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert post transpose kernel for op: " << kernel->name() << " failed.";
      return RET_ERROR;
    }
  }

  subgraph->SetInNodes(kernel::KernelExecUtil::SubgraphInputNodes(subgraph->nodes()));
  subgraph->SetOutNodes(kernel::KernelExecUtil::SubgraphOutputNodes(subgraph->nodes()));
  auto ret = subgraph->TopologicalSortNodes();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Topological sort kernels failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::lite::pass
