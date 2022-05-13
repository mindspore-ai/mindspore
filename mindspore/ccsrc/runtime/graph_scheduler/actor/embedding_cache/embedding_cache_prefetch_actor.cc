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

#include "runtime/graph_scheduler/actor/embedding_cache/embedding_cache_prefetch_actor.h"
#include "backend/common/optimizer/dynamic_shape/dynamic_shape_helper.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace runtime {
using kernel::Address;
using kernel::AddressPtrList;
using mindspore::session::KernelGraph;

// One and two dimensional shape placeholder.
const ShapeVector kOneDimensionalShape = {1};
const ShapeVector kTwoDimensionalShape = {1, 1};

namespace {
ParameterPtr NewParameter(const KernelGraphPtr &graph, TypePtr type, const ShapeVector &shape) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(type);

  auto param = graph->NewParameter();
  MS_EXCEPTION_IF_NULL(param);
  auto abstract = std::make_shared<abstract::AbstractTensor>(type, shape);
  param->set_abstract(abstract);

  auto mutable_inputs = graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(mutable_inputs);
  mutable_inputs->push_back(param);

  return param;
}

bool InferOpShape(const CNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  opt::dynamic_shape::InferOp(kernel);
  auto args = kernel::GetArgsFromCNode(kernel);
  MS_EXCEPTION_IF_NULL(args);

  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  if (!kernel_mod->Resize(args->op, args->inputs, args->outputs, args->depend_tensor_map)) {
    MS_LOG(ERROR) << "Kernel " << kernel->fullname_with_scope() << " resize failed.";
    return false;
  }
  return true;
}
}  // namespace

void EmbeddingCachePrefetchActor::Initialize() {
  MS_EXCEPTION_IF_NULL(device_context_);
  if (!device_context_->CreateStream(&stream_id_)) {
    MS_LOG(EXCEPTION) << "Create stream failed.";
  }
}

void EmbeddingCachePrefetchActor::Finalize() {
  embedding_cache_lookup_node_ = nullptr;
  embedding_cache_update_node_ = nullptr;
}

void EmbeddingCachePrefetchActor::BuildEmbeddingCacheLookupKernel() {
  auto graph = std::make_shared<KernelGraph>();

  // 1. Create parameter nodes which are inputs of embedding cache look up kernel(operator name: 'Gather').
  ParameterPtr input_param = NewParameter(graph, kFloat32, kTwoDimensionalShape);
  ParameterPtr input_indices = NewParameter(graph, kInt32, kOneDimensionalShape);

  // 2. Create a CNode for operator Gather.
  PrimitivePtr emb_lookup_primitive = std::make_shared<Primitive>(kGatherV2OpName);
  emb_lookup_primitive->set_attr(kAttrAxis, MakeValue<int64_t>(0));
  emb_lookup_primitive->set_attr(kAttrInputIsDynamicShape, MakeValue(true));
  emb_lookup_primitive->set_attr(kAttrOutputIsDynamicShape, MakeValue(true));
  emb_lookup_primitive->set_attr(kAttrStream, MakeValue(stream_id_));

  std::vector<AnfNodePtr> emb_lookup_input_nodes{NewValueNode(emb_lookup_primitive), input_param, input_indices};
  embedding_cache_lookup_node_ = graph->NewCNode(emb_lookup_input_nodes);
  MS_EXCEPTION_IF_NULL(embedding_cache_lookup_node_);
  auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, kTwoDimensionalShape);
  embedding_cache_lookup_node_->set_abstract(abstract);

  // 3. Kernel build process.
  MS_EXCEPTION_IF_NULL(device_context_);
  device_context_->CreateKernel({embedding_cache_lookup_node_});
}

void EmbeddingCachePrefetchActor::BuildEmbeddingCacheUpdateKernel() {
  auto graph = std::make_shared<KernelGraph>();

  // 1. Create parameter nodes which are inputs of embedding cache update kernel(operator name: 'ScatterUpdate').
  ParameterPtr input_param = NewParameter(graph, kFloat32, kTwoDimensionalShape);
  ParameterPtr input_indices = NewParameter(graph, kInt32, kOneDimensionalShape);
  ParameterPtr update_values = NewParameter(graph, kFloat32, kTwoDimensionalShape);

  // 2. Create a CNode for operator ScatterUpdate.
  PrimitivePtr embedding_cache_update_primitive = std::make_shared<Primitive>(kScatterUpdateOpName);
  embedding_cache_update_primitive->set_attr(kAttrInputIsDynamicShape, MakeValue(true));
  embedding_cache_update_primitive->set_attr(kAttrStream, MakeValue(stream_id_));

  std::vector<AnfNodePtr> embedding_cache_update_input_nodes{NewValueNode(embedding_cache_update_primitive),
                                                             input_param, input_indices, update_values};
  embedding_cache_update_node_ = graph->NewCNode(embedding_cache_update_input_nodes);
  MS_EXCEPTION_IF_NULL(embedding_cache_update_node_);
  auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, kTwoDimensionalShape);
  embedding_cache_update_node_->set_abstract(abstract);

  // 3. Kernel build process.
  MS_EXCEPTION_IF_NULL(device_context_);
  device_context_->CreateKernel({embedding_cache_update_node_});
}

bool EmbeddingCachePrefetchActor::LookupDeviceEmbeddingCache(void *indices, void *embedding_cache, size_t indices_num,
                                                             size_t cache_size, size_t embedding_size, void *outputs) {
  MS_EXCEPTION_IF_NULL(indices);
  MS_EXCEPTION_IF_NULL(embedding_cache);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(embedding_cache_lookup_node_);

  // 1. Update parameter nodes' shape.
  auto input_param_node = common::AnfAlgo::GetInputNode(embedding_cache_lookup_node_, 0);
  MS_EXCEPTION_IF_NULL(input_param_node);
  const ShapeVector input_param_shape = {SizeToLong(cache_size), SizeToLong(embedding_size)};
  auto input_param_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, input_param_shape);
  input_param_node->set_abstract(input_param_abstract);

  auto input_indices_node = common::AnfAlgo::GetInputNode(embedding_cache_lookup_node_, 1);
  MS_EXCEPTION_IF_NULL(input_indices_node);
  const ShapeVector input_indices_shape = {SizeToLong(indices_num)};
  auto input_indices_abstract = std::make_shared<abstract::AbstractTensor>(kInt32, input_indices_shape);
  input_indices_node->set_abstract(input_indices_abstract);

  // 2. Infer shape for embedding cache look up kernel(operator name: 'Gather') which is dynamic shape kernel.
  if (!InferOpShape(embedding_cache_lookup_node_)) {
    MS_LOG(ERROR) << "Infer operator shape failed, op name: " << embedding_cache_lookup_node_->fullname_with_scope();
    return false;
  }

  // 3. Do embedding cache look up on device.
  AddressPtrList kernel_inputs = {
    std::make_shared<Address>(embedding_cache, cache_size * embedding_size * sizeof(float)),
    std::make_shared<Address>(indices, indices_num * sizeof(int))};
  AddressPtrList kernel_outputs = {std::make_shared<Address>(outputs, indices_num * embedding_size * sizeof(float))};

  MS_EXCEPTION_IF_NULL(device_context_);
  auto ret = device_context_->LaunchKernel(embedding_cache_lookup_node_, kernel_inputs, {}, kernel_outputs);
  if (!ret) {
    MS_LOG(ERROR) << "Launch kernel: " << embedding_cache_lookup_node_->fullname_with_scope() << " failed.";
    return false;
  }
  return true;
}

bool EmbeddingCachePrefetchActor::UpdateDeviceEmbeddingCache(void *indices, void *update_value, size_t indices_num,
                                                             size_t cache_size, size_t embedding_size,
                                                             void *embedding_cache) {
  MS_EXCEPTION_IF_NULL(indices);
  MS_EXCEPTION_IF_NULL(update_value);
  MS_EXCEPTION_IF_NULL(embedding_cache);
  MS_EXCEPTION_IF_NULL(embedding_cache_update_node_);

  // 1. Update parameter nodes' shape.
  auto input_param_node = common::AnfAlgo::GetInputNode(embedding_cache_update_node_, 0);
  MS_EXCEPTION_IF_NULL(input_param_node);
  const ShapeVector input_param_shape = {SizeToLong(cache_size), SizeToLong(embedding_size)};
  auto input_param_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, input_param_shape);
  input_param_node->set_abstract(input_param_abstract);

  auto input_indices_node = common::AnfAlgo::GetInputNode(embedding_cache_update_node_, 1);
  MS_EXCEPTION_IF_NULL(input_indices_node);
  const ShapeVector input_indices_shape = {SizeToLong(indices_num)};
  auto input_indices_abstract = std::make_shared<abstract::AbstractTensor>(kInt32, input_indices_shape);
  input_indices_node->set_abstract(input_indices_abstract);

  auto update_values_node = common::AnfAlgo::GetInputNode(embedding_cache_update_node_, 2);
  MS_EXCEPTION_IF_NULL(update_values_node);
  const ShapeVector update_values_shape = {SizeToLong(indices_num), SizeToLong(embedding_size)};
  auto update_values_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, update_values_shape);
  update_values_node->set_abstract(update_values_abstract);

  // 2. Infer shape for embedding cache update kernel(operator name: 'ScatterUpdate') which is dynamic shape kernel.
  if (!InferOpShape(embedding_cache_update_node_)) {
    MS_LOG(ERROR) << "Infer operator shape failed, op name: " << embedding_cache_update_node_->fullname_with_scope();
    return false;
  }

  // 3. Do update cache on device.
  AddressPtrList kernel_inputs = {
    std::make_shared<Address>(embedding_cache, cache_size * embedding_size * sizeof(float)),
    std::make_shared<Address>(indices, indices_num * sizeof(int)),
    std::make_shared<Address>(update_value, indices_num * embedding_size * sizeof(float))};
  AddressPtrList kernel_outputs = {
    std::make_shared<Address>(embedding_cache, cache_size * embedding_size * sizeof(float))};

  MS_EXCEPTION_IF_NULL(device_context_);
  auto ret = device_context_->LaunchKernel(embedding_cache_update_node_, kernel_inputs, {}, kernel_outputs);
  if (!ret) {
    MS_LOG(ERROR) << "Launch kernel: " << embedding_cache_update_node_->fullname_with_scope() << " failed.";
    return false;
  }
  return true;
}
}  // namespace runtime
}  // namespace mindspore
