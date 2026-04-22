/**
 * Copyright 2025-2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_MS_BACKEND_MS_BACKEND_BASE_UTIL_H
#define MINDSPORE_CCSRC_BACKEND_MS_BACKEND_MS_BACKEND_BASE_UTIL_H

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <utility>
#include <functional>

#include "ir/anf.h"
#include "ir/tensor.h"
#include "ir/func_graph.h"
#include "abstract/abstract_value.h"
#include "backend/ms_backend/segment_runner.h"
#include "backend/ms_backend/graph_partition.h"
#include "backend/ms_backend/ms_backend_base.h"
#include "backend/ms_backend/runtime/actors/base/abstract_actor.h"
#include "backend/ms_backend/runtime/actors/base/actor_set.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_tensor.h"
#include "include/backend/common/pass_manager/graph_optimizer.h"

namespace mindspore {
namespace backend {
namespace ms_backend {

constexpr auto kControlNodeJsonSuffix = "_backinfo.json";
using PrimTypePair = std::pair<PrimitivePtr, abstract::AbstractFunctionPtr>;
using TypedPrimitiveAbstractClosurePtr = std::shared_ptr<abstract::TypedPrimitiveAbstractClosure>;
using KernelTensorPtr = std::shared_ptr<runtime::KernelTensor>;
using AbstractActorPtr = std::shared_ptr<runtime::AbstractActor>;
using KernelRunnerPtr = std::shared_ptr<runtime::KernelRunner>;
using ActorSetPtr = std::shared_ptr<runtime::ActorSet>;

int64_t GetTupleGetItemOutIndex(const CNodePtr &tuple_get_item);
bool EnableKBKCompileCache(const FuncGraphPtr &func_graph, const device::DeviceType &device_type);
bool ExportCompileCacheKBK(const FuncGraphPtr &func_graph, const device::DeviceType &device_type);
KernelWithIndex VisitRealNodeWithNestLevel(const AnfNodePtr &anf_node, size_t index, size_t *nest_level);
bool NeedConvertToRealTupleGetItem(const CNodePtr &cnode);
void CheckNodeValid(const AnfNodePtr &node);
void UnifyIR(const CNodePtr &cnode);
void DoUnifyMindIRPass(const FuncGraphPtr &graph, const std::shared_ptr<opt::GraphOptimizer> &optimizer);
bool HasSwitchNode(const FuncGraphPtr &func_graph);
bool HasAbstractRefOutput(const abstract::AbstractBasePtr &abs);
bool IsNodeValid(const AnfNodePtr &node);
bool IsTopoDependNode(const std::set<AnfNodePtr> &checked, const AnfNodePtr &node, std::set<AnfNodePtr> *checked_node);
bool HasParallelSwitchCall(const FuncGraphPtr &func_graph);
bool IsFuncGraphSupportSwitchInline(const FuncGraphPtr &graph);
bool IsEnableControlFlowInline(const FuncGraphPtr &graph);
bool IsTupleOutputOfAnyType(const abstract::AbstractBasePtr &abstract, const tensor::TensorPtr &tensor);
bool EnableSymbolEngine(const FuncGraphPtr &func_graph);
void BuildSymbolEngine(const FuncGraphPtr &func_graph);
std::string GetUniqueNodeId(const AnfNodePtr &node, bool must_have_unique_name = true);
void PushTensor(const VectorRef &args, const std::vector<AnfNodePtr> &parameters, const AnfNodePtr &front_node,
                std::vector<tensor::TensorPtr> *input_tensors);
void PushTupleTensor(const VectorRef &args, const std::vector<AnfNodePtr> &parameters, const AnfNodePtr &front_node,
                     size_t index, std::map<size_t, std::vector<tensor::TensorPtr>> *flatten_values,
                     std::vector<tensor::TensorPtr> *input_tensors);
bool IsEmptySequence(const AnfNodePtr &output_node, const std::vector<tensor::TensorPtr> &output_tensors,
                     const size_t *const output_position);
void TraverseGraphMap(
  const FuncGraphManagerPtr &manager_ptr, FuncGraphTransaction *tr, const FuncGraphSet &fgs,
  const std::function<std::shared_ptr<FuncGraph>(const PrimitivePtr, const AbstractFunctionPtr)> &get_prim_graph);
runtime::KernelMapPosition FetchOriginOutputOrder(const AnfNodePtr &node);

BackendGraphId CompileSegment(const GraphSegmentPtr &segment, const AnfNodePtrList &inputs,
                              const AnfNodePtrList &outputs, MSBackendBase *backend);

bool IsMemoryLeak(const KernelTensorPtr &kernel_tensor);
void CheckMemoryLeak(const AbstractActorPtr &actor, const KernelTensorPtr &kernel_tensor);
void CheckMemoryLeakV2(const KernelRunnerPtr &actor, const KernelTensorPtr &kernel_tensor);
void StrictCheckForDeviceAddress(const runtime::ActorSet *actor_set);

}  // namespace ms_backend
}  // namespace backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_MS_BACKEND_MS_BACKEND_BASE_UTIL_H
