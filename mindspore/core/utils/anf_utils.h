/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_ANF_UTILS_H_
#define MINDSPORE_CORE_UTILS_ANF_UTILS_H_
#include <functional>
#include <vector>
#include <string>
#include <utility>
#include "ir/anf.h"
#include "ir/dtype.h"
#include "base/base.h"
#include "ir/primitive.h"

namespace mindspore {
constexpr auto kInfer = "DS_Infer";
constexpr auto kInit = "DS_Init";
constexpr auto kUpdate = "DS_Update";

class MS_CORE_API AbstractScope {
 public:
  explicit AbstractScope(std::recursive_mutex *mu);
  AbstractScope(const AbstractScope &other) = delete;
  AbstractScope operator=(const AbstractScope &other) = delete;
  AbstractScope(AbstractScope &&other);
  AbstractScope &operator=(AbstractScope &&other);
  ~AbstractScope();

 private:
  std::recursive_mutex *mu_;
};

class MS_CORE_API AnfUtils {
 public:
  using CustomActorCallback = std::function<void(void *args)>;
  static bool IsDimUnknown(const abstract::ShapePtr &shape);
  static bool IsShapeDynamic(const abstract::ShapePtr &shape);
  static bool IsShapeDynamic(const std::vector<size_t> &shape);
  static bool IsNodeOutputDynamicShape(const CNodePtr &node);
  static bool IsDimUnknown(const AnfNodePtr &node);
  // check whether the anf node is a real kernel that can run on device,parameter and constant is real kernel too
  static bool IsRealKernel(const AnfNodePtr &node);
  // check whether the anf node is a real kernel that is a cnode and can run on device
  static bool IsRealCNodeKernel(const AnfNodePtr &node);
  // get kernel name of anf node
  static std::string GetCNodeName(const AnfNodePtr &node);
  // get the num of inputs exclude monads for real_kernel (which can be build and run in device)
  static size_t GetInputTensorNum(const AnfNodePtr &node);
  // get the num of output real_kernel(which can be build and run in device)
  static size_t GetOutputTensorNum(const AnfNodePtr &node);
  // set attr of anf node
  static void SetNodeAttr(const std::string &key, const ValuePtr &value, const AnfNodePtr &node);
  // get the node's real kernel recursively
  static std::pair<AnfNodePtr, size_t> VisitKernel(const AnfNodePtr &anf_node, size_t index);
  // check whether the node is a GraphKernel node.
  static bool IsGraphKernel(const AnfNodePtr &node);
  // check whether the node is a node in GraphKernel's subgraph.
  static bool IsNodeInGraphKernel(const AnfNodePtr &node);
  // Set dump flag to CNode's primitive.
  static void SetDumpFlag(const AnfNodePtr &node);
  // Get dump flag from CNode's primitive.
  static bool GetDumpFlag(const AnfNodePtr &node);
  static AbstractScope GetAbstractLock(const AnfNode *node);
  static void OpenAbstractLock();
  static void CloseAbstractLock();

  // Custom actor node is for dynamic shape.
  // Generate a Init custom actor node.
  static AnfNodePtr NewInitActorNode(CustomActorCallback f, const CNodePtr &base_cnode);
  // Generate a Infer custom actor node. If `is_fake` is set to true, this node is a fake node without any infer action.
  static AnfNodePtr NewInferActorNode(CustomActorCallback f, const CNodePtr &base_cnode, bool is_fake);
  // Generate a Update custom actor node. If `is_just_sync` is set to true, this node is just for a stream-sync call.
  static AnfNodePtr NewUpdateActorNode(CustomActorCallback f, const CNodePtr &base_cnode, bool is_just_sync);
  static bool IsCustomActorNode(const AnfNodePtr &node);
  static std::string GetCustomActorType(const AnfNodePtr &node);
  static std::string GetCustomActorName(const AnfNodePtr &node);
  static CNodePtr GetCustomActorBaseNode(const AnfNodePtr &node);
  static bool GetCustomActorJustSyncFlag(const AnfNodePtr &node);
  static CustomActorCallback GetCustomFunc(const AnfNodePtr &node);
  static bool IsCutomActorNodeSame(const AnfNodePtr &node1, const AnfNodePtr &node2);
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_UTILS_ANF_UTILS_H_
