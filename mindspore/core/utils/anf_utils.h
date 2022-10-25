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
#include <set>
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include "ir/anf.h"
#include "ir/dtype.h"
#include "base/base.h"
#include "ir/primitive.h"
#include "utils/hash_map.h"

namespace mindspore {
constexpr auto kInfer = "DS_Infer";
constexpr auto kInit = "DS_Init";
constexpr auto kUpdate = "DS_Update";

// Define constant about size number here.
constexpr size_t kSizeZero = 0;
constexpr size_t kSizeOne = 1;
constexpr size_t kSizeTwo = 2;
constexpr size_t kSizeThree = 3;
constexpr size_t kSizeFour = 4;
constexpr size_t kSizeFive = 5;

// Define constant about index number here.
constexpr size_t kIndexZero = 0;
constexpr size_t kIndexOne = 1;
constexpr size_t kIndexTwo = 2;
constexpr size_t kIndexThree = 3;
constexpr size_t kIndexFour = 4;
constexpr size_t kIndexFive = 5;

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
  // get int value of value node
  static int64_t GetIntValue(const AnfNodePtr &anf_node);
  // get int value of value pointer
  static int64_t GetIntValue(const ValuePtr &value);
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
  // Check whether the node has dump flag or not.
  static bool HasDumpFlag(const AnfNodePtr &node);
  static AbstractScope GetAbstractLock(const AnfNode *node);
  static void OpenAbstractLock();
  static void CloseAbstractLock();

  // Custom actor node is for dynamic shape.
  // Generate a Init custom actor node.
  static AnfNodePtr NewInitActorNode(CustomActorCallback f, const CNodePtr &base_cnode);
  // Generate a Infer custom actor node.
  static AnfNodePtr NewInferActorNode(CustomActorCallback f, const CNodePtr &base_cnode);
  static bool IsCustomActorNode(const AnfNodePtr &node);
  static std::string GetCustomActorType(const AnfNodePtr &node);
  static std::string GetCustomActorName(const AnfNodePtr &node);
  static CNodePtr GetCustomActorBaseNode(const AnfNodePtr &node);
  static CustomActorCallback GetCustomFunc(const AnfNodePtr &node);
  static bool IsCutomActorNodeSame(const AnfNodePtr &node1, const AnfNodePtr &node2);
  // set the inferop,initop to base_node's user_data
  static void SetCustomInfoToBaseNode(const AnfNodePtr &base_cnode, const AnfNodePtr &inferop,
                                      const AnfNodePtr &initop);
  static AnfNodePtr GetCustomInferopNode(const AnfNodePtr &base_cnode);
  static mindspore::HashMap<size_t, std::pair<AnfNodeWeakPtr, size_t>> &GetRealInputNodes(const CNodePtr &cnode);
  static std::vector<size_t> TransShapeToSizet(const abstract::ShapePtr &shape);
  // Judge whether node's monad output should be skipped. Currently this method returns true in one scenarios:
  // 1. The node is a RpcRecv node with monad type output.
  static bool NeedJumpMonadOutput(const AnfNodePtr &node);
};

//
// FlatParameterFinder finds flat parameters from parameters.
//
class MS_CORE_API FlatParameterFinder {
 public:
  FlatParameterFinder() = default;
  ~FlatParameterFinder() = default;

  // Add a parameter for search.
  void AddParameter(const ParameterPtr &param);

  // Add nodes for search, parameter nodes will be added.
  void AddNodes(const std::vector<AnfNodePtr> &nodes);

  // Get the flat parameter and data offset for the given parameter.
  // return (nullptr, 0) when flat parameter not found.
  std::pair<ParameterPtr, size_t> FindFlatParameter(const ParameterPtr &param);

  // Get all flat parameters.
  const std::set<ParameterPtr> &GetFlatParameters();

 private:
  struct FlatParamInfo {
    ParameterPtr flat_param = nullptr;
    void *chunk = nullptr;
    size_t offset = 0;
  };

  void UpdateFlatParameters();

  mindspore::HashMap<void *, ParameterPtr> candidate_flat_params_;
  mindspore::HashMap<ParameterPtr, FlatParamInfo> param_to_flat_param_;
  std::set<ParameterPtr> flat_params_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_UTILS_ANF_UTILS_H_
