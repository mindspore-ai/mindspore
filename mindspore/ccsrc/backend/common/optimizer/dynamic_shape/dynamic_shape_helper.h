/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_OPTIMIZER_DYNAMIC_SHAPE_DYNAMIC_SHAPE_HELPER_H
#define MINDSPORE_CCSRC_BACKEND_COMMON_OPTIMIZER_DYNAMIC_SHAPE_DYNAMIC_SHAPE_HELPER_H

#include <vector>
#include <memory>
#include <string>

#include "ir/anf.h"
#include "ir/functor.h"
#include "utils/ms_utils.h"
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/helper.h"
#include "runtime/pynative/op_compiler.h"
#include "kernel/framework_utils.h"

namespace mindspore::opt::dynamic_shape {
BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);

void UpdateKernelTensorShape(const BaseShapePtr &base_shape,
                             const std::vector<kernel::KernelTensor *> &output_kernel_tensors);

abstract::AbstractBasePtr InferShapeAndType(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args);

void UpdateKernelTensorType(const TypePtr &type, const std::vector<kernel::KernelTensor *> &output_kernel_tensors);

bool IsRealCNode(const BaseRef &n);
void InferOp(const CNodePtr &node, void *args = nullptr);
AnfNodePtr GenInferNode(const AnfNodePtr &node);
AnfNodePtr GenInitNode(const AnfNodePtr &node);

struct RelatedCustomActorNode {
  AnfNodePtr infer_node;
  AnfNodePtr init_node;
};

class CustomActorNodeManager {
 public:
  static CustomActorNodeManager &Instance();
  void Reset() { custom_nodes_map_.clear(); }
  void Register(const AnfNodePtr &node, const RelatedCustomActorNode &custom_nodes) {
    (void)custom_nodes_map_.emplace(node, custom_nodes);
  }
  bool IsRegistered(const AnfNodePtr &node) const { return custom_nodes_map_.find(node) != custom_nodes_map_.end(); }
  const RelatedCustomActorNode &GetCustomActorNodes(const AnfNodePtr &node) const {
    if (auto iter = custom_nodes_map_.find(node); iter != custom_nodes_map_.end()) {
      return iter->second;
    }

    MS_LOG(EXCEPTION) << "Not registered node!";
  }

 private:
  CustomActorNodeManager() = default;
  ~CustomActorNodeManager() = default;
  DISABLE_COPY_AND_ASSIGN(CustomActorNodeManager)
  OrderedMap<AnfNodePtr, RelatedCustomActorNode> custom_nodes_map_;
};

/// \brief The class to implement an InferShape function, which is decoupled from the mindspore/core.
class InferShapeFunctor : public Functor {
 public:
  /// \brief Constructor of InferShapeFunctor.
  explicit InferShapeFunctor(const std::string &name) : Functor(name) {}

  /// \brief Destructor of InferShapeFunctor.
  ~InferShapeFunctor() override = default;
  MS_DECLARE_PARENT(InferShapeFunctor, Functor)

  /// \brief Infer output shape.
  /// \param[in] args AbstractBasePtrList of the inputs.
  /// \return Result BaseShapePtr.
  virtual BaseShapePtr InferShape(const AbstractBasePtrList &args) = 0;

  /// \brief Pack functor name to a Value
  /// \return The name of this infershape functor.
  ValuePtr ToValue() const override { return MakeValue(name_); };

  /// \brief Rename the functor.
  void FromValue(const ValuePtr &value) override { name_ = GetValue<std::string>(value); };
};
using InferShapeFunctorPtr = std::shared_ptr<InferShapeFunctor>;
constexpr auto kAttrInferShapeFunctor = "infer_shape_functor";
}  // namespace mindspore::opt::dynamic_shape
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_OPTIMIZER_DYNAMIC_SHAPE_DYNAMIC_SHAPE_HELPER_H
