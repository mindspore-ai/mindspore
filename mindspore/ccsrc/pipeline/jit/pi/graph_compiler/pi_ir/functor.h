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
#ifndef MINDSPORE_PI_JIT_IR_FUNCTOR_H_
#define MINDSPORE_PI_JIT_IR_FUNCTOR_H_

#include <map>
#include <utility>
#include "pipeline/jit/pi/graph_compiler/pi_ir/node.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace pijit {
namespace ir {
template <typename FType>
class NodeFunctor;

template <typename R, typename... Args>
class NodeFunctor<R(const NodePtr &node, Args...)> {
 private:
  /*! \brief internal function pointer type */
  typedef R (*FPointer)(const NodePtr &node, Args...);
  /*! \brief refer to itself. */
  using TSelf = NodeFunctor<R(const NodePtr &node, Args...)>;
  /*! \brief internal function table */
  std::map<uint32_t, FPointer> func_;

 public:
  /*!
   * \brief Whether the functor can dispatch the corresponding Node
   * \param n The node to be dispatched
   * \return Whether dispatching function is registered for n's type.
   */
  bool can_dispatch(const NodePtr &node) const { return func_.find(node->GetClassId()) != func_.end(); }
  /*!
   * \brief invoke the functor, dispatch on type of n
   * \param n The Node argument
   * \return The result.
   */
  R operator()(const NodePtr &node, Args... args) const {
    MS_EXCEPTION_IF_CHECK_FAIL(can_dispatch(node), "NodeFunctor not defined for " + node->GetNodeName() + ".");
    return (*func_.at(node->GetClassId()))(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief set the dispacher for type TNode
   * \param f The function to be set.
   * \tparam TNode the type of Node to be dispatched.
   * \return reference to self.
   */
  template <typename OP>
  TSelf &set_dispatch(FPointer f) {  // NOLINT(*)
    func_[OP::kClassId] = f;
    return *this;
  }
  /*!
   * \brief unset the dispacher for type TNode
   *
   * \tparam TNode the type of Node to be dispatched.
   * \return reference to self.
   */
  template <typename OP>
  TSelf &clear_dispatch() {  // NOLINT(*)
    func_.erase(OP::kClassId);
    return *this;
  }
};

/*! \brief helper macro to suppress unused warning */
#if defined(__GNUC__)
#define IR_ATTRIBUTE_UNUSED __attribute__((unused))
#else
#define IR_ATTRIBUTE_UNUSED
#endif

#define STATIC_IR_FUNCTOR(ClsName, FField) \
  static IR_ATTRIBUTE_UNUSED auto &__make_functor##_##ClsName##__COUNTER__ = ClsName::FField()
}  // namespace ir
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_IR_FUNCTOR_H_
