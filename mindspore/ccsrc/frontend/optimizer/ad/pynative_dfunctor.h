/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_PYNATIVE_D_FUNCTOR_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_PYNATIVE_D_FUNCTOR_H_

#include <memory>
#include <vector>

#include "ir/anf.h"
#include "frontend/optimizer/ad/adjoint.h"

namespace mindspore {
namespace ad {
class PynativeDFunctor {
 public:
  static ValueNodePtr GenNewTensor(const CNodePtr &forward_node);
  static tensor::TensorPtr GenNewTensorInner(const TypePtr &type_elem, const BaseShapePtr &shape_elem);
  static void GetForwardOutNodeAndBpropGraph(const CNodePtr &k_app, CNodePtr *forward_node, FuncGraphPtr *bprop_graph,
                                             FuncGraphPtr *fprop_graph);
  static std::vector<AnfNodePtr> RunOutputReplace(const CNodePtr &forward_node, const FuncGraphPtr &bprop_graph,
                                                  const FuncGraphPtr &fprop_graph, const CNodePtr &cnode_morph);
  static std::vector<AnfNodePtr> RunInputReplace(const FuncGraphPtr &bprop_graph, const FuncGraphPtr &fprop_graph,
                                                 const CNodePtr &cnode_morph);
  static void ReplaceEquivdout(const CNodePtr &k_app, const CNodePtr &cnode_morph);
};
}  // namespace ad
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_PYNATIVE_D_FUNCTOR_H_
