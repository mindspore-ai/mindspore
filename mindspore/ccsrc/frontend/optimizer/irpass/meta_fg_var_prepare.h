/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_META_FG_VAR_PREPARE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_META_FG_VAR_PREPARE_H_

#include <memory>

#include "utils/hash_map.h"
#include "frontend/operator/composite/composite.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"

namespace mindspore {
namespace opt {
namespace irpass {
class Matcher {
 public:
  Matcher() {}
  virtual ~Matcher() = default;

  virtual bool Match(const ValuePtr &meta_fg_ptr) const = 0;
  virtual bool Match(const AnfNodePtr &node) const = 0;
};
using MatcherPtr = std::shared_ptr<Matcher>;

// MetaFgMatcher is used to check whether the object is a specific meta_fg_opration
template <typename T>
class MetaFgMatcher : public Matcher {
 public:
  MetaFgMatcher() {}
  ~MetaFgMatcher() override = default;

  bool Match(const ValuePtr &meta_fg_ptr) const override { return meta_fg_ptr->isa<T>(); }

  bool Match(const AnfNodePtr &node) const override {
    if (node == nullptr) {
      return false;
    }
    auto value = GetValueWithoutDoSignature(node);
    if (value == nullptr) {
      return false;
    }
    return value->isa<T>();
  }
};

// Complete the preparation of MetaFuncGraph's variables.
// 1) Handle the varying number of arguments of the MetaFuncGraph.
//    eg.grad(fn)(*args) or vmap(fn)(*args), where fn(*args).
// 2) Handle the case of the sens_param of GradOperation customized by users.
//    eg.GradOperation(sens_param = True)(net)(*real_inputs, sense_para_inputs)
class MetaFgVarPrepare : public AnfVisitor {
 public:
  MetaFgVarPrepare()
      : grad_op_(std::make_shared<MetaFgMatcher<prim::GradOperation>>()),
        unpack_op_(std::make_shared<MetaFgMatcher<prim::UnpackCall>>()) {}
  ~MetaFgVarPrepare() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;

 private:
  MatcherPtr grad_op_;
  MatcherPtr unpack_op_;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_META_FG_VAR_PREPARE_H_
