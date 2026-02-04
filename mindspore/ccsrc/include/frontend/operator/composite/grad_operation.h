/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_FRONTEND_OPERATOR_COMPOSITE_GRAD_OPERATION_H_
#define MINDSPORE_CCSRC_INCLUDE_FRONTEND_OPERATOR_COMPOSITE_GRAD_OPERATION_H_

#include <string>

#include "base/base.h"
#include "ir/meta_func_graph.h"
#include "include/utils/visible.h"

namespace mindspore {
namespace prim {
enum TailType { kGradAll, kGradFirst, kGradByPosition, kNotGrad };

class Tail : public MetaFuncGraph {
 public:
  explicit Tail(const std::string &name, TailType tail_type = kNotGrad, bool return_ids = false)
      : MetaFuncGraph(name), tail_type_(tail_type), enable_tuple_grad_first_(false), return_ids_(return_ids) {}
  ~Tail() override = default;
  MS_DECLARE_PARENT(Tail, MetaFuncGraph)

  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;

  friend bool operator==(const Tail &lhs, const Tail &rhs) { return lhs.name_ == rhs.name_; }
  void set_enable_tuple_grad_first(bool enable_tuple_grad_first) { enable_tuple_grad_first_ = enable_tuple_grad_first; }

 private:
  FuncGraphPtr GenerateTailFuncGraph(const abstract::AbstractSequencePtr &sequence_arg) const;
  FuncGraphPtr GenerateGradFuncGraph(const abstract::AbstractTuplePtr &tuple_arg,
                                     const abstract::AbstractTuplePtr &position = nullptr) const;

  TailType tail_type_;
  bool enable_tuple_grad_first_;
  bool return_ids_;
};
using TailPtr = std::shared_ptr<Tail>;

class FRONTEND_EXPORT GradOperation : public MetaFuncGraph {
 public:
  explicit GradOperation(const std::string &name, bool get_all = false, bool get_by_list = false,
                         bool sens_param = false, bool get_by_position = false, bool has_aux = false,
                         bool get_value = false, bool return_ids = false, bool merge_forward = false);
  ~GradOperation() override = default;
  MS_DECLARE_PARENT(GradOperation, MetaFuncGraph)

  FuncGraphPtr GetGrad(const AnfNodePtr &j, const AnfNodePtr &weights, const AnfNodePtr &position,
                       const FuncGraphPtr &forward_graph, bool is_weights_none) const;

  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;

  bool sens_param() const { return sens_param_; }

  bool get_all_;
  bool get_by_list_;
  bool sens_param_;
  bool get_by_position_;
  bool has_aux_;
  bool get_value_;
  bool return_ids_;
  bool merge_forward_;

 private:
  void GradByParameter(const FuncGraphPtr &k_child, const AnfNodePtr &f_app, const AnfNodePtr &bprop,
                       const AnfNodePtr &weights, const AnfNodePtr &position, const FuncGraphPtr &forward_graph,
                       bool is_weights_none) const;
  CNodePtr SetNodeByParameter(const CNodePtr &grad, const FuncGraphPtr &fg) const;

  AbstractBasePtr weight_value_;
};
using GradOperationPtr = std::shared_ptr<GradOperation>;
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_FRONTEND_OPERATOR_COMPOSITE_GRAD_OPERATION_H_
