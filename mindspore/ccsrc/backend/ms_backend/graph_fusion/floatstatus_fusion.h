/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_FLOATSTATUS_FUSION__H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_FLOATSTATUS_FUSION__H_

#include <memory>
#include <string>
#include "include/backend/common/pass_manager/optimizer.h"

namespace mindspore::graphkernel {
/**
 * @brief Fuse IsFinite and its user to FloatStatus
 * @example
 *  main_graph {
 *     %1 = IsFinite(%0)
 *     %2 = ReduceAll(%1)
 *     %3 = Cast(%2)
 *     %4 = Sub(1, %3)
 *     return %4
 *   }
 *   ---------->
 *   main_graph {
 *     %1 = FloatStatus(%0)
 *     return %1
 *   }
 */
class FloatStatusBaseFusion : public opt::PatternProcessPass {
 public:
  explicit FloatStatusBaseFusion(const std::string &pass_name, bool multigraph = true)
      : PatternProcessPass(pass_name, multigraph),
        input_{std::make_shared<Var>()},
        axis_{std::make_shared<Var>()},
        keep_dims_{std::make_shared<Var>()},
        type_{std::make_shared<Var>()},
        s_{std::make_shared<Var>()},
        isfinite_prim_{std::make_shared<Var>(std::make_shared<Primitive>("IsFinite"))} {}
  ~FloatStatusBaseFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const override;

 protected:
  VarPtr input_;
  VarPtr axis_;
  VarPtr keep_dims_;
  VarPtr type_;
  VarPtr s_;
  VarPtr isfinite_prim_;
};

/**
 * @brief Fuse IsFinite and its user to FloatStatus
 * @example
 *   main_graph {
 *     %1 = IsFinite(%0)
 *     %2 = ReduceAll(%1)
 *     %3 = Cast(%2)
 *     %4 = Sub(1, %3)
 *     %5 = Reshape(%4, (1,))
 *     return %5
 *   }
 *   ---------->
 *   main_graph {
 *     %1 = FloatStatus(%0)
 *     return %1
 *   }
 */
class FloatStatusReshapeFusion : public FloatStatusBaseFusion {
 public:
  explicit FloatStatusReshapeFusion(const std::string &pass_name, bool multigraph = true)
      : FloatStatusBaseFusion(pass_name, multigraph), to_shape_{std::make_shared<Var>()} {}
  ~FloatStatusReshapeFusion() override = default;
  const BaseRef DefinePattern() const override;

 private:
  VarPtr to_shape_;
};

/**
 * @brief Fuse IsFinite and its user to FloatStatus
 * @example
 *   main_graph {
 *     %1 = Cast(%0)
 *     %2 = IsFinite(%1)
 *     %3 = ReduceAll(%2)
 *     %4 = Cast(%3)
 *     %5 = Sub(1, %4)
 *     return %5
 *   }
 *   ---------->
 *   main_graph {
 *     %1 = FloatStatus(%0)
 *     return %1
 *   }
 */
class CastFloatStatusBaseFusion : public FloatStatusBaseFusion {
 public:
  explicit CastFloatStatusBaseFusion(const std::string &pass_name, bool multigraph = true)
      : FloatStatusBaseFusion(pass_name, multigraph), type_fp32_{std::make_shared<Var>()} {}
  ~CastFloatStatusBaseFusion() override = default;
  const BaseRef DefinePattern() const override;

 private:
  VarPtr type_fp32_;
};

/**
 * @brief Fuse IsFinite and its user to FloatStatus
 * @example
 *   main_graph {
 *     %1 = Cast(%0)
 *     %2 = IsFinite(%1)
 *     %3 = ReduceAll(%2)
 *     %4 = Cast(%3)
 *     %5 = Sub(1, %4)
 *     %6 = Reshape(%5, (1,))
 *     return %6
 *   }
 *   ---------->
 *   main_graph {
 *     %1 = FloatStatus(%0)
 *     return %1
 *   }
 */
class CastFloatStatusReshapeFusion : public CastFloatStatusBaseFusion {
 public:
  explicit CastFloatStatusReshapeFusion(const std::string &pass_name, bool multigraph = true)
      : CastFloatStatusBaseFusion(pass_name, multigraph), to_shape_{std::make_shared<Var>()} {}
  ~CastFloatStatusReshapeFusion() override = default;
  const BaseRef DefinePattern() const override;

 private:
  VarPtr to_shape_;
};

class FloatStatusFusion : public opt::Pass {
 public:
  FloatStatusFusion() : Pass("floatstatus_fusion") {
    cast_floatstatus_reshape_ = std::make_shared<CastFloatStatusReshapeFusion>("cast_floatstatus_reshape_fusion");
    cast_floatstatus_base_ = std::make_shared<CastFloatStatusBaseFusion>("cast_floatstatus_base_fusion");
    floatstatus_reshape_ = std::make_shared<FloatStatusReshapeFusion>("floatstatus_reshape_fusion");
    floatstatus_base_ = std::make_shared<FloatStatusBaseFusion>("floatstatus_base_fusion");
  }
  ~FloatStatusFusion() override = default;
  bool Run(const FuncGraphPtr &func_graph) override {
    cast_floatstatus_reshape_->Run(func_graph);
    cast_floatstatus_base_->Run(func_graph);
    floatstatus_reshape_->Run(func_graph);
    floatstatus_base_->Run(func_graph);
    return true;
  }

 private:
  opt::PassPtr cast_floatstatus_reshape_;
  opt::PassPtr cast_floatstatus_base_;
  opt::PassPtr floatstatus_reshape_;
  opt::PassPtr floatstatus_base_;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_FLOATSTATUS_FUSION__H_
