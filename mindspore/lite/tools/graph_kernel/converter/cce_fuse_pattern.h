/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_TOOLS_CCE_FUSE_PATTERN_H_
#define MINDSPORE_LITE_TOOLS_CCE_FUSE_PATTERN_H_
#include <string>
#include <map>
#include <set>
#include <memory>
#include "ir/func_graph.h"
#include "ir/primitive.h"
#include "ops/nn_ops.h"
#include "ops/array_ops.h"
#include "tools/optimizer/common/pattern_process_pass_extends.h"

namespace mindspore::graphkernel {
class AddReshapeTransposeFusion : public opt::LitePatternProcessPass {
 public:
  explicit AddReshapeTransposeFusion(bool multigraph = true)
      : opt::LitePatternProcessPass("add_reshape_transpose_fusion", multigraph) {
    add_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimAdd->name()));
    reshape_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimReshape->name()));
    trans_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimTranspose->name()));
    InitValidShapes();
  }
  ~AddReshapeTransposeFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
  void InitValidShapes();
  const bool IsValidShape(AnfNodePtr const &node) const;
  VarPtr add_;
  VarPtr reshape_;
  VarPtr trans_;
  std::map<std::string, std::set<ShapeVector>> valid_shape_;
};

}  // namespace mindspore::graphkernel

#endif  // MINDSPORE_LITE_TOOLS_CCE_FUSE_PATTERN_H_
