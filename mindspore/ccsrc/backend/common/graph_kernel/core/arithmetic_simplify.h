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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ARITHMETIC_SIMPLIFY_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ARITHMETIC_SIMPLIFY_H_

#include <memory>
#include <vector>
#include <string>

#include "utils/hash_map.h"
#include "include/backend/optimizer/pass.h"
#include "ir/func_graph.h"
#include "backend/common/graph_kernel/model/lite_graph.h"

namespace mindspore::graphkernel {
class PatternTree;
using PatternTreePtr = std::shared_ptr<PatternTree>;
class ArithmeticSimplify : public opt::Pass {
 public:
  ArithmeticSimplify() : Pass("arithmetic_simplify") {}
  ~ArithmeticSimplify() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  bool DoArithmeticTrans(const inner::LiteGraphPtr &litegraph);
  bool DoConstantFold(const inner::LiteGraphPtr &litegraph);
  mindspore::HashMap<std::string, std::vector<PatternTreePtr>> expressions_map_;
};
using ArithmeticSimplifyPtr = std::shared_ptr<ArithmeticSimplify>;
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ARITHMETIC_SIMPLIFY_H_
