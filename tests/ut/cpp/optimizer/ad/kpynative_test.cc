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
#include <iostream>
#include <unordered_map>

#include "frontend/optimizer/ad/kpynative.h"
#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "ir/manager.h"
#include "ir/value.h"
#include "ir/func_graph_cloner.h"
#include "utils/log_adapter.h"
#include "ir/graph_utils.h"
#include "pipeline/jit/resource.h"
#include "pipeline/jit/parse/parse.h"
#include "pipeline/jit/debug/anf_ir_utils.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace ad {
class TestKPynative : public UT::Common {
 public:
  pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();

 protected:
  AbstractBasePtr BuildArg() {
    std::vector<int64_t> shp = {2, 2};
    tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
    auto abstract = tensor->ToAbstract();
    return abstract;
  }

  FuncGraphPtr BuildPrimalFuncGraph(const std::string &testCase) {
    auto g = std::make_shared<FuncGraph>();
    auto x = g->add_parameter();
    auto y = g->add_parameter();
    x->set_abstract(BuildArg());
    y->set_abstract(BuildArg());
    auto c_node = g->NewCNode({NewValueNode(prim::GetPythonOps("tensor_mul", "mindspore.ops.functional")), x, y});
    c_node->set_abstract(BuildArg());
    g->set_output(c_node);
    return g;
  }

  // a = x * y
  // b = stop_gradient(a)
  // c = b * y
  // return c
  FuncGraphPtr BuildStopGradient(const std::string &testCase) {
    auto g = std::make_shared<FuncGraph>();
    auto x = g->add_parameter();
    x->debug_info()->set_name("x");
    auto y = g->add_parameter();
    y->debug_info()->set_name("y");
    x->set_abstract(BuildArg());
    y->set_abstract(BuildArg());
    auto a_node = g->NewCNode({NewValueNode(prim::GetPythonOps("tensor_mul", "mindspore.ops.functional")), x, y});
    a_node->set_abstract(BuildArg());
    auto b_node = g->NewCNode({NewValueNode(prim::kPrimStopGradient), a_node});
    b_node->set_abstract(BuildArg());
    auto c_node = g->NewCNode({NewValueNode(prim::GetPythonOps("tensor_mul", "mindspore.ops.functional")), b_node, y});
    c_node->set_abstract(BuildArg());
    auto d_node =
      g->NewCNode({NewValueNode(prim::GetPythonOps("tensor_mul", "mindspore.ops.functional")), a_node, c_node});
    d_node->set_abstract(BuildArg());
    g->set_output(d_node);
    return g;
  }

  FuncGraphPtr BuildBpropFuncGraph(const FuncGraphPtr &primal_fg) {
    auto input_params = primal_fg->parameters();
    std::vector<ValuePtr> input_param_values;
    std::for_each(input_params.begin(), input_params.end(),
                  [&](const AnfNodePtr &param) { input_param_values.emplace_back(param->abstract()->BuildValue()); });
    auto k_pynative_cell = GradPynativeCellBegin(input_params, input_param_values);
    auto node_list = TopoSort(primal_fg->output());
    for (auto node : node_list) {
      if (node->isa<CNode>()) {
        auto c_node = node->cast<CNodePtr>();
        auto out = c_node->abstract()->GetValueTrack();
        ValuePtrList args;
        for (size_t i = 1; i < c_node->inputs().size(); ++i) {
          args.push_back(c_node->input(i)->abstract()->GetValueTrack());
        }
        GradPynativeOp(k_pynative_cell, c_node, args, out);
      }
    }
    GradAttr grad_attr(true, false, false, false, true);
    auto bprop_fg = GradPynativeCellEnd(k_pynative_cell, AnfNodePtrList{}, std::vector<size_t>{0}, grad_attr, true);
    return bprop_fg;
  }
};

TEST_F(TestKPynative, test_simple_add) {
  auto primal_fg = BuildPrimalFuncGraph("test_simple_add");
  resource->manager()->KeepRoots({primal_fg});

  auto bprop_fg = BuildBpropFuncGraph(primal_fg);
  resource->manager()->KeepRoots({bprop_fg});
}

TEST_F(TestKPynative, test_stop_gradient) {
  auto primal_fg = BuildStopGradient("test_stop_gradient");
  resource->manager()->KeepRoots({primal_fg});

  auto bprop_fg = BuildBpropFuncGraph(primal_fg);
  resource->manager()->KeepRoots({bprop_fg});
}
}  // namespace ad
}  // namespace mindspore
