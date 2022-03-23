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

#define USE_DEPRECATED_API
#include "test/ut/tools/optimizer/fusion/fusion_inout_test/fusion_inout_test.h"
#include <memory>
#include "src/common/log_adapter.h"
#include "tools/common/tensor_util.h"
#include "ops/make_tuple.h"
#include "ops/return.h"
#include "ir/func_graph.h"
#include "ops/fusion/conv2d_fusion.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
FuncGraphPtr FusionInoutTest::Fuse() {
  if (graph_ == nullptr) {
    MS_LOG(WARNING) << "Graph not inited";
    return nullptr;
  }
  if (pass_ == nullptr) {
    MS_LOG(WARNING) << "Pass not inited";
    return graph_;
  }
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  MS_CHECK_TRUE_MSG(optimizer != nullptr, nullptr, "Create GraphOptimizer failed");
  auto fusion_pm = std::make_shared<opt::PassManager>("anf fusion pass manager", false);
  MS_CHECK_TRUE_MSG(fusion_pm != nullptr, nullptr, "Create PassManager failed");

  fusion_pm->AddPass(pass_);
  optimizer->AddPassManager(fusion_pm);
  if (optimizer->Optimize(graph_) == nullptr) {
    MS_LOG(ERROR) << "run op fusion failed.";
    return nullptr;
  }
  return graph_;
}

ParameterPtr FusionInoutTest::AddParameter(const FuncGraphPtr &graph, size_t data_size,
                                           const std::vector<int64_t> &shape, TypeId data_type,
                                           const std::string &name) {
  MS_ASSERT(graph != nullptr);
  auto parameter = graph->add_parameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "CreateParameter failed";
    return nullptr;
  }
  void *data = nullptr;
  if (data_size > 0) {
    data = malloc(data_size);
    if (data == nullptr) {
      MS_LOG(ERROR) << "Malloc tensor data failed";
      return nullptr;
    }
  }
  auto tensor_info = lite::CreateTensorInfo(data, data_size, shape, data_type);

  free(data);
  data = nullptr;

  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "CreateTensorInfo failed";
    return nullptr;
  }
  auto abstract_tensor = tensor_info->ToAbstract();
  if (abstract_tensor == nullptr) {
    MS_LOG(ERROR) << "CreateTensorAbstract failed";
    return nullptr;
  }
  parameter->set_abstract(abstract_tensor);
  if (data_size > 0) {
    parameter->set_default_param(tensor_info);
  }
  parameter->set_name(name);
  return parameter;
}

CNodePtr FusionInoutTest::AddReturn(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &return_inputs) {
  if (return_inputs.empty()) {
    return nullptr;
  }
  AnfNodePtr return_input = nullptr;
  if (return_inputs.size() == 1) {
    return_input = return_inputs.front();
  } else {
    auto make_tuple_prim_ptr = std::make_shared<ops::MakeTuple>();
    if (make_tuple_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new MakeTuple failed";
      return nullptr;
    }
    auto prim_c = make_tuple_prim_ptr->GetPrim();
    MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "prim_c is nullptr");
    auto return_input_cnode = graph->NewCNode(prim_c, return_inputs);
    if (return_input_cnode == nullptr) {
      MS_LOG(ERROR) << "new make tuple cnode failed";
      return nullptr;
    }
    return_input_cnode->set_fullname_with_scope("return tuple");
    return_input = return_input_cnode;
  }

  auto return_prim = std::make_shared<ops::Return>();
  MS_CHECK_TRUE_MSG(return_prim != nullptr, nullptr, "create return primitivec failed");
  auto return_prim_c = return_prim->GetPrim();
  MS_CHECK_TRUE_MSG(return_prim_c != nullptr, nullptr, "prim_c is nullptr");
  auto return_cnode = graph->NewCNode(return_prim_c, {return_input});
  MS_CHECK_TRUE_MSG(return_cnode != nullptr, nullptr, "create Return failed");
  return_cnode->set_fullname_with_scope("Return");
  graph->set_return(return_cnode);
  return return_cnode;
}

std::vector<std::string> FusionInoutTest::GetInputNames() {
  if (graph_ == nullptr) {
    return {};
  }
  auto inputs = graph_->get_inputs();
  std::vector<std::string> ret(inputs.size());
  std::transform(inputs.begin(), inputs.end(), ret.begin(),
                 [](const AnfNodePtr &node) { return node->fullname_with_scope(); });
  return ret;
}

size_t FusionInoutTest::GetOutputNumber() {
  if (graph_ == nullptr) {
    return 0;
  }
  auto ret = graph_->get_return();
  auto ret_input = ret->input(1);
  if (utils::isa<CNodePtr>(ret_input)) {
    auto ret_input_cnode = utils::cast<CNodePtr>(ret_input);
    if (!opt::CheckPrimitiveType(ret_input_cnode, prim::kPrimMakeTuple)) {
      return 1;
    } else {
      return ret_input_cnode->inputs().size() - 1;
    }
  } else {
    return 1;
  }
}

bool FusionInoutTest::DoTest() {
  InitPass();
  InitGraph();
  auto old_inputs = GetInputNames();
  auto old_outputs_num = GetOutputNumber();
  auto ret_graph = Fuse();
  if (ret_graph == nullptr) {
    MS_LOG(ERROR) << "Fusion failed";
    return false;
  }
  auto new_inputs = GetInputNames();
  auto new_outputs_num = GetOutputNumber();
  return old_inputs == new_inputs && old_outputs_num == new_outputs_num;
}
}  // namespace mindspore
