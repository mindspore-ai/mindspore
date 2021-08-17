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

#include "tools/converter/acl/acl_pass.h"
#include <set>
#include "tools/converter/ops/ops_def.h"
#include "tools/common/graph_util.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/acl/deparser/primitive_deparser_register.h"
#include "tools/converter/acl/deparser/spatial_node_adapter.h"
#include "tools/converter/parser/parser_utils.h"
#include "tools/converter/optimizer_manager.h"
#include "include/registry/pass_registry.h"
#include "common/utils.h"
#include "ops/custom.h"
#include "base/core_ops.h"
#include "cxx_api/model/acl/model_converter.h"
#include "backend/kernel_compiler/cpu/nnacl/op_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kMakeTuple = "MakeTuple";
constexpr auto kOutputNames = "outputs_names";
constexpr auto kCustomPrimTypeACL = "ACL";
constexpr auto kCustomNodeName = "Custom";
}  // namespace

ParameterPtr AclPass::CreateOmParameter(const FuncGraphPtr &func_graph, const Buffer &om_data) {
  ParameterPtr om_parameter = func_graph->add_parameter();
  om_parameter->set_name("ACL_om_data");

  auto type_ptr = TypeIdToType(kNumberTypeUInt8);
  ShapeVector shape_vector = {static_cast<int64_t>(om_data.DataSize())};
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
  om_parameter->set_abstract(abstract_tensor);

  auto param_value =
    std::make_shared<tensor::Tensor>(kNumberTypeUInt8, ShapeVector({static_cast<int64_t>(om_data.DataSize())}));
  auto tensor_data = param_value->data_c();
  if (tensor_data == nullptr) {
    MS_LOG(ERROR) << "New Tensor failed.";
    return nullptr;
  }
  if (param_value->Size() < om_data.DataSize()) {
    MS_LOG(ERROR) << "Dst buff size  " << param_value->Size() << " should be greater than src buff size "
                  << om_data.DataSize();
    return nullptr;
  }
  if (memcpy_s(tensor_data, param_value->Size(), om_data.Data(), om_data.DataSize()) != EOK) {
    MS_LOG(ERROR) << "Memcpy om data failed.";
    return nullptr;
  }
  om_parameter->set_default_param(param_value);
  return om_parameter;
}

// now build the whole graph, not split
STATUS AclPass::BuildGraph(const FuncGraphPtr &func_graph) {
  Buffer om_data;
  if (ConvertGraphToOm(func_graph, &om_data) != lite::RET_OK) {
    MS_LOG(ERROR) << "Convert graph  to om failed.";
    return lite::RET_ERROR;
  }
  om_parameter_ = CreateOmParameter(func_graph, om_data);
  if (om_parameter_ == nullptr) {
    MS_LOG(ERROR) << "Convert graph  to om failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS AclPass::RunPrimitiveDeparser(const FuncGraphPtr &func_graph) {
  MS_LOG(INFO) << "Deparser graph start.";
  MS_ASSERT(func_graph != nullptr);
  std::set<FuncGraphPtr> all_func_graphs = {};
  lite::GetAllFuncGraph(func_graph, &all_func_graphs);
  for (auto graph : all_func_graphs) {
    auto node_list = TopoSort(graph->get_return());
    for (auto &node : node_list) {
      if (!utils::isa<CNodePtr>(node)) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      auto prim = GetCNodePrimitive(cnode);
      if (prim == nullptr) {
        MS_LOG(ERROR) << "Prim is nullptr.";
        return lite::RET_ERROR;
      }
      auto name = prim->name();
      auto deparser = lite::PrimitiveDeparserRegister::GetInstance().GetPrimitiveDeparser(name);
      if (deparser == nullptr) {
        MS_LOG(DEBUG) << "Name: " << name << " not need to deparser.";
        continue;
      }
      MS_LOG(INFO) << "Deparser cnode: " << name;
      auto status = deparser->Deparser(cnode);
      if (status != lite::RET_OK) {
        MS_LOG(ERROR) << "Deparser primitive failed.";
        return lite::RET_ERROR;
      }
    }
  }
  return lite::RET_OK;
}

STATUS AclPass::DeparseGraph(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  if (fmk_type_ == converter::kFmkTypeMs) {
    MS_LOG(INFO) << "MindIr no need to deparser graph";
    return lite::RET_OK;
  }
  if (RunPrimitiveDeparser(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Run deparser primitive failed.";
    return lite::RET_ERROR;
  }

  if (lite::AdapteSpatialNode(func_graph, manager) != lite::RET_OK) {
    MS_LOG(ERROR) << "Adapter spatial node failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS AclPass::PreProcGraph(const FuncGraphPtr &func_graph) {
  if (fmk_type_ == converter::kFmkTypeMs) {
    MS_LOG(INFO) << "MindIr no need to pre proc graph";
    return lite::RET_OK;
  }
  // The format of nodes (cnode, parameter, val) must be nchw due to interface of convert om
  if (!lite::RunOptimizerPass(func_graph, {"ToNCHWFormat", "DeleteRedundantTranspose"})) {
    MS_LOG(ERROR) << "To nchw format success.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS AclPass::PostProcGraph(const FuncGraphPtr &func_graph) {
  // The format must be nhwc due to ms model
  if (!lite::RunOptimizerPass(func_graph, {"ToNHWCFormat"})) {
    MS_LOG(ERROR) << "To NHWC Format failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

bool AclPass::Run(const FuncGraphPtr &func_graph) {
  MS_LOG(INFO) << "Acl pass run start.";
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Func_graph is nullptr.";
    return false;
  }
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "Manager is nullptr.";
    return false;
  }

  if (PreProcGraph(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Pre proc graph failed.";
    return false;
  }

  if (DeparseGraph(func_graph, manager) != lite::RET_OK) {
    MS_LOG(ERROR) << "Deparse graph failed.";
    return false;
  }

  if (BuildGraph(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Build graph failed.";
    return false;
  }

  custom_node_ = CreateCustomNode(func_graph);
  if (custom_node_ == nullptr) {
    MS_LOG(ERROR) << "Create custom node failed.";
    return false;
  }
  // prepare graph for export create
  if (ModifyGraphByCustomNode(func_graph, manager, custom_node_) != lite::RET_OK) {
    MS_LOG(ERROR) << "Modify func graph by custom failed.";
    return false;
  }

  if (PostProcGraph(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Post proc graph failed.";
    return false;
  }
  MS_LOG(INFO) << "Acl pass run end.";
  return true;
}

STATUS AclPass::ConvertGraphToOm(const FuncGraphPtr &func_graph, Buffer *om_data) {
  if (om_data == nullptr) {
    MS_LOG(ERROR) << "Om data is nullptr.";
    return lite::RET_ERROR;
  }
  SetAclModelOptions(func_graph);
  // call interface of cloud
  ModelConverter model_converter;
  model_converter.set_options(options_.get());
  *om_data = model_converter.LoadMindIR(func_graph);
  if (om_data->Data() == nullptr || om_data->DataSize() == 0) {
    MS_LOG(ERROR) << "Model converter load mindir failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

void AclPass::SetAclModelOptions(const FuncGraphPtr &func_graph) {
  MS_LOG(INFO) << "Set acl model options start.";
  auto model_context = std::make_shared<mindspore::Context>();
  auto ascend310_info = std::make_shared<Ascend310DeviceInfo>();
  ascend310_info->SetDeviceID(0);
  model_context->MutableDeviceInfo().emplace_back(ascend310_info);
  // set options
  options_ = std::make_unique<AclModelOptions>(model_context);
  if (options_ == nullptr) {
    MS_LOG(ERROR) << "Acl option make shared failed.";
    return;
  }
  auto inputs = func_graph->get_inputs();
  std::vector<std::string> input_names;
  for (auto node : inputs) {
    if (node == nullptr) {
      MS_LOG(ERROR) << "Node is nullptr.";
      return;
    }
    auto para = node->cast<ParameterPtr>();
    if (para == nullptr) {
      MS_LOG(ERROR) << "Parameter is nullptr.";
      return;
    }
    std::string name = para->name();
    for (auto pos = name.find(':'); pos != std::string::npos; pos = name.find(':')) {
      name = name.substr(0, pos) + "_" + name.substr(pos + 1);
      MS_LOG(INFO) << name;
    }
    para->set_name(name);
    input_names.push_back(name);
  }
  options_->RenameInput(input_names);
  MS_LOG(INFO) << "Set acl model options end.";
}

STATUS AclPass::GetFuncGraphOutputInfo(const FuncGraphPtr &func_graph, AnfNodePtrList *graph_outputs,
                                       std::vector<std::string> *graph_output_names,
                                       std::vector<std::vector<int64_t>> *graph_output_dims) {
  CHECK_NULL_RETURN(graph_outputs);
  CHECK_NULL_RETURN(graph_output_names);
  CHECK_NULL_RETURN(graph_output_dims);
  AnfNodePtr return_input = func_graph->output();
  CHECK_NULL_RETURN(return_input);
  auto input_cnode = return_input->cast<CNodePtr>();
  CHECK_NULL_RETURN(input_cnode);
  auto primitive = mindspore::GetValueNode<PrimitivePtr>(input_cnode->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr, node: " << input_cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  // not consider custom op
  std::string primitive_type = primitive->name();
  if (primitive_type == kMakeTuple) {
    for (size_t j = 1; j < input_cnode->inputs().size(); j++) {
      auto item = input_cnode->input(j);
      MS_ASSERT(item != nullptr);
      graph_outputs->emplace_back(item);
      graph_output_names->emplace_back(item->fullname_with_scope());
      auto item_cnode = item->cast<CNodePtr>();
      if (item_cnode == nullptr) {
        MS_LOG(ERROR) << "Input of MakeTuple is not a cnode for input_id: " << j;
        return lite::RET_ERROR;
      }
      std::vector<int64_t> dims;
      if (lite::acl::GetShapeVectorFromCNode(item_cnode, &dims) != lite::RET_OK) {
        MS_LOG(ERROR) << "Get node shape failed.";
        return lite::RET_ERROR;
      }
      graph_output_dims->emplace_back(dims);
    }
  } else {
    graph_outputs->emplace_back(input_cnode);
    graph_output_names->emplace_back(input_cnode->fullname_with_scope());
    std::vector<int64_t> dims;
    if (lite::acl::GetShapeVectorFromCNode(input_cnode, &dims) != lite::RET_OK) {
      MS_LOG(ERROR) << "Get node shape failed.";
      return lite::RET_ERROR;
    }
    graph_output_dims->emplace_back(dims);
  }
  return lite::RET_OK;
}

STATUS AclPass::SetMultiOutputs(const CNodePtr &new_cnode, TypeId data_type) {
  AbstractBasePtrList abstract_list;
  for (size_t j = 0; j < graph_outputs_.size(); j++) {
    auto abstract_tensor = lite::CreateTensorAbstract(graph_outputs_dims_[j], data_type);
    if (abstract_tensor == nullptr) {
      MS_LOG(ERROR) << "Abstract tensor is nullptr for output " << j;
      return lite::RET_ERROR;
    }
    abstract_list.emplace_back(abstract_tensor);
  }
  new_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  return lite::RET_OK;
}

STATUS AclPass::SetCustomOutputs(const FuncGraphPtr &func_graph, const CNodePtr &custom_node) {
  STATUS ret = GetFuncGraphOutputInfo(func_graph, &graph_outputs_, &graph_output_names_, &graph_outputs_dims_);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Get output info of graph failed.";
    return lite::RET_ERROR;
  }
  if (graph_outputs_.empty() || graph_outputs_.size() != graph_outputs_dims_.size()) {
    MS_LOG(ERROR) << "Graph output size is error, num size: " << graph_outputs_.size()
                  << " dim size: " << graph_outputs_dims_.size();
    return lite::RET_ERROR;
  }
  custom_node->AddAttr(kOutputNames, MakeValue(graph_output_names_));

  TypeId type = lite::acl::GetTypeFromNode(graph_outputs_[0]);
  if (graph_outputs_.size() == 1) {
    auto abstract_tensor = lite::CreateTensorAbstract(graph_outputs_dims_[0], type);
    if (abstract_tensor == nullptr) {
      MS_LOG(ERROR) << "Abstract_tensor is nullptr.";
      return lite::RET_ERROR;
    }
    custom_node->set_abstract(abstract_tensor);
    return lite::RET_OK;
  }
  if (SetMultiOutputs(custom_node, type) != lite::RET_OK) {
    MS_LOG(ERROR) << "Set multi graph output failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

CNodePtr AclPass::CreateCustomNode(const FuncGraphPtr &func_graph) {
  auto prim = std::make_unique<mindspore::ops::Custom>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "New custom op failed.";
    return nullptr;
  }
  prim->set_type(kCustomPrimTypeACL);
  auto graph_input = func_graph->get_inputs();
  CNodePtr custom_node = func_graph->NewCNode(std::shared_ptr<ops::PrimitiveC>(prim.release()), graph_input);
  if (custom_node == nullptr) {
    MS_LOG(ERROR) << "Custom cnode failed.";
    return nullptr;
  }
  custom_node->set_fullname_with_scope(kCustomNodeName);
  custom_node->add_input(om_parameter_);

  if (SetCustomOutputs(func_graph, custom_node) != lite::RET_OK) {
    MS_LOG(ERROR) << "Set custom outputs failed.";
    return nullptr;
  }
  return custom_node;
}

STATUS AclPass::ModifyGraphByCustomNode(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager,
                                        const CNodePtr &custom_node) {
  if (graph_outputs_.size() == 1) {
    if (!manager->Replace(graph_outputs_[0], custom_node)) {
      MS_LOG(ERROR) << "Replace node failed.";
      return lite::RET_ERROR;
    }
  } else {
    for (size_t j = 0; j < graph_outputs_.size(); ++j) {
      auto tuple_get_item_prim_ptr = std::make_shared<lite::TupleGetItem>();
      if (tuple_get_item_prim_ptr == nullptr) {
        MS_LOG(ERROR) << "New TupleGetItem failed for output " << j;
        return lite::RET_ERROR;
      }
      auto tuple_get_item_prim = NewValueNode(tuple_get_item_prim_ptr);
      auto get_item_value = NewValueNode(MakeValue<int>(j));
      AnfNodePtrList inputs{tuple_get_item_prim, custom_node, get_item_value};
      CNodePtr get_item_cnode = func_graph->NewCNode(inputs);
      if (get_item_cnode == nullptr) {
        MS_LOG(ERROR) << "New get item cnode failed for output " << j;
        return lite::RET_ERROR;
      }
      get_item_cnode->set_fullname_with_scope(custom_node->fullname_with_scope() + "_getitem_" + std::to_string(j));
      if (!manager->Replace(graph_outputs_[j], get_item_cnode)) {
        MS_LOG(ERROR) << "Replace node failed for output " << j;
        return lite::RET_ERROR;
      }
    }
  }
  return lite::RET_OK;
}
}  //  namespace opt
}  // namespace mindspore
