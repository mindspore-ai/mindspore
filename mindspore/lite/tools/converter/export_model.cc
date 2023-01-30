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
#include "tools/converter/export_model.h"
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "backend/common/optimizer/optimizer.h"
#include "include/errorcode.h"
#include "ir/func_graph.h"
#include "tools/lite_exporter/anf_exporter.h"
#include "tools/optimizer/common/pass_manager_extends.h"
#include "tools/converter/graphdef_transform.h"
#include "tools/converter/optimizer_manager.h"
#include "tools/converter/parser/parser_utils.h"
#include "tools/optimizer/graph/control_flow_pass.h"
#include "tools/optimizer/graph/clip_convert_activation_pass.h"
#include "nnacl/op_base.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace lite {
namespace {
using NodesMap = std::map<std::string, std::vector<AnfNodePtr>>;
void CloneGraphInputs(const FuncGraphPtr &origin, const FuncGraphPtr &mirror, NodesMap *origin_map,
                      NodesMap *mirror_map) {
  MS_ASSERT(origin != nullptr && mirror != nullptr);
  MS_ASSERT(origin_map != nullptr && mirror_map != nullptr);
  auto origin_inputs = origin->get_inputs();
  for (auto &input : origin_inputs) {
    auto mirror_input = mirror->add_parameter();
    MS_CHECK_TRUE_RET_VOID(mirror_input != nullptr);
    if (input->abstract() != nullptr) {
      mirror_input->set_abstract(input->abstract()->Clone());
    }
    mirror_input->set_name(input->fullname_with_scope());
    MS_ASSERT(origin_map->find(input->fullname_with_scope()) != origin_map->end());
    MS_ASSERT(mirror_map->find(input->fullname_with_scope()) != mirror_map->end());
    (*origin_map)[input->fullname_with_scope()].push_back(input);
    (*mirror_map)[input->fullname_with_scope()].push_back(mirror_input);
  }
}

bool CheckTupleGetItemSharedWeight(const AnfNodePtr &node, const FuncGraphManagerPtr &manager,
                                   const DataInfo &data_info) {
  if (!utils::isa<ValueNode>(node)) {
    return false;
  }
  for (auto &node_user : manager->node_users()[node]) {
    auto user = node_user.first;
    if (opt::CheckPrimitiveType(user, prim::kPrimTupleGetItem) && data_info.data_.size() >= sizeof(int)) {
      return true;
    }
  }
  return false;
}

AnfNodePtr CloneParameterAndValueNode(const CNodePtr &cnode, size_t index, const FuncGraphPtr &mirror_graph,
                                      const FuncGraphManagerPtr &manager, const std::shared_ptr<ConverterPara> &param) {
  MS_ASSERT(cnode != nullptr && mirror_graph != nullptr);
  MS_CHECK_TRUE_RET(index < cnode->size(), nullptr);
  auto node = cnode->input(index);
  if (node == nullptr || utils::isa<mindspore::CNode>(node)) {
    MS_LOG(ERROR) << "this func cannot copy cnode.";
    return nullptr;
  }
  if (utils::isa<ValueNode>(node)) {
    auto value_node = node->cast<ValueNodePtr>();
    MS_CHECK_TRUE_RET(value_node != nullptr, nullptr);
    auto value_ptr = value_node->value();
    MS_CHECK_TRUE_RET(value_ptr != nullptr, nullptr);
    if (utils::isa<Monad>(value_ptr)) {
      std::shared_ptr<Monad> mirror_monad;
      if (utils::isa<UMonad>(value_ptr)) {
        mirror_monad = std::make_shared<UMonad>();
      } else {
        mirror_monad = std::make_shared<IOMonad>();
      }
      MS_CHECK_TRUE_RET(mirror_monad != nullptr, nullptr);
      auto monad_abs = mirror_monad->ToAbstract();
      MS_CHECK_TRUE_RET(monad_abs != nullptr, nullptr);
      auto mirror_value_node = NewValueNode(mirror_monad);
      MS_CHECK_TRUE_RET(mirror_value_node != nullptr, nullptr);
      mirror_value_node->set_abstract(monad_abs);
      return mirror_value_node;
    }
  }
  DataInfo data_info;
  STATUS status = RET_ERROR;
  if (utils::isa<Parameter>(node)) {
    status = FetchDataFromParameterNode(cnode, index, param->fmk_type, &data_info, true);
  } else if (utils::isa<ValueNode>(node)) {
    status = FetchDataFromValueNode(cnode, index, param->fmk_type, param->train_model, &data_info, true);
  }
  if (status != RET_OK && status != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "fetch data failed.";
    return nullptr;
  }
  if (CheckTupleGetItemSharedWeight(node, manager, data_info)) {
    return NewValueNode(MakeValue<int>(*reinterpret_cast<int *>(data_info.data_.data())));
  }
  ShapeVector shape_vec(data_info.shape_.begin(), data_info.shape_.end());
  if (data_info.data_type_ == kObjectTypeTensorType) {
    shape_vec = ShapeVector{static_cast<int64_t>(data_info.data_.size() / sizeof(int))};
  }
  std::shared_ptr<tensor::Tensor> tensor_info;
  if (static_cast<TensorCompressionType>(data_info.compress_type_) == TensorCompressionType::kNoCompression) {
    tensor_info = std::make_shared<tensor::Tensor>(static_cast<TypeId>(data_info.data_type_), shape_vec);
  } else {
    tensor_info =
      std::make_shared<tensor::Tensor>(static_cast<TypeId>(data_info.data_type_), shape_vec, data_info.data_.size(),
                                       static_cast<TensorCompressionType>(data_info.compress_type_));
  }
  MS_CHECK_TRUE_RET(tensor_info != nullptr, nullptr);
  if (!data_info.data_.empty()) {
    auto tensor_data = reinterpret_cast<uint8_t *>(tensor_info->data_c());
    if (tensor_data == nullptr || tensor_info->data().nbytes() < 0) {
      MS_LOG(ERROR) << "tensor info data is nullptr or the size is smaller than zero.";
      return nullptr;
    }
    if (memcpy_s(tensor_data, tensor_info->data().nbytes(), data_info.data_.data(), data_info.data_.size()) != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed";
      return nullptr;
    }
  }
  auto mirror_parameter = mirror_graph->add_parameter();
  MS_CHECK_TRUE_RET(mirror_parameter != nullptr, nullptr);

  mirror_parameter->set_name(node->fullname_with_scope());
  mirror_parameter->set_default_param(tensor_info);
  mirror_parameter->set_abstract(tensor_info->ToAbstract());
  return mirror_parameter;
}

PrimitivePtr ClonePrimitive(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  auto origin_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (origin_prim == nullptr) {
    return nullptr;
  }
  PrimitivePtr prim;
  auto op_primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
  if (op_primc_fns.find(origin_prim->name()) != op_primc_fns.end()) {
    prim = op_primc_fns[origin_prim->name()]();
    MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  } else {
    prim = std::make_shared<PrimitiveC>(origin_prim->name());
    MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
    prim->set_instance_name(origin_prim->name());
  }
  prim->SetAttrs(origin_prim->attrs());
  if (prim->GetAttr("quant_params") != nullptr) {
    auto quant_holder = prim->GetAttr("quant_params")->cast<QuantParamHolderPtr>();
    prim->AddAttr("quant_params", std::make_shared<QuantParamHolder>(*quant_holder));
  }
  return prim;
}
}  // namespace

FuncGraphPtr CloneFuncGraph(const FuncGraphPtr &graph, const std::shared_ptr<ConverterPara> &param,
                            std::map<FuncGraphPtr, FuncGraphPtr> *cloned_func_graph) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(param != nullptr);
  MS_ASSERT(cloned_func_graph != nullptr);
  auto cloned_func_graph_iter = cloned_func_graph->find(graph);
  if (cloned_func_graph_iter != cloned_func_graph->end()) {
    return cloned_func_graph_iter->second;
  }
  auto mirror_graph = std::make_shared<FuncGraph>();
  MS_CHECK_TRUE_RET(mirror_graph != nullptr, nullptr);
  auto ret = cloned_func_graph->emplace(graph, mirror_graph);
  if (!ret.second) {
    MS_LOG(ERROR) << "emplace mirror graph into map failed.";
    return nullptr;
  }
  mirror_graph->set_attrs(graph->attrs());
  NodesMap origin_nodes;
  NodesMap mirror_nodes;
  CloneGraphInputs(graph, mirror_graph, &origin_nodes, &mirror_nodes);
  auto node_list = TopoSort(graph->get_return());
  auto manager = graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, nullptr);
  for (auto &node : node_list) {
    if (!utils::isa<mindspore::CNode>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    std::vector<AnfNodePtr> node_inputs;
    size_t begin_index = 1;
    auto mirror_prim = ClonePrimitive(cnode);
    if (mirror_prim == nullptr) {
      begin_index = 0;
    }
    for (size_t i = begin_index; i < cnode->size(); ++i) {
      auto origin_input = cnode->input(i);
      MS_CHECK_TRUE_RET(origin_input != nullptr, nullptr);
      AnfNodePtr mirror_input = nullptr;
      auto value = origin_nodes[origin_input->fullname_with_scope()];
      auto iter = std::find(value.begin(), value.end(), origin_input);
      if (iter != value.end()) {
        mirror_input = mirror_nodes[origin_input->fullname_with_scope()][iter - value.begin()];
      }
      if (mirror_input == nullptr) {
        if (IsValueNode<FuncGraph>(origin_input)) {
          auto sub_func_graph = GetValueNode<FuncGraphPtr>(origin_input);
          MS_CHECK_TRUE_RET(sub_func_graph != nullptr, nullptr);
          auto mirror_sub_graph = CloneFuncGraph(sub_func_graph, param, cloned_func_graph);
          mirror_input = NewValueNode(mirror_sub_graph);
        } else {
          mirror_input = CloneParameterAndValueNode(cnode, i, mirror_graph, manager, param);
        }
        if (mirror_input == nullptr) {
          MS_LOG(ERROR) << "node input cannot be found.";
          return nullptr;
        }
        origin_nodes[origin_input->fullname_with_scope()].push_back(origin_input);
        mirror_nodes[origin_input->fullname_with_scope()].push_back(mirror_input);
      }
      node_inputs.push_back(mirror_input);
    }
    auto mirror_cnode =
      mirror_prim == nullptr ? mirror_graph->NewCNode(node_inputs) : mirror_graph->NewCNode(mirror_prim, node_inputs);
    MS_CHECK_TRUE_RET(mirror_cnode != nullptr, nullptr);
    mirror_cnode->set_fullname_with_scope(cnode->fullname_with_scope());
    if (cnode->abstract() != nullptr) {
      mirror_cnode->set_abstract(cnode->abstract()->Clone());
    }
    origin_nodes[cnode->fullname_with_scope()].push_back(cnode);
    mirror_nodes[cnode->fullname_with_scope()].push_back(mirror_cnode);
    if (opt::CheckPrimitiveType(cnode, prim::kPrimReturn)) {
      mirror_graph->set_return(mirror_cnode);
    }
  }
  return mirror_graph;
}

STATUS ExportModel(const FuncGraphPtr &graph, const std::shared_ptr<ConverterPara> &param) {
  CHECK_NULL_RETURN(graph);
  CHECK_NULL_RETURN(param);
  std::map<FuncGraphPtr, FuncGraphPtr> cloned_func_graph;
  auto mirror_graph = CloneFuncGraph(graph, param, &cloned_func_graph);
  if (mirror_graph == nullptr) {
    MS_LOG(ERROR) << "Clone funcGraph failed.";
    return RET_ERROR;
  }
  auto manager = Manage(mirror_graph, true);
  MS_CHECK_TRUE_RET(manager != nullptr, RET_ERROR);
  std::set<FuncGraphPtr> all_func_graphs;
  GetAllFuncGraph(mirror_graph, &all_func_graphs);
  for (auto &func_graph : all_func_graphs) {
    manager->AddFuncGraph(func_graph);
  }
  auto clip_transfer = std::make_shared<opt::ClipConvertActivationPass>();
  CHECK_NULL_RETURN(clip_transfer);
  (void)clip_transfer->Run(mirror_graph);
  if (!RunOptimizerPass(mirror_graph, {"ToNHWCFormat", "InferShapePass", "SpecialNodePostProcess"})) {
    MS_LOG(ERROR) << "Run transpose opt pass failed.";
    return RET_ERROR;
  }
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  CHECK_NULL_RETURN(optimizer);
  auto graph_pm = std::make_shared<opt::LitePassManager>("anf graph pass manager", true);
  CHECK_NULL_RETURN(graph_pm);
  if (param->fmk_type == converter::kFmkTypeTflite || param->fmk_type == converter::kFmkTypeTf ||
      param->fmk_type == converter::kFmkTypeOnnx) {
    graph_pm->AddPass(std::make_shared<opt::ControlFlowPass>());
  }
  optimizer->AddPassManager(graph_pm);
  if (optimizer->Optimize(mirror_graph) == nullptr) {
    MS_LOG(ERROR) << "run  graph pass failed.";
    return RET_ERROR;
  }
  auto meta_graph = Export(mirror_graph);
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "Export to meta graph return nullptr";
    return RET_ERROR;
  }
  auto metagraph_transform = std::make_unique<GraphDefTransform>();
  if (metagraph_transform == nullptr) {
    MS_LOG(ERROR) << "Create metagraph_transform return nullptr";
    delete meta_graph;
    return RET_ERROR;
  }
  metagraph_transform->SetGraphDef(meta_graph);
  auto status = metagraph_transform->Transform(param);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Transform meta graph failed " << status;
    delete meta_graph;
    return RET_ERROR;
  }
  // set output tensor names to the original names, the output_names is null in nnie converter.
  auto output_names = ConverterInnerContext::GetInstance()->GetGraphOutputTensorNames();
  if (output_names.size() > meta_graph->outputIndex.size()) {
    MS_LOG(ERROR) << "the num of setting output_names is greater than actual, " << output_names.size() << " > "
                  << meta_graph->outputIndex.size() << ".";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    delete meta_graph;
    return RET_ERROR;
  }
  for (size_t idx = 0; idx < output_names.size(); idx++) {
    auto &tensor = meta_graph->allTensors.at(meta_graph->outputIndex.at(idx));
    tensor->name = output_names.at(idx);
  }
  meta_graph->version = Version();
  status = MetaGraphSerializer::Save(*meta_graph, "model");
  delete meta_graph;
  std::ostringstream oss;
  if (status != RET_OK) {
    oss << "SAVE GRAPH FAILED:" << status << " " << lite::GetErrorInfo(status);
    MS_LOG(ERROR) << oss.str();
    std::cout << oss.str() << std::endl;
    return status;
  }
  return status;
}
}  // namespace lite
}  // namespace mindspore
