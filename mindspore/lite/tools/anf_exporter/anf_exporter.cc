/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "tools/anf_exporter/anf_exporter.h"
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include "abstract/abstract_value.h"
#include "mindspore/core/ir/primitive.h"
#include "ops/fusion/partial_fusion.h"
#include "ops/control_depend.h"
#include "ops/depend.h"
#include "ops/make_tuple.h"
#include "ops/quant_dtype_cast.h"
#include "ops/tuple_get_item.h"
#include "tools/converter/quant_param_holder.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/tensor.h"
#include "src/param_value_lite.h"
#include "src/common/utils.h"
#include "ops/partial.h"
#include "tools/common/graph_util.h"
#include "src/ops/ops_utils.h"

using mindspore::ops::PrimitiveC;

namespace mindspore::lite {
namespace {
std::list<CNodePtr> GetOrderedCNodes(const FuncGraphPtr fg) {
  auto BelongSameGraph = std::bind(IncludeBelongGraph, fg, std::placeholders::_1);
  auto succ_include_fv = [&fg](const AnfNodePtr &node) -> std::vector<AnfNodePtr> {
    std::vector<AnfNodePtr> vecs;
    if (node == nullptr) {
      return vecs;
    }
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      auto &inputs = cnode->inputs();
      // Check if free variables used.
      for (const auto &input : inputs) {
        auto input_fg = GetValueNode<FuncGraphPtr>(input);
        if (input_fg) {
          for (auto &fv : input_fg->free_variables_nodes()) {
            if (fv->func_graph() == fg && fg->nodes().contains(fv)) {
              vecs.push_back(fv);
            }
          }
        }
      }
      (void)vecs.insert(vecs.end(), inputs.begin(), inputs.end());
    }
    return vecs;
  };

  std::list<CNodePtr> cnodes;
  auto nodes = TopoSort(fg->get_return(), succ_include_fv, BelongSameGraph);
  for (const auto &node : nodes) {
    auto cnode = dyn_cast<CNode>(node);
    if (cnode) {
      cnodes.push_back(cnode);
    }
  }
  return cnodes;
}
}  // namespace

void AnfExporter::RemoveIfMakeTuple(const CNodePtr &cnode) {
  bool has_make_tuple = false;
  std::vector<AnfNodePtr> inputs;
  inputs.clear();

  inputs.emplace_back(cnode->input(0));
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    AnfNodePtr input_node = cnode->input(i);
    if (!input_node->isa<CNode>()) {
      inputs.emplace_back(cnode->input(i));
      continue;
    }
    auto make_tuple_node = utils::cast<CNodePtr>(input_node);
    auto value_node = make_tuple_node->input(0)->cast<ValueNodePtr>();
    if (value_node == nullptr) {
      MS_LOG(ERROR) << "value node is invalid.";
      return;
    }
    if (value_node->value() != nullptr && (opt::CheckPrimitiveType(make_tuple_node, opt::kPrimMakeTuple) ||
                                           opt::CheckPrimitiveType(make_tuple_node, opt::kPrimMakeTupleV2))) {
      has_make_tuple = true;
      for (size_t j = 1; j < make_tuple_node->inputs().size(); ++j) {
        inputs.emplace_back(make_tuple_node->input(j));
      }
    } else {
      inputs.emplace_back(cnode->input(i));
    }
  }
  if (has_make_tuple) {
    cnode->set_inputs(inputs);
  }
}

void AnfExporter::RemoveIfDepend(const CNodePtr &cnode) {
  bool has_depend = false;
  std::vector<AnfNodePtr> inputs;
  inputs.clear();

  inputs.emplace_back(cnode->input(0));
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    AnfNodePtr inputNode = cnode->input(i);
    if (!inputNode->isa<CNode>()) {
      inputs.emplace_back(cnode->input(i));
      continue;
    }
    auto depend_node = utils::cast<CNodePtr>(inputNode);
    auto value_node = depend_node->input(0)->cast<ValueNodePtr>();
    if (value_node == nullptr) {
      MS_LOG(ERROR) << "value node is invalid.";
      return;
    }
    if (value_node->value() != nullptr && (opt::CheckPrimitiveType(depend_node, prim::kPrimDepend) ||
                                           opt::CheckPrimitiveType(depend_node, prim::kPrimControlDepend))) {
      has_depend = true;
      bool mask_out = (depend_node->inputs().size() == 3);
      for (size_t j = 1; j < depend_node->inputs().size(); ++j) {
        AnfNodePtr depend_input_node = depend_node->input(j);
        if (depend_input_node->isa<CNode>()) {
          inputs.emplace_back(depend_input_node);
          if (mask_out) {
            break;
          }
        }
      }
    } else {
      inputs.emplace_back(cnode->input(i));
    }
  }
  if (has_depend) {
    cnode->set_inputs(inputs);
  }
}

int AnfExporter::ConvertQuantParam(const std::unique_ptr<schema::MetaGraphT> &meta_graph,
                                   const std::shared_ptr<mindspore::Primitive> &primitive,
                                   const std::unique_ptr<schema::CNodeT> &dst_node) {
  MS_ASSERT(meta_graph != nullptr);
  MS_ASSERT(primitive != nullptr);
  MS_ASSERT(dst_node != nullptr);
  // add quant param
  MS_LOG(DEBUG) << "node: " << dst_node->name << " add QuantParam";
  // activation
  QuantParamsVector input_quant_params;
  QuantParamsVector output_quant_params;
  dst_node->quantType = schema::QuantType_QUANT_NONE;
  auto quant_param_valueptr = primitive->GetAttr("quant_params");
  if (quant_param_valueptr != nullptr) {
    auto quant_param_holder = quant_param_valueptr->cast<QuantParamHolderPtr>();
    if (quant_param_holder == nullptr) {
      MS_LOG(ERROR) << "quant param is invalid.";
      return RET_ERROR;
    }
    input_quant_params = quant_param_holder->input_quant_params();
    output_quant_params = quant_param_holder->output_quant_params();
    dst_node->quantType = quant_param_holder->quant_type();
  }
  // add quant param
  if (!input_quant_params.empty()) {
    for (size_t i = 0; i < input_quant_params.size(); i++) {
      if (i >= dst_node->inputIndex.size()) {
        MS_LOG(INFO) << "node: " << dst_node->name << " input has " << input_quant_params.size()
                     << " quant_params; but only " << dst_node->inputIndex.size() << " input";
        break;
      }
      auto activate_index = dst_node->inputIndex[i];
      auto tensor_input = meta_graph->allTensors[activate_index].get();
      if (tensor_input->quantParams.empty()) {
        for (auto input_quant_param : input_quant_params[i]) {
          std::unique_ptr<schema::QuantParamT> input_quant_param_ptr =
            std::make_unique<schema::QuantParamT>(input_quant_param);
          MS_LOG(DEBUG) << "[input][" << i << "]node: " << dst_node->name << " scale: " << input_quant_param_ptr->scale
                        << " zp: " << input_quant_param_ptr->zeroPoint;
          input_quant_param_ptr->dstDtype = tensor_input->dataType;
          tensor_input->quantParams.emplace_back(std::move(input_quant_param_ptr));
        }
      }
    }
  } else {
    MS_LOG(DEBUG) << "node: " << dst_node->name << " input quant params is empty";
  }
  // output
  if (output_quant_params.empty()) {
    if (primitive->name() != mindspore::ops::kNameQuantDTypeCast) {
      MS_LOG(DEBUG) << "node: " << dst_node->name << " output quant params is empty";
    }
  } else {
    if (dst_node->outputIndex.size() != output_quant_params.size()) {
      MS_LOG(INFO) << "node: " << dst_node->name << " output has " << output_quant_params.size()
                   << " quant_params; but only " << dst_node->outputIndex.size() << " output";
      return RET_ERROR;
    }
    int output_idx = 0;
    for (const auto &output_quant_param : output_quant_params) {
      auto output_tensor = meta_graph->allTensors[dst_node->outputIndex[output_idx]].get();
      output_idx++;
      for (const auto &channel_quant_param : output_quant_param) {
        if (output_tensor->quantParams.empty() && dst_node->quantType != schema::QuantType_WeightQuant) {
          std::unique_ptr<schema::QuantParamT> output_quant_param_ptr =
            std::make_unique<schema::QuantParamT>(channel_quant_param);
          MS_LOG(DEBUG) << "[output]node: " << dst_node->name << " scale: " << output_quant_param_ptr->scale
                        << " zp: " << output_quant_param_ptr->zeroPoint;
          output_quant_param_ptr->dstDtype = output_tensor->dataType;
          output_tensor->quantParams.emplace_back(std::move(output_quant_param_ptr));
        }
      }
    }
  }

  auto first_output_index = dst_node->outputIndex[0];
  auto first_tensor_output = meta_graph->allTensors[first_output_index].get();
  if (dst_node->quantType == schema::QuantType_PostTraining) {
    if (primitive->name() != mindspore::ops::kNameQuantDTypeCast) {
      first_tensor_output->dataType = kNumberTypeInt8;
    } else {
      auto primc = primitive->cast<std::shared_ptr<mindspore::ops::QuantDTypeCast>>();
      MS_ASSERT(primc != nullptr);
      if (primc->get_dst_t() != kNumberTypeFloat32) {
        first_tensor_output->dataType = kNumberTypeInt8;
      }
    }
  }
  return RET_OK;
}

std::vector<schema::CNodeT *> AnfExporter::GetSubgraphNodes(const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                                            const size_t &subgraph_index) {
  std::vector<schema::CNodeT *> subgraph_nodes{};
  subgraph_nodes.resize(meta_graphT->subGraph.at(subgraph_index)->nodeIndices.size());
  std::transform(meta_graphT->subGraph.at(subgraph_index)->nodeIndices.begin(),
                 meta_graphT->subGraph.at(subgraph_index)->nodeIndices.end(), subgraph_nodes.begin(),
                 [&meta_graphT](const uint32_t idx) { return meta_graphT->nodes.at(idx).get(); });
  return subgraph_nodes;
}

int AnfExporter::SetGraphInputIndex(const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                    const size_t &subgraph_index) {
  auto &subgraph = meta_graphT->subGraph.at(subgraph_index);
  auto subgraph_nodes = GetSubgraphNodes(meta_graphT, subgraph_index);
  std::vector<schema::CNodeT *> subgraph_input_nodes{};
  for (auto &node : subgraph_nodes) {
    if (IsContain(graph_input_nodes_, node)) {
      subgraph_input_nodes.push_back(node);
    }
  }
  std::vector<schema::TensorT *> subgraph_inputs{};
  for (auto &node : subgraph_input_nodes) {
    for (auto input : node->inputIndex) {
      auto tensor = meta_graphT->allTensors[input].get();
      if (tensor->nodeType != NodeType_CNode && tensor->data.empty()) {
        tensor->nodeType = NodeType_ValueNode;
        tensor->format = schema::Format_NHWC;
        if (!IsContain(subgraph->inputIndices, input)) {
          if (subgraph_index == kMainGraphIndex) {
            meta_graphT->inputIndex.push_back(input);
          }
          subgraph->inputIndices.push_back(input);
          subgraph_inputs.push_back(tensor);
        }
      }
    }
  }

  return RET_OK;
}

int AnfExporter::SetGraphoutputIndex(const CNodePtr &cnode, const size_t subgraph_index,
                                     const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                     const std::unique_ptr<schema::SubGraphT> &sub_graphT,
                                     schema::CNodeT *return_node) {
  MS_ASSERT(nullptr != meta_graphT);
  MS_ASSERT(nullptr != return_node);
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto input_node = cnode->input(i);
    if (input_node == nullptr) {
      MS_LOG(ERROR) << "output node is nullptr";
      return RET_NULL_PTR;
    } else if (input_node->isa<CNode>()) {
      auto ret = ConvertInputCNode(input_node, return_node);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "obtain outputs failed";
        return ret;
      }
    } else if (input_node->isa<Parameter>()) {
      MS_LOG(INFO) << "the node " << input_node->fullname_with_scope().c_str() << "is parameter node";
      continue;
    } else {
      MS_LOG(ERROR) << "the node " << input_node->fullname_with_scope().c_str() << "is not output node";
      return RET_ERROR;
    }
  }
  for (unsigned int &i : return_node->inputIndex) {
    if (subgraph_index == kMainGraphIndex) {
      meta_graphT->outputIndex.push_back(i);
    }
    meta_graphT->subGraph.at(subgraph_index)->outputIndices.push_back(i);
  }
  return RET_OK;
}

int AnfExporter::Anf2Fb(const FuncGraphPtr &func_graph, const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                        const size_t &subgraph_index, const bool &keep_graph, const bool &copy_primitive,
                        const std::unique_ptr<schema::SubGraphT> &sub_graphT) {
  int ret = RET_OK;
  auto cnodes = GetOrderedCNodes(func_graph);
  for (const auto &cnode : cnodes) {
    auto prim = GetValueNode<std::shared_ptr<Primitive>>(cnode->input(0));
    schema::PrimitiveT *primT = nullptr;
    if (prim == nullptr) {
      auto fg = GetValueNode<FuncGraphPtr>(cnode->input(0));
      if (fg != nullptr) {
        auto partial_cnode = CreatePartialCnode(fg, cnode);
        prim = GetValueNode<std::shared_ptr<Primitive>>(partial_cnode->input(0));
        primT = GetPrimitiveT(partial_cnode->input(0));
        MS_ASSERT(primT != nullptr);
        auto pos = fg_subgraph_map.find(fg);
        if (pos != fg_subgraph_map.end()) {
          MS_ASSERT(primT->value.AsPartialFusion() != nullptr);
          primT->value.AsPartialFusion()->sub_graph_index = fg_subgraph_map.at(fg);
        } else {
          size_t next_subgraph_index = fg_subgraph_map.size() + 1;
          fg_subgraph_map.insert(std::pair<FuncGraphPtr, int>{fg, next_subgraph_index});
          MS_ASSERT(primT->value.AsPartialFusion() != nullptr);
          primT->value.AsPartialFusion()->sub_graph_index = next_subgraph_index;
          ret = ExportSubgraph(fg, meta_graphT, next_subgraph_index, keep_graph, copy_primitive, cnode);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "ExportSubgraph failed";
            break;
          }
        }
      } else {
        MS_LOG(ERROR) << "primitive_c is nullptr";
        ret = RET_MEMORY_FAILED;
        break;
      }
    }

    RemoveIfDepend(cnode);
    if (prim->name() == mindspore::ops::kNameDepend || prim->name() == mindspore::ops::kNameControlDepend ||
        prim->name() == mindspore::ops::kNameTupleGetItem || prim->name() == mindspore::ops::kNameMakeTuple) {
      continue;
    }
    if (prim->name() == "make_tuple") {
      continue;
    }
    RemoveIfMakeTuple(cnode);

    auto node = std::make_unique<schema::CNodeT>();
    if (node == nullptr) {
      MS_LOG(ERROR) << "object failed to be constructed";
      ret = RET_MEMORY_FAILED;
      break;
    }
    if (opt::CheckPrimitiveType(cnode, opt::kPrimReturn)) {
      node->name = mindspore::ops::kNameReturn;
      ret = SetGraphoutputIndex(cnode, subgraph_index, meta_graphT, sub_graphT, node.get());
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "SetOpOutputN failed";
        break;
      }
      continue;
    }
    if (primT == nullptr) {
      primT = GetPrimitiveT(cnode->input(0));
    }
    node->name = cnode->fullname_with_scope();
    node->primitive = std::unique_ptr<schema::PrimitiveT>(primT);
    ret = SetOpInputNode(cnode, meta_graphT, node.get());
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "SetOpInputNode failed";
      break;
    }
    SetOpOutputNode(cnode, meta_graphT, node.get());
    ret = ConvertQuantParam(meta_graphT, prim, node);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ConvertQuantParam failed";
      break;
    }
    meta_graphT->nodes.push_back(std::move(node));
    meta_graphT->subGraph.at(subgraph_index)->nodeIndices.push_back(node_idx++);
  }
  return ret;
}

int AnfExporter::ExportSubgraph(const FuncGraphPtr &func_graph, const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                const size_t &subgraph_index, bool keep_graph, bool copy_primitive,
                                const std::shared_ptr<AnfNode> &partial_anode) {
  int ret = RET_OK;
  meta_graphT->subGraph.emplace_back(std::make_unique<schema::SubGraphT>());
  auto &sub_graphT = meta_graphT->subGraph.at(subgraph_index);
  auto subgraph_name = func_graph->get_attr("graph_name");
  MS_ASSERT(subgraph_name != nullptr);
  sub_graphT->name = GetValue<std::string>(subgraph_name);
  auto fmk = func_graph->get_attr("fmk");
  MS_ASSERT(fmk != nullptr);
  meta_graphT->fmkType = GetValue<int>(fmk);

  ret = Anf2Fb(func_graph, meta_graphT, subgraph_index, keep_graph, copy_primitive, sub_graphT);
  if (ret != RET_OK) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(ret);
    return ret;
  }

  ret = SetGraphInputIndex(meta_graphT, subgraph_index);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SetGraphInputIndex failed";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(ret);
    return ret;
  }

  ret = SetSubgraphTensorIndices(meta_graphT.get());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SetSubgraphTensorIndices failed";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(ret);
    return ret;
  }

  return RET_OK;
}

schema::MetaGraphT *AnfExporter::Export(const FuncGraphPtr &func_graph, bool keep_graph, bool copy_primitive,
                                        bool train_flag) {
  static int subgraph_index = 0;
  this->train_flag = train_flag;
  auto meta_graphT = std::make_unique<schema::MetaGraphT>();
  int ret = ExportSubgraph(func_graph, meta_graphT, subgraph_index, keep_graph, copy_primitive);
  if (ret != RET_OK) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(ret);
    return nullptr;
  }
  return meta_graphT.release();
}

int AnfExporter::ConvertInputCNodeCommonOp(const AnfNodePtr &input_anode, schema::CNodeT *output_cnode) {
  MS_ASSERT(input_anode != nullptr && output_cnode != nullptr);
  auto input_name = input_anode->fullname_with_scope();
  if (this->train_flag) {
    bool found = false;
    if (node_id_map_.find(input_name) != node_id_map_.end()) {
      output_cnode->inputIndex.emplace_back(node_id_map_[input_name]);
      found = true;
    }
    if (!found) {
      auto input_index_key = input_name + "_o:" + std::to_string(0);
      if (node_id_map_.find(input_index_key) != node_id_map_.end()) {
        output_cnode->inputIndex.emplace_back(node_id_map_[input_index_key]);
      }
    }
    return RET_OK;
  }
  if (utils::isa<abstract::AbstractTuple>(input_anode->abstract())) {
    auto tuple = std::reinterpret_pointer_cast<abstract::AbstractTuple>(input_anode->abstract());
    if (tuple == nullptr) {
      MS_LOG(ERROR) << "tuple is nullptr";
      return RET_ERROR;
    }
    auto elements = tuple->elements();
    for (size_t i = 0; i < elements.size(); i++) {
      if (elements.size() == 1) {
        if (node_id_map_.find(input_name) != node_id_map_.end()) {
          output_cnode->inputIndex.emplace_back(node_id_map_[input_name]);
        }
      } else {
        std::string name = input_name + "_o:" + std::to_string(i);
        if (node_id_map_.find(name) != node_id_map_.end()) {
          output_cnode->inputIndex.emplace_back(node_id_map_[name]);
        }
      }
    }
  } else {
    if (node_id_map_.find(input_name) != node_id_map_.end()) {
      output_cnode->inputIndex.emplace_back(node_id_map_[input_name]);
    }
  }
  return RET_OK;
}

int AnfExporter::ConvertInputCNode(const std::shared_ptr<AnfNode> &input_anode, schema::CNodeT *output_cnode) {
  auto input_cnode = utils::cast<CNodePtr>(input_anode);
  auto input_value_node = input_cnode->input(0)->cast<ValueNodePtr>();
  if (input_value_node == nullptr) {
    MS_LOG(ERROR) << "value node is invalid.";
    return RET_ERROR;
  }
  if (input_value_node->value() == nullptr || !opt::CheckPrimitiveType(input_cnode, prim::kPrimTupleGetItem)) {
    return ConvertInputCNodeCommonOp(input_anode, output_cnode);
  } else {
    auto inputs = input_cnode->inputs();

    if (inputs.size() != 3) {
      MS_LOG(ERROR) << "TupleGetItem should have 3 inputs, got " << inputs.size();
      return RET_ERROR;
    }
    auto get_item_input_cnode = inputs.at(1);
    auto index_vnode = inputs.at(2);
    if (!utils::isa<ValueNode>(index_vnode)) {
      MS_LOG(ERROR) << "TupleGetItem's input 2 is not valuenode";
      return RET_ERROR;
    }
    auto value_node = utils::cast<ValueNodePtr>(index_vnode);
    if (value_node == nullptr) {
      MS_LOG(ERROR) << "cast to ValueNode failed";
      return RET_ERROR;
    }
    auto input_index_key = get_item_input_cnode->fullname_with_scope() + "_o:" +
                           std::to_string(value_node->value()->type()->number_type() == kNumberTypeInt64
                                            ? GetValue<int64_t>(value_node->value())
                                            : GetValue<int>(value_node->value()));
    auto iter = node_id_map_.find(input_index_key);
    if (iter == node_id_map_.end()) {
      input_index_key = get_item_input_cnode->fullname_with_scope() + "_o:" + std::to_string(0);  // try name with 0
      iter = node_id_map_.find(input_index_key);
      if (iter == node_id_map_.end()) {
        MS_LOG(ERROR) << "Can not find get_item output tensor " << input_index_key;
        return RET_ERROR;
      }
    }
    output_cnode->inputIndex.emplace_back(iter->second);
  }
  return RET_OK;
}

int AnfExporter::ConvertInputParameter(const std::shared_ptr<AnfNode> &input_anode,
                                       const std::shared_ptr<PrimitiveC> &primitive_c,
                                       const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                       schema::CNodeT *output_cnode) {
  auto paramNode = input_anode->cast<ParameterPtr>();
  std::string input_name = paramNode->fullname_with_scope();
  if (node_id_map_.find(input_name) != node_id_map_.end()) {
    output_cnode->inputIndex.emplace_back(node_id_map_[paramNode->name()]);
    return RET_OK;
  }
  auto paramTensor = std::make_unique<schema::TensorT>();
  paramTensor->format = schema::Format_NHWC;
  paramTensor->name = paramNode->name();
  auto abstractBase = paramNode->abstract();
  if (abstractBase == nullptr) {
    MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << paramNode->name();
    return RET_PARAM_INVALID;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstractBase)) {
    MS_LOG(ERROR) << "Abstract of parameter should be anstract tensor, " << paramNode->name();
    return RET_INPUT_TENSOR_ERROR;
  }
  auto abstractTensor = utils::cast<abstract::AbstractTensorPtr>(abstractBase);
  auto typePtr = abstractTensor->element()->GetTypeTrack();
  MS_ASSERT(typePtr != nullptr);
  paramTensor->dataType = typePtr->type_id();
  if (!utils::isa<abstract::ShapePtr>(abstractTensor->BuildShape())) {
    MS_LOG(ERROR) << "Shape of Abstract of parameter should be ShapePtr, " << paramNode->name();
    return RET_PARAM_INVALID;
  }
  auto shape_vector = utils::cast<abstract::ShapePtr>(abstractTensor->BuildShape())->shape();
  std::vector<int32_t> dims;
  (void)std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(dims),
                       [](const int64_t &value) { return static_cast<int32_t>(value); });
  paramTensor->dims = dims;
  auto paramValue = std::dynamic_pointer_cast<ParamValueLite>(paramNode->default_param());
  if (paramValue != nullptr && paramValue->tensor_size() != 0) {
    paramTensor->data.resize(paramValue->tensor_size());
    paramTensor->format = schema::Format(paramValue->format());
    if (EOK != memcpy_s(paramTensor->data.data(), paramTensor->data.size(), paramValue->tensor_addr(),
                        paramValue->tensor_size())) {
      MS_LOG(ERROR) << "memcpy_s failed.";
      return RET_ERROR;
    }
  }

  paramTensor->name = input_name;
  QuantParamHolderPtr quant_param_holder = primitive_c->GetAttr("quant_params") == nullptr
                                             ? nullptr
                                             : primitive_c->GetAttr("quant_params")->cast<QuantParamHolderPtr>();
  if (quant_param_holder != nullptr && quant_param_holder->enable_huffman_code() &&
      paramTensor->dataType == kNumberTypeInt8) {
    paramTensor->enableHuffmanCode = true;
  }
  node_id_map_[input_name] = meta_graphT->allTensors.size();
  output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
  meta_graphT->allTensors.emplace_back(std::move(paramTensor));
  return RET_OK;
}

int AnfExporter::ProcessTensor(const ValueNodePtr &valueNode, std::unique_ptr<schema::TensorT> *paramTensor,
                               const std::shared_ptr<Value> &value, schema::CNodeT *output_cnode,
                               const std::unique_ptr<schema::MetaGraphT> &meta_graphT) {
  int ret;
  auto valueAbstract = valueNode->abstract();
  auto abstractTensor = utils::cast<abstract::AbstractTensorPtr>(valueAbstract);
  if (abstractTensor == nullptr || abstractTensor->element() == nullptr) {
    MS_LOG(ERROR) << "abstractTensor or abstractTensor->element() is nullptr";
    return RET_ERROR;
  }
  auto typePtr = abstractTensor->element()->GetTypeTrack();
  (*paramTensor)->dataType = typePtr->type_id();
  auto shape_vector = utils::cast<abstract::ShapePtr>(abstractTensor->BuildShape())->shape();
  std::vector<int32_t> dims;
  (void)std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(dims),
                       [](const int64_t &value) { return static_cast<int32_t>(value); });
  (*paramTensor)->dims = dims;
  if (train_flag && (*paramTensor)->dims.empty()) (*paramTensor)->dims = {1};
  (*paramTensor)->nodeType = NodeType_ValueNode;
  auto data = value->cast<tensor::TensorPtr>();
  (*paramTensor)->data.resize(data->Size());
  ret = memcpy_s((*paramTensor)->data.data(), data->Size(), data->data_c(), data->Size());
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy_s error.";
    return RET_ERROR;
  }
  node_id_map_[valueNode->fullname_with_scope()] = meta_graphT->allTensors.size();
  output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
  meta_graphT->allTensors.emplace_back(std::move(*paramTensor));
  return ret;
}
int AnfExporter::ProcessInt32OrInt64Imm(const ValueNodePtr &valueNode, std::unique_ptr<schema::TensorT> *paramTensor,
                                        const std::shared_ptr<Value> &value, schema::CNodeT *output_cnode,
                                        const std::unique_ptr<schema::MetaGraphT> &meta_graphT) {
  int ret;
  // data of int64 is converted to int32 here.
  (*paramTensor)->dataType = kNumberTypeInt32;
  (*paramTensor)->dims = {1};
  (*paramTensor)->nodeType = NodeType_ValueNode;
  int real_data = opt::CastToInt(value).front();
  (*paramTensor)->data.resize(sizeof(int32_t));
  ret = memcpy_s((*paramTensor)->data.data(), sizeof(int32_t), &real_data, sizeof(int32_t));
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy_s error.";
    return RET_ERROR;
  }
  node_id_map_[valueNode->fullname_with_scope()] = meta_graphT->allTensors.size();
  output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
  meta_graphT->allTensors.emplace_back(std::move(*paramTensor));
  return ret;
}
void AnfExporter::ProcessBoolImm(const ValueNodePtr &valueNode, std::unique_ptr<schema::TensorT> *paramTensor,
                                 const std::shared_ptr<Value> &value, schema::CNodeT *output_cnode,
                                 const std::unique_ptr<schema::MetaGraphT> &meta_graphT) {
  auto valueAbstract = valueNode->abstract();
  auto abstractScalar = utils::cast<abstract::AbstractScalarPtr>(valueAbstract);
  auto typePtr = abstractScalar->GetTypeTrack();
  (*paramTensor)->dataType = typePtr->type_id();
  (*paramTensor)->dims = {1};
  (*paramTensor)->nodeType = NodeType_ValueNode;
  auto data = value->cast<mindspore::BoolImmPtr>();
  (*paramTensor)->data.emplace_back(data->value());
  node_id_map_[valueNode->fullname_with_scope()] = meta_graphT->allTensors.size();
  output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
  meta_graphT->allTensors.emplace_back(std::move(*paramTensor));
}
int AnfExporter::ProcessNumber(const ValueNodePtr &valueNode, schema::TensorT *paramTensor,
                               schema::CNodeT *output_cnode, const std::unique_ptr<schema::MetaGraphT> &meta_graphT) {
  auto data = valueNode->value()->cast<NumberPtr>();
  paramTensor->data.resize(sizeof(int));
  int number_type = data->number_type();
  if (EOK != ::memcpy_s(paramTensor->data.data(), sizeof(int), &number_type, sizeof(int))) {
    MS_LOG(ERROR) << "memcpy_s failed";
    return RET_MEMORY_FAILED;
  }
  paramTensor->dataType = kNumberTypeInt32;
  paramTensor->dims = {1};
  paramTensor->nodeType = NodeType_ValueNode;
  node_id_map_[valueNode->fullname_with_scope()] = meta_graphT->allTensors.size();
  output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
  meta_graphT->allTensors.emplace_back(paramTensor);
  return RET_OK;
}
void AnfExporter::ProcessInt(const ValueNodePtr &valueNode, std::unique_ptr<schema::TensorT> *paramTensor,
                             schema::CNodeT *output_cnode, const std::unique_ptr<schema::MetaGraphT> &meta_graphT) {
  (*paramTensor)->dataType = kNumberTypeInt32;
  (*paramTensor)->dims = {1};
  (*paramTensor)->nodeType = NodeType_ValueNode;
  (*paramTensor)->data.emplace_back(kNumberTypeInt32);
  node_id_map_[valueNode->fullname_with_scope()] = meta_graphT->allTensors.size();
  output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
  meta_graphT->allTensors.emplace_back(std::move(*paramTensor));
}
int AnfExporter::ProcessValueSequence(const ValueNodePtr &valueNode, std::unique_ptr<schema::TensorT> *paramTensor,
                                      const std::shared_ptr<Value> &value, schema::CNodeT *output_cnode,
                                      const std::unique_ptr<schema::MetaGraphT> &meta_graphT) {
  int ret = RET_OK;
  auto valueAbstract = valueNode->abstract();
  auto abstractSequnce = utils::cast<abstract::AbstractSequeuePtr>(valueAbstract);
  if (abstractSequnce->isa<abstract::AbstractTuple>()) {
    auto abstractTuple = utils::cast<abstract::AbstractTuplePtr>(valueAbstract);
    auto x_shape_data = abstractTuple->elements();
    std::vector<int32_t> shape;
    for (std::size_t i = 0; i < abstractTuple->size(); ++i) {
      auto value_track = x_shape_data[i]->GetValueTrack();
      MS_ASSERT(value_track != nullptr);
      if (value_track->isa<Int32Imm>()) {
        shape.push_back((GetValue<int>(value_track)));
      } else if (value_track->isa<Int64Imm>()) {
        shape.push_back((GetValue<int64_t>(value_track)));
      } else {
        MS_LOG(ERROR) << "Value type is ValueSequence is not integer, it is " << value_track->ToString() << ".";
        return RET_ERROR;
      }
    }
    (*paramTensor)->dataType = kNumberTypeInt32;
    (*paramTensor)->dims = {static_cast<int32_t>(shape.size())};
    (*paramTensor)->nodeType = NodeType_ValueNode;
    (*paramTensor)->data.resize(shape.size() * sizeof(int));
    ret = memcpy_s((*paramTensor)->data.data(), shape.size() * sizeof(int32_t), shape.data(),
                   shape.size() * sizeof(int32_t));
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "memcpy_s data into paramTensor failed.";
      return RET_ERROR;
    }
    node_id_map_[valueNode->fullname_with_scope()] = meta_graphT->allTensors.size();
    output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
    meta_graphT->allTensors.emplace_back(std::move(*paramTensor));
  }
  return ret;
}
int AnfExporter::ProcessParamValueLite(const ValueNodePtr &valueNode, std::unique_ptr<schema::TensorT> *paramTensor,
                                       const std::shared_ptr<Value> &value, schema::CNodeT *output_cnode,
                                       const std::unique_ptr<schema::MetaGraphT> &meta_graphT) {
  int ret;
  auto valueLite = std::dynamic_pointer_cast<ParamValueLite>(value);
  (*paramTensor)->data.resize(valueLite->tensor_size());
  (*paramTensor)->format = schema::Format(valueLite->format());
  (*paramTensor)->dataType = valueLite->tensor_type();
  (*paramTensor)->dims = valueLite->tensor_shape();

  if (train_flag && (*paramTensor)->dims.empty()) {
    (*paramTensor)->dims = {1};
  }

  ret = memcpy_s((*paramTensor)->data.data(), valueLite->tensor_size() * sizeof(uint8_t), valueLite->tensor_addr(),
                 valueLite->tensor_size());
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy_s data into tensor failed.";
    return RET_ERROR;
  }
  node_id_map_[valueNode->fullname_with_scope()] = meta_graphT->allTensors.size();
  output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
  meta_graphT->allTensors.emplace_back(std::move(*paramTensor));
  return ret;
}

int AnfExporter::ConvertInputValueNode(const std::shared_ptr<AnfNode> &input_anode,
                                       const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                       schema::CNodeT *output_cnode) {
  auto valueNode = input_anode->cast<ValueNodePtr>();
  auto paramTensor = std::make_unique<schema::TensorT>();
  auto value = valueNode->value();
  int ret = RET_OK;
  if (train_flag) {
    paramTensor->name = valueNode->fullname_with_scope();
  }
  if (value->isa<tensor::Tensor>()) {
    ret = ProcessTensor(valueNode, &paramTensor, value, output_cnode, meta_graphT);
  } else if (value->isa<mindspore::Int32Imm>() || value->isa<mindspore::Int64Imm>()) {
    ret = ProcessInt32OrInt64Imm(valueNode, &paramTensor, value, output_cnode, meta_graphT);
  } else if (value->isa<mindspore::BoolImm>()) {
    ProcessBoolImm(valueNode, &paramTensor, value, output_cnode, meta_graphT);
  } else if (value->isa<mindspore::Int>()) {
    ProcessInt(valueNode, &paramTensor, output_cnode, meta_graphT);
  } else if (value->isa<mindspore::ValueSequeue>()) {
    ret = ProcessValueSequence(valueNode, &paramTensor, value, output_cnode, meta_graphT);
  } else if (value->isa<Number>()) {
    ret = ProcessNumber(valueNode, paramTensor.release(), output_cnode, meta_graphT);
  } else if (value->isa<mindspore::ParamValueLite>()) {
    ret = ProcessParamValueLite(valueNode, &paramTensor, value, output_cnode, meta_graphT);
  } else if (value->isa<FuncGraph>()) {
    MS_LOG(INFO) << "op name:" << input_anode->fullname_with_scope() << " input is func_graph";
    return RET_OK;
  } else if (value->isa<Monad>()) {
    MS_LOG(INFO) << "op name:" << input_anode->fullname_with_scope() << " input is Monad";
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "Not support value type , need add support.";
    return RET_ERROR;
  }
  return ret;
}

int AnfExporter::SetOpInputNode(const CNodePtr &cnode, const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                schema::CNodeT *fb_node) {
  MS_ASSERT(nullptr != meta_graphT);
  MS_ASSERT(nullptr != fb_node);
  if (cnode->inputs().size() <= 1) {
    return RET_OK;
  }
  auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "primitive_c is nullptr: " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  bool is_graph_input = false;
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto input_node = cnode->input(i);
    if (input_node->isa<CNode>()) {
      auto ret = ConvertInputCNode(input_node, fb_node);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "ConvertInputCNode failed";
        return ret;
      }
    } else if (input_node->isa<Parameter>()) {
      auto ret = ConvertInputParameter(input_node, primitive_c, meta_graphT, fb_node);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "ConvertInputParameter failed";
        return ret;
      }
      if (!input_node->cast<ParameterPtr>()->has_default()) {
        is_graph_input = true;
      }
    } else if (input_node->isa<ValueNode>()) {
      auto ret = ConvertInputValueNode(input_node, meta_graphT, fb_node);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "ConvertInputValueNode failed";
        return RET_ERROR;
      }
    }
  }
  fb_node->name = cnode->fullname_with_scope();
  if (is_graph_input) {
    graph_input_nodes_.emplace_back(fb_node);
  }
  return RET_OK;
}

void AnfExporter::SetOpOutputNode(const CNodePtr &cnode, const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                  schema::CNodeT *fb_node) {
  MS_ASSERT(nullptr != meta_graphT);
  MS_ASSERT(nullptr != fb_node);
  std::string cnode_name = fb_node->name;

  if (utils::isa<abstract::AbstractTuple>(cnode->abstract())) {
    auto tuple = std::reinterpret_pointer_cast<abstract::AbstractTuple>(cnode->abstract());
    if (tuple == nullptr) {
      MS_LOG(ERROR) << "tuple is nullptr";
      return;
    }
    auto elements = tuple->elements();
    for (size_t i = 0; i < elements.size(); i++) {
      auto msTensor = new (std::nothrow) schema::TensorT();
      if (msTensor == nullptr) {
        MS_LOG(ERROR) << "new msTensor failed";
        return;
      }
      msTensor->nodeType = NodeType_CNode;
      fb_node->outputIndex.emplace_back(meta_graphT->allTensors.size());
      if (train_flag) {
        std::string name = cnode_name + "_o:" + std::to_string(i);
        node_id_map_[name] = meta_graphT->allTensors.size();
        meta_graphT->allTensors.emplace_back(msTensor);
        if (opt::CheckPrimitiveType(cnode, prim::kPrimConv2DFusion) || opt::CheckPrimitiveType(cnode, prim::kPrimAdam))
          break;
      } else {
        if (elements.size() == 1) {
          node_id_map_[cnode_name] = meta_graphT->allTensors.size();
          msTensor->name = cnode_name;
        } else {
          std::string name = cnode_name + "_o:" + std::to_string(i);
          node_id_map_[name] = meta_graphT->allTensors.size();
          msTensor->name = name;
        }

        if (!utils::isa<abstract::AbstractTensorPtr>(elements[i])) {
          MS_LOG(ERROR) << "abstract is not AbstractTensor";
          delete (msTensor);
          return;
        }
        auto type = kNumberTypeFloat32;
        if (utils::isa<abstract::AbstractTensorPtr>(elements[i])) {
          auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(elements[i]);
          auto typePtr = abstract_tensor->element()->GetTypeTrack();
          type = typePtr->type_id();
        }
        msTensor->dataType = type;
        meta_graphT->allTensors.emplace_back(msTensor);
        if (opt::CheckPrimitiveType(cnode, prim::kPrimConv2DFusion) ||
            opt::CheckPrimitiveType(cnode, prim::kPrimFusedBatchNorm)) {
          break;
        }
      }
    }
  } else {
    auto ms_tensor = new (std::nothrow) schema::TensorT();
    if (ms_tensor == nullptr) {
      MS_LOG(ERROR) << "new tensor failed";
      return;
    }
    auto type = kNumberTypeFloat32;
    if (utils::isa<abstract::AbstractTensorPtr>(cnode->abstract())) {
      auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(cnode->abstract());
      auto typePtr = abstract_tensor->element()->GetTypeTrack();
      type = typePtr->type_id();
    }
    ms_tensor->dataType = type;
    ms_tensor->nodeType = NodeType_CNode;
    ms_tensor->name = cnode_name;
    fb_node->outputIndex.emplace_back(meta_graphT->allTensors.size());
    node_id_map_[cnode_name] = meta_graphT->allTensors.size();
    meta_graphT->allTensors.emplace_back(ms_tensor);
  }
}

ValueNodePtr AnfExporter::GetPartialAnfPrim() {
  auto partial_prim = std::make_shared<mindspore::ops::PartialFusion>();
  ValueNodePtr partial_anf_prim = NewValueNode(partial_prim);
  return partial_anf_prim;
}

CNodePtr AnfExporter::CreatePartialCnode(const FuncGraphPtr &fg, AnfNodePtr node) {
  if (utils::isa<CNodePtr>(node)) {
    auto cnode = utils::cast<CNodePtr>(node);
    auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
    if (primitive_c != nullptr) {
      return cnode;
    }
    auto partial_anf_prim_vnode = GetPartialAnfPrim();
    auto cnode_input = cnode->inputs();
    cnode_input.insert(cnode_input.begin(), partial_anf_prim_vnode);
    cnode->set_inputs(cnode_input);
    return cnode;
  } else if (utils::isa<ValueNodePtr>(node)) {
    auto partial_anf_prim_vnode = GetPartialAnfPrim();
    std::vector<AnfNodePtr> inputs{partial_anf_prim_vnode, node};
    auto cnode = fg->NewCNode(inputs);
    return cnode;
  } else {
    MS_LOG(ERROR) << "failed to create partial cnode.";
    return nullptr;
  }
}

schema::MetaGraphT *Export(const FuncGraphPtr &func_graph, bool keep_graph, bool copy_primitive, bool train_flag) {
  AnfExporter anf_exporter;
  return anf_exporter.Export(func_graph, keep_graph, copy_primitive, train_flag);
}
}  // namespace mindspore::lite
