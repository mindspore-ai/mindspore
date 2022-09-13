/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "tools/lite_exporter/anf_exporter.h"
#include <list>
#include <memory>
#include <string>
#include <functional>
#include <utility>
#include <vector>
#include "abstract/abstract_value.h"
#include "mindspore/core/ir/primitive.h"
#include "mindspore/core/ops/op_name.h"
#include "mindspore/core/ops/op_utils.h"
#include "ops/fusion/partial_fusion.h"
#include "ops/call.h"
#include "ops/depend.h"
#include "ops/quant_dtype_cast.h"
#include "ops/make_tuple.h"
#include "ops/return.h"
#include "ops/tuple_get_item.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/quantizer/bitpacking.h"
#include "src/common/utils.h"
#include "tools/common/graph_util.h"
#include "tools/common/meta_graph_utils.h"
#include "src/common/ops/anf_utils.h"
#include "src/litert/tensor_category.h"
#include "src/litert/weight_decoder.h"
#include "tools/common/node_util.h"
#include "src/common/log_util.h"
#include "tools/converter/converter_context.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/converter/quantizer/fse_encoder.h"
#include "tools/converter/quantizer/tensor_compressor.h"
#include "nnacl/op_base.h"

using mindspore::ops::PrimitiveC;

namespace {
constexpr const int kMainGraphIndex = 0;
constexpr const int kFirstDataIndex = 1;
constexpr const int kSecondDataIndex = 2;
constexpr const int kThirdDataIndex = 3;
constexpr const int kPrimIndex = 0;
};  // namespace

namespace mindspore::lite {
namespace {
constexpr int kIndexOfValueInputOfGetTupleItem = 2;
constexpr int kMaxDepth = 2048;
std::list<CNodePtr> GetOrderedCNodes(const FuncGraphPtr fg) {
  MS_CHECK_TRUE_MSG(fg != nullptr, {}, "fg is nullptr.");
  auto BelongSameGraph = std::bind(IncludeBelongGraph, fg, std::placeholders::_1);
  auto succ_include_fv = [&fg](const AnfNodePtr &node) -> std::vector<AnfNodePtr> {
    std::vector<AnfNodePtr> vecs{};
    if (node == nullptr) {
      return vecs;
    }
    if (node->isa<mindspore::CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      MS_ASSERT(cnode != nullptr);
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

  std::list<CNodePtr> cnodes{};
  auto nodes = TopoSort(fg->get_return(), succ_include_fv, BelongSameGraph);
  for (const auto &node : nodes) {
    auto cnode = dyn_cast<mindspore::CNode>(node);
    if (cnode) {
      cnodes.push_back(cnode);
    }
  }
  return cnodes;
}
}  // namespace

int AnfExporter::SetPostTrainOutputTensorType(const std::unique_ptr<schema::MetaGraphT> &meta_graph,
                                              const std::shared_ptr<mindspore::Primitive> &primitive,
                                              const std::unique_ptr<schema::CNodeT> &dst_node) {
  MS_CHECK_TRUE_MSG(dst_node->outputIndex.size() > 0, RET_ERROR, "dst_node->outputIndex is empty.");
  auto first_output_index = dst_node->outputIndex[0];
  MS_CHECK_TRUE_MSG(meta_graph->allTensors.size() > first_output_index, RET_ERROR, "allTensors size is wrong.");
  auto first_tensor_output = meta_graph->allTensors[first_output_index].get();
  if (dst_node->quantType == schema::QuantType_QUANT_ALL) {
    if (primitive->name() != mindspore::ops::kNameQuantDTypeCast) {
      first_tensor_output->dataType = kNumberTypeInt8;
    } else {
      auto primc = api::MakeShared<mindspore::ops::QuantDTypeCast>(primitive);
      MS_CHECK_TRUE_MSG(primc != nullptr, RET_ERROR, "cast ptr failed");
      if (primc->get_dst_t() != kNumberTypeFloat32) {
        first_tensor_output->dataType = kNumberTypeInt8;
      }
    }
  }
  return RET_OK;
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
  auto quant_tensor_info_ptr = primitive->GetAttr("quant_params");
  QuantParamHolderPtr quant_param_holder = nullptr;
  if (quant_tensor_info_ptr == nullptr ||
      (quant_param_holder = quant_tensor_info_ptr->cast<QuantParamHolderPtr>()) == nullptr) {
    quant_param_holder = std::make_shared<QuantParamHolder>(dst_node->inputIndex.size(), dst_node->outputIndex.size());
  }

  CHECK_NULL_RETURN(quant_param_holder);
  input_quant_params = quant_param_holder->get_input_quant_params();
  output_quant_params = quant_param_holder->get_output_quant_params();
  dst_node->quantType = quant_param_holder->quant_type();

  // convert input quant param
  for (size_t i = 0; i < dst_node->inputIndex.size(); i++) {
    if (i >= input_quant_params.size()) {
      MS_LOG(INFO) << "node: " << dst_node->name << " has " << dst_node->inputIndex.size() << " input, but only has "
                   << input_quant_params.size() << " quant params";
      break;
    }
    auto activate_index = dst_node->inputIndex[i];
    MS_CHECK_TRUE_MSG(meta_graph->allTensors.size() > activate_index, RET_ERROR, "allTensors size is wrong.");
    auto tensor_input = meta_graph->allTensors[activate_index].get();
    CHECK_NULL_RETURN(tensor_input);

    tensor_input->quantClusters = quant_param_holder->GetQuantClusters(i);

    if (!quant::TensorQuantParamsInited(*tensor_input)) {
      tensor_input->quantParams.clear();
      for (auto input_quant_param : input_quant_params[i]) {
        auto input_quant_param_ptr = std::make_unique<schema::QuantParamT>(input_quant_param);
        MS_CHECK_TRUE_MSG(input_quant_param_ptr != nullptr, RET_ERROR, "input_quant_param_ptr is nullptr");
        MS_LOG(DEBUG) << "[input][" << i << "]node: " << dst_node->name << " scale: " << input_quant_param_ptr->scale
                      << " zp: " << input_quant_param_ptr->zeroPoint;
        tensor_input->quantParams.emplace_back(std::move(input_quant_param_ptr));
      }
    }
  }

  // output_quant_params
  for (size_t index = 0; index < dst_node->outputIndex.size(); ++index) {
    if (index >= output_quant_params.size()) {
      MS_LOG(INFO) << "node: " << dst_node->name << " has " << dst_node->outputIndex.size() << " output, but only has"
                   << output_quant_params.size() << " quant params";
      break;
    }
    auto output_tensor = meta_graph->allTensors[dst_node->outputIndex[index]].get();
    auto &output_quant_param = output_quant_params[index];
    for (const auto &channel_quant_param : output_quant_param) {
      if (output_tensor->quantParams.empty() && dst_node->quantType != schema::QuantType_QUANT_WEIGHT) {
        std::unique_ptr<schema::QuantParamT> output_quant_param_ptr =
          std::make_unique<schema::QuantParamT>(channel_quant_param);
        CHECK_NULL_RETURN(output_quant_param_ptr);
        MS_LOG(DEBUG) << "[output]node: " << dst_node->name << " scale: " << output_quant_param_ptr->scale
                      << " zp: " << output_quant_param_ptr->zeroPoint;
        output_tensor->quantParams.emplace_back(std::move(output_quant_param_ptr));
      }
    }
  }

  return RET_OK;
}

int AnfExporter::CreateNewTensorForParameter(const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                             const AnfNodePtr &input) {
  lite::DataInfo data_info;
  auto param_node = input->cast<ParameterPtr>();
  MS_CHECK_TRUE_MSG(param_node != nullptr, RET_NULL_PTR, "cast ptr failed");
  if (FetchFromDefaultParam(param_node, converter::FmkType(meta_graphT->fmkType), &data_info, true) != RET_OK) {
    MS_LOG(ERROR) << "FetchFromDefaultParam failed.";
    return RET_ERROR;
  }
  auto schema_tensor = std::make_unique<schema::TensorT>();
  MS_CHECK_TRUE_MSG(schema_tensor != nullptr, RET_ERROR, "schema_tensor is nullptr");
  schema_tensor->format = static_cast<schema::Format>(data_info.format_);
  schema_tensor->name = param_node->name();
  schema_tensor->dims = data_info.shape_;
  schema_tensor->dataType = data_info.data_type_;
  schema_tensor->data = data_info.data_;
  schema_tensor->enableHuffmanCode = data_info.enable_huffman_code_;
  schema_tensor->weightQuantCompressType = static_cast<schema::WeightQuantCompressType>(data_info.compress_type_);
  schema_tensor->nodeType = NodeType_CNode;
  auto key = std::make_pair(input, 0);
  node_id_map_[key] = static_cast<int>(meta_graphT->allTensors.size());
  meta_graphT->allTensors.emplace_back(std::move(schema_tensor));
  return RET_OK;
}

int AnfExporter::SetSubGraphInputIndex(const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                       const size_t &subgraph_index) {
  auto &subgraph = meta_graphT->subGraph.at(subgraph_index);
  FuncGraphPtr fg = nullptr;
  std::for_each(fg_subgraph_map_.begin(), fg_subgraph_map_.end(),
                [&subgraph_index, &fg](const std::pair<const FuncGraphPtr, size_t> &it) {
                  if (it.second == subgraph_index) {
                    fg = it.first;
                  }
                });

  auto inputs = fg->get_inputs();
  for (auto &input : inputs) {
    auto key = std::make_pair(input, 0);
    auto iter = node_id_map_.find(key);
    if (iter != node_id_map_.end()) {
      subgraph->inputIndices.emplace_back(iter->second);
    } else {
      if (CreateNewTensorForParameter(meta_graphT, input) != RET_OK) {
        MS_LOG(ERROR) << "CreateNewTensorForParameter failed.";
        return RET_ERROR;
      }
      subgraph->inputIndices.emplace_back(meta_graphT->allTensors.size() - 1);
    }
  }
  return RET_OK;
}

int AnfExporter::SetSubGraphOutputIndex(const CNodePtr &cnode, const size_t subgraph_index,
                                        const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                        schema::CNodeT *return_node) {
  MS_ASSERT(meta_graphT != nullptr);
  MS_ASSERT(return_node != nullptr);
  for (size_t i = kFirstDataIndex; i < cnode->inputs().size(); i++) {
    auto input_node = cnode->input(i);
    if (input_node == nullptr) {
      MS_LOG(ERROR) << "output node is nullptr";
      return RET_NULL_PTR;
    } else if (input_node->isa<mindspore::CNode>()) {
      auto ret = ConvertInputCNode(input_node, return_node);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "obtain outputs failed";
        return ret;
      }
    } else if (input_node->isa<Parameter>()) {
      auto key = std::make_pair(input_node, 0);
      auto iter = node_id_map_.find(key);
      if (iter != node_id_map_.end()) {
        return_node->inputIndex.emplace_back(iter->second);
      } else {
        if (CreateNewTensorForParameter(meta_graphT, input_node) != RET_OK) {
          MS_LOG(ERROR) << "CreateNewTensorForParameter failed.";
          return RET_ERROR;
        }
        return_node->inputIndex.emplace_back(meta_graphT->allTensors.size() - 1);
      }
    } else {
      MS_LOG(ERROR) << "the node " << input_node->fullname_with_scope().c_str() << "is not output node";
      return RET_ERROR;
    }
  }
  for (unsigned int &i : return_node->inputIndex) {
    meta_graphT->subGraph.at(subgraph_index)->outputIndices.push_back(i);
  }
  return RET_OK;
}

bool AnfExporter::HasExported(const FuncGraphPtr &func_graph) {
  if (fg_subgraph_map_.find(func_graph) != fg_subgraph_map_.end()) {
    return true;
  }
  return false;
}

int AnfExporter::ExportPartialNode(const std::unique_ptr<schema::MetaGraphT> &meta_graphT, const bool &keep_graph,
                                   const bool &copy_primitive, const CNodePtr &partial_cnode,
                                   const std::unique_ptr<schema::CNodeT> &schema_cnode) {
  auto prim = GetValueNode<std::shared_ptr<mindspore::Primitive>>(partial_cnode->input(0));
  MS_CHECK_TRUE_MSG(prim != nullptr, RET_NULL_PTR, "GetValueNode failed");
  if (prim->name() != mindspore::ops::kNamePartialFusion) {
    MS_LOG(INFO) << "not is partial";
    return RET_OK;
  }

  auto partial_fusion_primc = schema_cnode->primitive->value.AsPartialFusion();
  auto vnode = partial_cnode->input(kFirstDataIndex)->cast<ValueNodePtr>();
  MS_CHECK_TRUE_MSG(partial_fusion_primc != nullptr, RET_NULL_PTR, "partial_fusion_primc is invalid");
  MS_CHECK_TRUE_MSG(vnode != nullptr, RET_NULL_PTR, "vnode is invalid");
  auto fg = vnode->value()->cast<FuncGraphPtr>();
  MS_CHECK_TRUE_MSG(fg != nullptr, RET_NULL_PTR, "func graph is nullptr.");
  if (fg_subgraph_map_.find(fg) != fg_subgraph_map_.end()) {
    partial_fusion_primc->sub_graph_index = static_cast<int>(fg_subgraph_map_.at(fg));
    return RET_OK;
  }

  partial_fusion_primc->sub_graph_index = static_cast<int>(meta_graphT->subGraph.size());
  auto ret = ExportSubgraph(fg, meta_graphT, keep_graph, copy_primitive, partial_cnode);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ExportSubgraph failed";
    return ret;
  }
  return RET_OK;
}

std::list<CNodePtr> AnfExporter::InsertCallNode(const FuncGraphPtr &func_graph) {
  auto cnodes = GetOrderedCNodes(func_graph);
  for (auto it = cnodes.begin(); it != cnodes.end();) {
    auto prim = GetValueNode<std::shared_ptr<mindspore::Primitive>>((*it)->input(kPrimIndex));
    if (prim == nullptr) {
      auto fg = GetValueNode<FuncGraphPtr>((*it)->input(kPrimIndex));
      if (fg != nullptr) {
        auto partial_cnode = CreatePartialCnode(fg, (*it));
        auto call_cnode = CreateCallCnode(fg, partial_cnode);
        ++it;
        it = cnodes.insert(it, call_cnode);
        continue;
      } else {
        auto call_anf_prim_vnode = GetCallAnfPrim();
        auto cnode_input = (*it)->inputs();
        cnode_input.insert(cnode_input.begin(), call_anf_prim_vnode);
        (*it)->set_inputs(cnode_input);
      }
    }
    ++it;
  }
  return cnodes;
}

void AnfExporter::SetNonTailCall(const CNodePtr &cnode, schema::CNodeT *node) {
  if (!opt::CheckPrimitiveType(cnode, prim::kPrimCall)) {
    return;
  }
  node->primitive->value.AsCall()->is_tail_call = false;
  call_node_map_[cnode] = node;
  return;
}

int AnfExporter::SetTailCallForReturn(const CNodePtr &return_cnode) {
  auto return_cnode_input_size = return_cnode->inputs().size();
  for (size_t i = 1; i < return_cnode_input_size; ++i) {
    if (!utils::isa<CNodePtr>(return_cnode->input(i))) {
      continue;
    }
    if (!opt::CheckPrimitiveType(return_cnode->input(i), prim::kPrimCall)) {
      continue;
    }
    auto call_cnode = return_cnode->input(i)->cast<CNodePtr>();
    if (call_node_map_.find(call_cnode) == call_node_map_.end()) {
      MS_LOG(ERROR) << "Not found call cnode in call_node_map.";
      return RET_ERROR;
    }
    call_node_map_[call_cnode]->primitive->value.AsCall()->is_tail_call = true;
  }
  return RET_OK;
}

int AnfExporter::SetTailCallForNonOutput() {
  for (auto item : call_node_map_) {
    auto call_cnode = item.first;
    auto mg = call_cnode->func_graph()->manager();
    if (mg == nullptr) {
      MS_LOG(ERROR) << "manager is nullptr.";
      return RET_NULL_PTR;
    }
    auto node_user = mg->node_users()[call_cnode];
    if (node_user.empty()) {
      (item.second)->primitive->value.AsCall()->is_tail_call = true;
    }
  }
  return RET_OK;
}

bool AnfExporter::CaseToContinue(const string &prim_name) {
  return prim_name == mindspore::ops::kNameDepend || prim_name == mindspore::ops::kNameTupleGetItem ||
         prim_name == mindspore::ops::kNameMakeTuple || prim_name == "make_tuple";
}

int AnfExporter::Anf2Fb(const FuncGraphPtr &func_graph, const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                        const size_t &subgraph_index, const bool &keep_graph, const bool &copy_primitive) {
  int ret = RET_OK;
  auto cnodes = InsertCallNode(func_graph);
  for (const auto &cnode : cnodes) {
    auto prim = GetValueNode<std::shared_ptr<mindspore::Primitive>>(cnode->input(kPrimIndex));
    if (prim == nullptr) {
      MS_LOG(ERROR) << "get prim from value node failed.";
      return RET_ERROR;
    }
    std::unique_ptr<schema::PrimitiveT> primT;

    ret = RemoveIfDepend(cnode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "RemoveIfDepend failed";
      break;
    }
    if (CaseToContinue(prim->name())) {
      continue;
    }
    ret = RemoveIfMakeTuple(cnode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "RemoveIfMakeTuple failed";
      break;
    }
    auto node = std::make_unique<schema::CNodeT>();
    if (node == nullptr) {
      MS_LOG(ERROR) << "object failed to be constructed";
      ret = RET_MEMORY_FAILED;
      break;
    }
    if (opt::CheckPrimitiveType(cnode, prim::kPrimReturn)) {
      node->name = mindspore::ops::kNameReturn;
      ret = SetSubGraphOutputIndex(cnode, subgraph_index, meta_graphT, node.get());
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "SetOpOutputN failed";
        break;
      }
      ret = SetTailCallForReturn(cnode);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "SetTailCallForReturn failed";
        return ret;
      }

      continue;
    }
    primT = GetPrimitiveT(cnode->input(kPrimIndex));
    node->name = cnode->fullname_with_scope();
    node->primitive = std::move(primT);
    auto device_type_attr = cnode->GetAttr(mindspore::ops::kDeviceType);
    node->deviceType = (device_type_attr != nullptr) ? GetValue<int32_t>(device_type_attr) : -1;
    ret = SetOpInputNode(cnode, meta_graphT, node.get());
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "SetOpInputNode failed";
      break;
    }
    // set all call op to non tail call
    SetNonTailCall(cnode, node.get());

    ret = ExportPartialNode(meta_graphT, keep_graph, copy_primitive, cnode, node);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ExportPartialNode failed.";
      return ret;
    }

    ret = SetOpOutputNode(cnode, meta_graphT, node.get());
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "SetOpOutputNode failed";
      break;
    }
    ret = ConvertQuantParam(meta_graphT, prim, node);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ConvertQuantParam failed";
      break;
    }

    auto status = SetPostTrainOutputTensorType(meta_graphT, prim, node);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "set quant output tensor data type failed.";
      break;
    }

    meta_graphT->nodes.push_back(std::move(node));
    meta_graphT->subGraph.at(subgraph_index)->nodeIndices.push_back(node_idx_++);
  }
  return ret;
}

int AnfExporter::ExportSubgraph(const FuncGraphPtr &func_graph, const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                bool keep_graph, bool copy_primitive, const std::shared_ptr<AnfNode> &partial_anode) {
  if (HasExported(func_graph)) {
    MS_LOG(INFO) << "Has been exported.";
    return RET_OK;
  }

  auto subgraph_ptr = std::make_unique<schema::SubGraphT>();
  CHECK_NULL_RETURN(subgraph_ptr);
  meta_graphT->subGraph.emplace_back(std::move(subgraph_ptr));
  auto subgraph_index = meta_graphT->subGraph.size() - 1;
  fg_subgraph_map_[func_graph] = subgraph_index;
  auto subgraph_name = func_graph->get_attr("graph_name");
  MS_CHECK_TRUE_MSG(subgraph_name != nullptr, RET_ERROR, "subgraph_name is nullptr");
  meta_graphT->subGraph.back()->name =
    "subgraph_" + std::to_string(meta_graphT->subGraph.size() - 1) + "_" + GetValue<std::string>(subgraph_name);

  auto ret = Anf2Fb(func_graph, meta_graphT, subgraph_index, keep_graph, copy_primitive);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Anf2Fb failed";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(ret);
    return ret;
  }

  ret = SetSubGraphInputIndex(meta_graphT, subgraph_index);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SetSubGraphInputIndex failed";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(ret);
    return ret;
  }

  SetSubgraphTensorIndices(meta_graphT.get());

  return RET_OK;
}

FuncGraphPtr GetFinalGraph(const FuncGraphPtr &func_graph, int i) {
  if (i > kMaxDepth) {
    MS_LOG(ERROR) << "exceed max depth 2048, i " << i;
    return nullptr;
  }
  i++;
  // get output
  CNodePtr call_cnode = nullptr;
  auto fg_output = func_graph->output();
  if (opt::CheckPrimitiveType(fg_output, prim::kPrimCall)) {
    call_cnode = fg_output->cast<CNodePtr>();
  } else {
    return func_graph;
  }

  // if call input is switch, meta output is call switch false partial's fg'output!
  auto cnode = call_cnode->input(kFirstDataIndex)->cast<CNodePtr>();
  if (IsSwitch(cnode)) {
    auto false_cnode = cnode->input(kThirdDataIndex)->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(false_cnode != nullptr, nullptr, "cast failed");
    auto false_fg = GetValueNode<FuncGraphPtr>(false_cnode->input(kFirstDataIndex));
    MS_CHECK_TRUE_MSG(false_fg != nullptr, nullptr, "GetValueNode failed");
    return GetFinalGraph(false_fg, i);
  } else if (IsSwitchLayer(cnode)) {
    auto first_partial_cnode = cnode->input(kSecondDataIndex)->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(first_partial_cnode != nullptr, nullptr, "cast failed");
    auto next_fg = GetValueNode<FuncGraphPtr>(first_partial_cnode->input(kFirstDataIndex));
    MS_CHECK_TRUE_MSG(next_fg != nullptr, nullptr, "GetValueNode failed");
    return GetFinalGraph(next_fg, i);
  } else {
    auto fg = GetValueNode<FuncGraphPtr>(cnode->input(kFirstDataIndex));
    MS_CHECK_TRUE_MSG(fg != nullptr, nullptr, "GetValueNode failed");
    return GetFinalGraph(fg, i);
  }
}

int AnfExporter::SetMetaGraphInput(const FuncGraphPtr &func_graph,
                                   const std::unique_ptr<schema::MetaGraphT> &meta_graphT) {
  MS_ASSERT(func_graph != nullptr);
  meta_graphT->inputIndex.clear();
  for (const auto &input : func_graph->get_inputs()) {
    auto iter = graph_inputs_map_.find(input);
    if (iter == graph_inputs_map_.end()) {
      MS_LOG(ERROR) << "input " << input->ToString() << " not found in graph" << std::endl;
      return RET_ERROR;
    }
    meta_graphT->inputIndex.emplace_back(iter->second);
  }
  return RET_OK;
}

int AnfExporter::SetMetaGraphOutput(const FuncGraphPtr &func_graph,
                                    const std::unique_ptr<schema::MetaGraphT> &meta_graphT) {
  FuncGraphPtr final_fg = nullptr;
  if (meta_graphT->fmkType == converter::kFmkTypeMs) {
    final_fg = func_graph;
  } else {
    int i = 0;
    final_fg = GetFinalGraph(func_graph, i);
  }
  MS_CHECK_TRUE_MSG(final_fg != nullptr, RET_ERROR, "GetFinalGraph failed.");
  auto final_meta_graph_index = fg_subgraph_map_.at(final_fg);
  auto &final_meta_graph = meta_graphT->subGraph.at(final_meta_graph_index);
  meta_graphT->outputIndex.assign(final_meta_graph->outputIndices.begin(), final_meta_graph->outputIndices.end());

  for (auto &output_index : meta_graphT->outputIndex) {
    auto &tensor = meta_graphT->allTensors.at(output_index);
    ConverterInnerContext::GetInstance()->UpdateGraphOutputDType(meta_graphT->outputIndex.size(), tensor->dataType);
  }

  return RET_OK;
}

schema::MetaGraphT *AnfExporter::Export(const FuncGraphPtr &func_graph, bool keep_graph, bool copy_primitive,
                                        bool train_flag) {
  this->train_flag_ = train_flag;
  // hardcode for nnie and train
  this->graph_inputs_map_.clear();
  auto meta_graphT = std::make_unique<schema::MetaGraphT>();
  MS_CHECK_TRUE_MSG(meta_graphT != nullptr, nullptr, "meta_graphT is nullptr");
  auto fmk = func_graph->get_attr("fmk");
  MS_CHECK_TRUE_MSG(fmk != nullptr, nullptr, "fmk is nullptr");
  if (fmk->isa<Int64Imm>()) {
    meta_graphT->fmkType = GetValue<int64_t>(fmk);
  } else {
    meta_graphT->fmkType = GetValue<int>(fmk);
  }

  graph_inputs_ = func_graph->get_inputs();

  int ret = ExportSubgraph(func_graph, meta_graphT, keep_graph, copy_primitive);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Export subgraph failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(ret);
    return nullptr;
  }

  ret = SetTailCallForNonOutput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SetTailCallForNonOutput failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(ret);
    return nullptr;
  }

  ret = SetMetaGraphInput(func_graph, meta_graphT);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SetMetaGraphInput failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(ret);
    return nullptr;
  }
  ret = SetMetaGraphOutput(func_graph, meta_graphT);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SetMetaGraphOutput failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(ret);
    return nullptr;
  }

  return meta_graphT.release();
}

int AnfExporter::ConvertInputCNodeCommonOp(const AnfNodePtr &input_anode, schema::CNodeT *output_cnode) {
  MS_ASSERT(input_anode != nullptr && output_cnode != nullptr);
  if (this->train_flag_) {
    auto key = std::make_pair(input_anode, 0);
    if (node_id_map_.find(key) != node_id_map_.end()) {
      output_cnode->inputIndex.emplace_back(node_id_map_[key]);
    }
    return RET_OK;
  }
  if (utils::isa<abstract::AbstractTuple>(input_anode->abstract())) {
    auto tuple = std::reinterpret_pointer_cast<abstract::AbstractTuple>(input_anode->abstract());
    MS_CHECK_TRUE_MSG(tuple != nullptr, RET_ERROR, "tuple is nullptr");
    auto elements = tuple->elements();
    for (size_t i = 0; i < elements.size(); i++) {
      auto key = std::make_pair(input_anode, i);
      if (node_id_map_.find(key) != node_id_map_.end()) {
        output_cnode->inputIndex.emplace_back(node_id_map_[key]);
      }
    }
  } else {
    auto key = std::make_pair(input_anode, 0);
    if (node_id_map_.find(key) != node_id_map_.end()) {
      output_cnode->inputIndex.emplace_back(node_id_map_[key]);
    }
  }
  return RET_OK;
}

int AnfExporter::ConvertInputCNode(const std::shared_ptr<AnfNode> &input_anode, schema::CNodeT *output_cnode) {
  auto input_cnode = utils::cast<CNodePtr>(input_anode);
  MS_CHECK_TRUE_MSG(input_cnode != nullptr, RET_ERROR, "cast ptr failed");
  auto input_value_node = input_cnode->input(kPrimIndex)->cast<ValueNodePtr>();
  if (input_value_node == nullptr) {
    if (!IsCall(input_cnode)) {
      MS_LOG(ERROR) << "value node is invalid.";
      return RET_ERROR;
    } else {
      auto call_anf_prim_vnode = GetCallAnfPrim();
      auto cnode_input = input_cnode->inputs();
      MS_CHECK_TRUE_MSG(call_anf_prim_vnode != nullptr, RET_ERROR, "GetCallAnfPrim failed");
      cnode_input.insert(cnode_input.begin(), call_anf_prim_vnode);
      input_cnode->set_inputs(cnode_input);
    }
  }

  input_value_node = input_cnode->input(kPrimIndex)->cast<ValueNodePtr>();

  if (input_value_node->value() == nullptr || !opt::CheckPrimitiveType(input_cnode, prim::kPrimTupleGetItem)) {
    return ConvertInputCNodeCommonOp(input_anode, output_cnode);
  } else {
    auto inputs = input_cnode->inputs();

    if (inputs.size() != 3) {
      MS_LOG(ERROR) << "TupleGetItem should have 3 inputs, got " << inputs.size();
      return RET_ERROR;
    }
    auto get_item_input_cnode = inputs.at(1);
    auto index_vnode = inputs.at(kIndexOfValueInputOfGetTupleItem);
    if (!utils::isa<ValueNode>(index_vnode)) {
      MS_LOG(ERROR) << "TupleGetItem's input 2 is not valuenode";
      return RET_ERROR;
    }
    auto value_node = utils::cast<ValueNodePtr>(index_vnode);
    MS_CHECK_TRUE_MSG(value_node != nullptr, RET_ERROR, "cast to ValueNode failed");
    auto idx = value_node->value()->type()->number_type() == kNumberTypeInt64 ? GetValue<int64_t>(value_node->value())
                                                                              : GetValue<int>(value_node->value());
    auto key = std::make_pair(get_item_input_cnode, idx);
    auto iter = node_id_map_.find(key);
    if (iter == node_id_map_.end()) {
      key = std::make_pair(get_item_input_cnode, 0);  // try name with 0
      iter = node_id_map_.find(key);
      if (iter == node_id_map_.end()) {
        MS_LOG(ERROR) << "Can not find get_item output tensor "
                      << get_item_input_cnode->fullname_with_scope() + "_o:" + std::to_string(idx);
        return RET_ERROR;
      }
    }
    output_cnode->inputIndex.emplace_back(iter->second);
  }
  return RET_OK;
}

int AnfExporter::ConvertInputParameter(const CNodePtr &cnode, size_t index, const PrimitivePtr &primitive,
                                       const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                       schema::CNodeT *op_node) {
  auto param_node = cnode->input(index)->cast<ParameterPtr>();
  MS_ASSERT(param_node != nullptr);
  auto key = std::make_pair(param_node, 0);
  if (node_id_map_.find(key) != node_id_map_.end()) {
    op_node->inputIndex.emplace_back(node_id_map_[key]);
    return RET_OK;
  }
  DataInfo data_info;
  if (FetchDataFromParameterNode(cnode, index, converter::FmkType(meta_graphT->fmkType), &data_info, true) != RET_OK) {
    MS_LOG(ERROR) << "parse const node failed.";
    return RET_ERROR;
  }
  auto schema_tensor = std::make_unique<schema::TensorT>();
  MS_CHECK_TRUE_MSG(schema_tensor != nullptr, RET_ERROR, "schema_tensor is nullptr");
  schema_tensor->format = static_cast<schema::Format>(data_info.format_);
  schema_tensor->name = param_node->name();
  schema_tensor->dims = data_info.shape_;
  schema_tensor->dataType = data_info.data_type_;
  schema_tensor->data = data_info.data_;
  if (param_node->has_default()) {
    schema_tensor->nodeType = NodeType_ValueNode;
  } else {
    schema_tensor->nodeType = NodeType_CNode;
  }
  schema_tensor->enableHuffmanCode = data_info.enable_huffman_code_;
  schema_tensor->weightQuantCompressType =
    static_cast<mindspore::schema::WeightQuantCompressType>(data_info.compress_type_);
  node_id_map_[key] = meta_graphT->allTensors.size();
  op_node->inputIndex.emplace_back(meta_graphT->allTensors.size());
  meta_graphT->allTensors.emplace_back(std::move(schema_tensor));
  return RET_OK;
}

int AnfExporter::ConvertInputValueNode(const CNodePtr &cnode, size_t index, const PrimitivePtr &primitive,
                                       const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                       schema::CNodeT *op_node) {
  DataInfo data_info;
  auto status =
    FetchDataFromValueNode(cnode, index, converter::FmkType(meta_graphT->fmkType), train_flag_, &data_info, true);
  if (status == RET_NO_CHANGE) {
    return RET_OK;
  }
  if (status != RET_OK) {
    MS_LOG(ERROR) << "parse value node failed.";
    return status;
  }
  auto schema_tensor = std::make_unique<schema::TensorT>();
  MS_CHECK_TRUE_MSG(schema_tensor != nullptr, RET_ERROR, "schema is nullptr");
  schema_tensor->name = cnode->input(index)->fullname_with_scope();
  schema_tensor->format = static_cast<schema::Format>(data_info.format_);
  schema_tensor->dataType = data_info.data_type_;
  schema_tensor->dims = data_info.shape_;
  schema_tensor->data = data_info.data_;

  auto key = std::make_pair(cnode->input(index), 0);
  node_id_map_[key] = meta_graphT->allTensors.size();
  op_node->inputIndex.emplace_back(meta_graphT->allTensors.size());
  meta_graphT->allTensors.emplace_back(std::move(schema_tensor));
  return RET_OK;
}

int AnfExporter::SetOpInputNode(const CNodePtr &cnode, const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                schema::CNodeT *fb_node) {
  MS_ASSERT(meta_graphT != nullptr);
  MS_ASSERT(fb_node != nullptr);
  if (cnode->inputs().size() <= 1) {
    return RET_OK;
  }
  auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "primitive_c is nullptr: " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto input_node = cnode->input(i);
    if (input_node->isa<mindspore::CNode>()) {
      auto ret = ConvertInputCNode(input_node, fb_node);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "ConvertInputCNode failed";
        return ret;
      }
    } else if (input_node->isa<Parameter>()) {
      auto ret = ConvertInputParameter(cnode, i, primitive_c, meta_graphT, fb_node);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "ConvertInputParameter failed";
        return ret;
      }
      if (IsContain(graph_inputs_, input_node->cast<AnfNodePtr>()) &&
          graph_inputs_has_exported_.find(input_node) == graph_inputs_has_exported_.end()) {
        graph_inputs_has_exported_.insert(input_node);
        graph_inputs_map_[input_node] = meta_graphT->allTensors.size() - 1;
      }
    } else if (input_node->isa<ValueNode>()) {
      auto ret = ConvertInputValueNode(cnode, i, primitive_c, meta_graphT, fb_node);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "ConvertInputValueNode failed";
        return RET_ERROR;
      }
    }
  }
  fb_node->name = cnode->fullname_with_scope();
  return RET_OK;
}

int AnfExporter::SetOpOutputNode(const CNodePtr &cnode, const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                 schema::CNodeT *fb_node) {
  MS_ASSERT(meta_graphT != nullptr);
  MS_ASSERT(fb_node != nullptr);
  std::string cnode_name = fb_node->name;

  // new anf export and import will add abstract tuple for control flow op, which contains abstract closure,
  // abstract tuple and abstract tensor. For inference, we don't need this information. So skip export abstract tuple
  // for control flow op. Just use a abstract tensor link the control flow ops.
  if (utils::isa<abstract::AbstractTuple>(cnode->abstract()) && !IsControlFlowOp(cnode)) {
    auto tuple = std::reinterpret_pointer_cast<abstract::AbstractTuple>(cnode->abstract());
    MS_CHECK_TRUE_MSG(tuple != nullptr, RET_ERROR, "tuple is nullptr");
    auto elements = tuple->elements();
    for (size_t i = 0; i < lite::GetCNodeOutputsSize(cnode, train_flag_); i++) {
      auto ms_tensor = new (std::nothrow) schema::TensorT();
      if (ms_tensor == nullptr) {
        MS_LOG(ERROR) << "new msTensor failed";
        return RET_ERROR;
      }
      ms_tensor->nodeType = NodeType_CNode;
      fb_node->outputIndex.emplace_back(meta_graphT->allTensors.size());
      auto key = std::make_pair(cnode, i);
      if (train_flag_) {
        node_id_map_[key] = meta_graphT->allTensors.size();
        meta_graphT->allTensors.emplace_back(ms_tensor);
      } else {
        auto val_ptr = cnode->GetAttr("outputs_names");
        if (val_ptr != nullptr) {
          auto outputs_names = GetValue<std::vector<std::string>>(val_ptr);
          if (elements.size() == 1) {
            key = std::make_pair(cnode, 0);
            node_id_map_[key] = meta_graphT->allTensors.size();
            ms_tensor->name = outputs_names[0];
          } else {
            node_id_map_[key] = meta_graphT->allTensors.size();
            ms_tensor->name = outputs_names[i];
          }
        } else {
          if (elements.size() == 1) {
            key = std::make_pair(cnode, 0);
            node_id_map_[key] = meta_graphT->allTensors.size();
            ms_tensor->name = cnode_name;
          } else {
            node_id_map_[key] = meta_graphT->allTensors.size();
            ms_tensor->name = cnode_name + "_o:" + std::to_string(i);
          }
        }

        if (!utils::isa<abstract::AbstractTensorPtr>(elements[i])) {
          MS_LOG(ERROR) << "abstract is not AbstractTensor";
          delete (ms_tensor);
          return RET_ERROR;
        }
        auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(elements[i]);
        MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, RET_ERROR, "Cast to abstract tensor failed!");
        auto type_ptr = abstract_tensor->element()->GetTypeTrack();
        MS_CHECK_TRUE_MSG(type_ptr != nullptr, RET_ERROR, "type_ptr is nullptr");
        ms_tensor->dataType = type_ptr->type_id();
        meta_graphT->allTensors.emplace_back(ms_tensor);
        if (opt::CheckPrimitiveType(cnode, prim::kPrimConv2DFusion) ||
            opt::CheckPrimitiveType(cnode, prim::kPrimFusedBatchNorm) ||
            opt::CheckPrimitiveType(cnode, prim::kPrimLayerNormFusion)) {
          break;
        }
      }
    }
  } else {
    auto ms_tensor = new (std::nothrow) schema::TensorT();
    if (ms_tensor == nullptr) {
      MS_LOG(ERROR) << "new tensor failed";
      return RET_ERROR;
    }
    auto type = kNumberTypeFloat32;
    if (utils::isa<abstract::AbstractTensorPtr>(cnode->abstract())) {
      auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(cnode->abstract());
      MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, RET_ERROR, "Cast to abstract tensor failed!");
      auto typePtr = abstract_tensor->element()->GetTypeTrack();
      type = typePtr->type_id();
    }
    ms_tensor->dataType = type;
    ms_tensor->nodeType = NodeType_CNode;
    auto val_ptr = cnode->GetAttr("outputs_names");
    if (val_ptr != nullptr) {
      auto outputs_names = GetValue<std::vector<std::string>>(val_ptr);
      ms_tensor->name = outputs_names[0];
    } else {
      ms_tensor->name = cnode_name;
    }
    fb_node->outputIndex.emplace_back(meta_graphT->allTensors.size());

    auto key = std::make_pair(cnode, 0);
    node_id_map_[key] = meta_graphT->allTensors.size();
    meta_graphT->allTensors.emplace_back(ms_tensor);
  }
  return RET_OK;
}

CNodePtr AnfExporter::CreateCallCnode(const FuncGraphPtr &fg, const AnfNodePtr &node) {
  auto call_anf_prim_vnode = GetCallAnfPrim();
  MS_CHECK_TRUE_MSG(call_anf_prim_vnode != nullptr, nullptr, "GetCallAnfPrim failed");
  std::vector<AnfNodePtr> inputs{call_anf_prim_vnode, node};
  auto cnode = fg->NewCNodeInOrder(inputs);
  MS_CHECK_TRUE_MSG(cnode != nullptr, nullptr, "NewCNode failed");
  cnode->set_func_graph(fg);
  return cnode;
}

CNodePtr AnfExporter::CreatePartialCnode(const FuncGraphPtr &fg, const AnfNodePtr &node) {
  if (utils::isa<CNodePtr>(node)) {
    auto cnode = utils::cast<CNodePtr>(node);
    MS_CHECK_TRUE_MSG(cnode != nullptr, nullptr, "cast ptr failed");
    auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(kPrimIndex));
    if (primitive_c != nullptr) {
      return cnode;
    }
    auto partial_anf_prim_vnode = GetPartialFusionPrim();
    auto cnode_input = cnode->inputs();
    MS_CHECK_TRUE_MSG(partial_anf_prim_vnode != nullptr, nullptr, "GetPartialFusionPrim failed");
    cnode_input.insert(cnode_input.begin(), partial_anf_prim_vnode);
    cnode->set_inputs(cnode_input);
    return cnode;
  } else if (utils::isa<ValueNodePtr>(node)) {
    auto partial_anf_prim_vnode = GetPartialFusionPrim();
    MS_CHECK_TRUE_MSG(partial_anf_prim_vnode != nullptr, nullptr, "GetPartialFusionPrim failed");
    std::vector<AnfNodePtr> inputs{partial_anf_prim_vnode, node};
    auto cnode = fg->NewCNode(inputs);
    MS_CHECK_TRUE_MSG(cnode != nullptr, nullptr, "New cnode failed");
    return cnode;
  } else {
    MS_LOG(ERROR) << "failed to create partial cnode.";
    return nullptr;
  }
}

schema::MetaGraphT *Export(const FuncGraphPtr &func_graph, bool keep_graph, bool copy_primitive, bool train_flag) {
  AnfExporter lite_exporter;
  return lite_exporter.Export(func_graph, keep_graph, copy_primitive, train_flag);
}
}  // namespace mindspore::lite
