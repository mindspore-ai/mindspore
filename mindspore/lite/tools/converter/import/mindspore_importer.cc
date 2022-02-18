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

#include "tools/converter/import/mindspore_importer.h"
#include <memory>
#include <map>
#include <set>
#include <vector>
#include <regex>
#include <queue>
#include "tools/converter/parser/parser_utils.h"
#include "tools/converter/import/cast_op_adjust.h"
#include "tools/converter/import/primitive_adjust.h"
#include "tools/converter/import/mindir_adjust.h"
#include "tools/converter/import/mindir_control_flow_adjust.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/parser/unify_format.h"
#include "tools/converter/parser/lstm_adjust_pass.h"
#include "nnacl/op_base.h"

namespace mindspore::lite {
namespace {
constexpr size_t kConvWeightIndex = 2;
}  // namespace
STATUS MindsporeImporter::Mindir2AnfAdjust(const FuncGraphPtr &func_graph, const converter::Flags &flag) {
  MS_ASSERT(func_graph != nullptr);
  auto primitive_adjust_pass = std::make_shared<PrimitiveAdjust>();
  MS_CHECK_TRUE_MSG(primitive_adjust_pass != nullptr, RET_NULL_PTR, "primitive_adjust_pass is nullptr.");
  primitive_adjust_pass->SetFmkType(flag.fmk);
  if (!primitive_adjust_pass->Run(func_graph)) {
    MS_LOG(ERROR) << "primitive adjust failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return RET_ERROR;
  }
  auto mindir_adjust_pass = std::make_shared<MindirAdjust>();
  MS_CHECK_TRUE_MSG(mindir_adjust_pass != nullptr, RET_NULL_PTR, "mindir_adjust_pass is nullptr.");
  mindir_adjust_pass->SetFmkType(flag.fmk);
  mindir_adjust_pass->SetTrainFlag(flag.trainModel);
  if (!mindir_adjust_pass->Run(func_graph)) {
    MS_LOG(ERROR) << "MindIr adjust failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return RET_ERROR;
  }
  if (!flag.trainModel) {
    auto cast_op_adjust = std::make_shared<CastOpAdjust>();
    MS_CHECK_TRUE_MSG(cast_op_adjust != nullptr, RET_NULL_PTR, "cast_op_adjust is nullptr.");
    if (!cast_op_adjust->Run(func_graph)) {
      MS_LOG(ERROR) << "MindIr adjust cast operator failed.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
      return RET_ERROR;
    }
  }
  auto mindir_control_flow_adjust = std::make_shared<MindIRControlFlowAdjust>();
  MS_CHECK_TRUE_MSG(mindir_control_flow_adjust != nullptr, RET_NULL_PTR, "mindir_control_flow_adjust is nullptr.");
  mindir_control_flow_adjust->SetFmkType(flag.fmk);
  if (!mindir_control_flow_adjust->Run(func_graph)) {
    MS_LOG(ERROR) << "MindIR control flow adjust failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return RET_ERROR;
  }
  return RET_OK;
}

size_t MindsporeImporter::Hex2ByteArray(const std::string &hex_str, unsigned char *byte_array, size_t max_len) {
  std::regex r("[0-9a-fA-F]+");
  if (!std::regex_match(hex_str, r)) {
    MS_LOG(ERROR) << "Some characters of dec_key not in [0-9a-fA-F]";
    return 0;
  }
  if (hex_str.size() % 2 == 1) {  // Mod 2 determines whether it is odd
    MS_LOG(ERROR) << "the hexadecimal dec_key length must be even";
    return 0;
  }
  size_t byte_len = hex_str.size() / 2;  // Two hexadecimal characters represent a byte
  if (byte_len > max_len) {
    MS_LOG(ERROR) << "the hexadecimal dec_key length exceeds the maximum limit: 64";
    return 0;
  }
  constexpr int32_t a_val = 10;  // The value of 'A' in hexadecimal is 10
  constexpr size_t half_byte_offset = 4;
  for (size_t i = 0; i < byte_len; ++i) {
    size_t p = i * 2;  // The i-th byte is represented by the 2*i and 2*i+1 hexadecimal characters
    if (hex_str[p] >= 'a' && hex_str[p] <= 'f') {
      byte_array[i] = hex_str[p] - 'a' + a_val;
    } else if (hex_str[p] >= 'A' && hex_str[p] <= 'F') {
      byte_array[i] = hex_str[p] - 'A' + a_val;
    } else {
      byte_array[i] = hex_str[p] - '0';
    }
    if (hex_str[p + 1] >= 'a' && hex_str[p + 1] <= 'f') {
      byte_array[i] = (byte_array[i] << half_byte_offset) | (hex_str[p + 1] - 'a' + a_val);
    } else if (hex_str[p] >= 'A' && hex_str[p] <= 'F') {
      byte_array[i] = (byte_array[i] << half_byte_offset) | (hex_str[p + 1] - 'A' + a_val);
    } else {
      byte_array[i] = (byte_array[i] << half_byte_offset) | (hex_str[p + 1] - '0');
    }
  }
  return byte_len;
}

STATUS MindsporeImporter::ProcessDependCnode(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  if (!opt::CheckPrimitiveType(cnode, prim::kPrimDepend)) {
    output_tensor_name_.push_back(cnode->fullname_with_scope());
    return RET_NO_CHANGE;
  }
  auto depend_input = cnode->input(1);
  MS_CHECK_TRUE_MSG(depend_input != nullptr, RET_ERROR, "depend_input is nullptr");
  if (utils::isa<CNodePtr>(depend_input)) {
    auto depend_input_cnode = utils::cast<CNodePtr>(depend_input);
    auto status = ProcessDependCnode(depend_input_cnode);
    if (status == RET_NO_CHANGE) {
      return RET_OK;
    }
  } else if (utils::isa<ParameterPtr>(depend_input) || utils::isa<ValueNode>(depend_input)) {
    output_tensor_name_.push_back(depend_input->fullname_with_scope());
  }
  return RET_OK;
}

STATUS MindsporeImporter::GetFuncGraphOutputName(const CNodePtr &return_node) {
  MS_ASSERT(return_node != nullptr);
  for (size_t i = 0; i < return_node->inputs().size(); i++) {
    auto output_node = return_node->input(i);
    if (output_node == nullptr) {
      MS_LOG(ERROR) << "output_node is nullptr.";
      return RET_ERROR;
    } else if (output_node->isa<mindspore::CNode>()) {
      if (opt::CheckPrimitiveType(output_node, prim::kPrimUpdateState) ||
          opt::CheckPrimitiveType(output_node, prim::kPrimLoad)) {
        continue;
      }
      auto output_cnode = utils::cast<CNodePtr>(output_node);
      if (opt::CheckPrimitiveType(output_node, prim::kPrimMakeTuple)) {
        for (size_t j = 0; j < output_cnode->inputs().size(); j++) {
          auto tuple_input = output_cnode->input(j);
          MS_CHECK_TRUE_MSG(tuple_input != nullptr, RET_ERROR, "tuple_input is nullptr");
          if (!utils::isa<CNodePtr>(tuple_input)) {
            continue;
          }
          auto tuple_input_cnode = utils::cast<CNodePtr>(tuple_input);
          if (opt::CheckPrimitiveType(output_node, prim::kPrimUpdateState) ||
              opt::CheckPrimitiveType(output_node, prim::kPrimLoad)) {
            continue;
          }
          auto status = ProcessDependCnode(tuple_input_cnode);
          if (status != RET_OK && status != RET_NO_CHANGE) {
            MS_LOG(ERROR) << "ProcessDependCnode failed.";
          }
        }
      } else if (opt::CheckPrimitiveType(output_node, prim::kPrimDepend)) {
        auto status = ProcessDependCnode(output_cnode);
        if (status != RET_OK && status != RET_NO_CHANGE) {
          MS_LOG(ERROR) << "ProcessDependCnode failed.";
        }
      } else {
        output_tensor_name_.push_back(output_cnode->fullname_with_scope());
      }
    }
  }
  return RET_OK;
}

namespace {
bool IsEmptyOp(const AnfNodePtr &node) {
  MS_ASSERT(node != nullptr);
  return (opt::CheckPrimitiveType(node, prim::kPrimMakeTuple) || opt::CheckPrimitiveType(node, prim::kPrimReturn) ||
          opt::CheckPrimitiveType(node, prim::kPrimTupleGetItem) || opt::CheckPrimitiveType(node, prim::kPrimDepend) ||
          opt::CheckPrimitiveType(node, prim::kPrimUpdateState) || opt::CheckPrimitiveType(node, prim::kPrimLoad));
}

void RemovePostEdgeOfParameter(const AnfNodePtr &parameter) {
  MS_ASSERT(parameter != nullptr);
  auto func_graph = parameter->func_graph();
  MS_ASSERT(func_graph != nullptr);
  auto manager = Manage(func_graph);
  MS_ASSERT(maneger != nullptr);
  auto nodes_users = manager->node_users();
  auto node_users_iter = nodes_users.find(parameter);
  MS_ASSERT(node_users_iter != nodes_users.end());
  for (const auto &node_user_iter : node_users_iter->second) {
    MS_ASSERT(utils::isa<CNodePtr>(node_user_iter.first));
    auto node_user_cnode = utils::cast<CNodePtr>(node_user_iter.first);
    auto &node_user_cnode_inputs = node_user_cnode->inputs();
    std::vector<AnfNodePtr> new_node_user_cnode_inputs;
    for (size_t i = 0; i < node_user_cnode_inputs.size(); i++) {
      if (static_cast<int>(i) == node_user_iter.second) {
        continue;
      }
      new_node_user_cnode_inputs.emplace_back(node_user_cnode_inputs.at(i));
    }
    node_user_cnode->set_inputs(new_node_user_cnode_inputs);
  }
}
}  // namespace

void MindsporeImporter::RemoveUnusedGraphInput(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  // drop unused input_parameter and disconnect edge
  auto graph_inputs = func_graph->get_inputs();
  auto manager = Manage(func_graph);
  MS_ASSERT(manager != nullptr);
  auto nodes_users = manager->node_users();
  std::vector<AnfNodePtr> unused_inputs;
  for (const auto &input : graph_inputs) {
    bool found_used = false;
    std::queue<AnfNodePtr> q;
    q.push(input);
    while (!q.empty()) {
      auto cur_node = q.front();
      q.pop();
      if (cur_node != input && !IsEmptyOp(cur_node)) {
        found_used = true;
        break;
      }
      auto node_users_itr = nodes_users.find(cur_node);
      if (node_users_itr == nodes_users.end()) {
        continue;
      }
      for (const auto &node_user_itr : node_users_itr->second) {
        MS_ASSERT(utils::isa<CNodePtr>(node_user_itr.first));
        auto node_user_cnode = utils::cast<CNodePtr>(node_user_itr.first);
        q.push(node_user_cnode);
      }
    }
    if (!found_used) {
      if (nodes_users.find(input) != nodes_users.end()) {
        RemovePostEdgeOfParameter(input);
      }
      unused_inputs.push_back(input);
    }
  }
  for (auto &input : unused_inputs) {
    func_graph->DropNode(input);
  }
}

FuncGraphPtr MindsporeImporter::ImportMindIR(const converter::Flags &flag, const void *buff, const size_t &size) {
  MindIRLoader mindir_loader;
  auto func_graph = mindir_loader.LoadMindIR(buff, size);
  return CheckAndUpdateFuncGraph(flag, func_graph);
}

FuncGraphPtr MindsporeImporter::ImportMindIR(const converter::Flags &flag) {
  FuncGraphPtr func_graph;
  if (!flag.dec_key.empty()) {
    unsigned char key[32];
    const size_t key_len = Hex2ByteArray(flag.dec_key, key, 32);
    if (key_len == 0) {
      return nullptr;
    }
    MindIRLoader mindir_loader(false, key, key_len, flag.dec_mode, false);
    func_graph = mindir_loader.LoadMindIR(flag.modelFile);
    auto ret = memset_s(key, sizeof(key), 0, key_len);
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "memset_s error";
    }
  } else {
    MindIRLoader mindir_loader;
    func_graph = mindir_loader.LoadMindIR(flag.modelFile);
  }

  return CheckAndUpdateFuncGraph(flag, func_graph);
}

FuncGraphPtr MindsporeImporter::CheckAndUpdateFuncGraph(const converter::Flags &flag, FuncGraphPtr func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "get funcGraph failed for fmk:MINDIR";
    MS_LOG(ERROR)
      << "The model maybe an old model, Please download the package whose version is before 1.2 and then try again.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return nullptr;
  }

  if (ConverterInnerContext::GetInstance()->GetGraphInputTensorShapeMapSize() > 0) {
    for (const auto &input : func_graph->get_inputs()) {
      MS_ASSERT(input->isa<Parameter>());
      auto name = input->cast<ParameterPtr>()->name();
      std::vector<int64_t> shape = ConverterInnerContext::GetInstance()->GetGraphInputTensorShape(name);
      if (shape.empty()) {
        MS_LOG(WARNING) << "Can not find name in map. name is " << name;
      } else {
        input->abstract()->set_shape(std::make_shared<mindspore::abstract::Shape>(shape));
      }
    }
  }

  func_graph->set_attr("graph_name", MakeValue("main_graph"));
  func_graph->set_attr("fmk", MakeValue(static_cast<int>(converter::kFmkTypeMs)));
  RemoveUnusedGraphInput(func_graph);
  if (CommonAnfAdjust(func_graph) != RET_OK) {
    MS_LOG(ERROR) << "AdjustForAnf failed.";
    return nullptr;
  }

  auto status = GetFuncGraphOutputName(func_graph->get_return());
  if (status != RET_OK) {
    MS_LOG(ERROR) << "GetFuncGraphOutputName failed.";
    return nullptr;
  }
  if (output_tensor_name_.empty()) {
    MS_LOG(ERROR) << "Can not find output name.";
    return nullptr;
  }
  ConverterInnerContext::GetInstance()->SetGraphOutputTensorNames(output_tensor_name_);
#ifdef ENABLE_LITE_ACL
  MS_LOG(INFO) << "There is no need to adjust and pass graph when in Ascend.";
  return func_graph;
#endif
  if ((status = Mindir2AnfAdjust(func_graph, flag)) != RET_OK) {
    MS_LOG(ERROR) << "Mindir2AnfAdjust failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  auto unify_format = std::make_shared<UnifyFormatToNHWC>(converter::kFmkTypeMs, flag.trainModel);
  MS_CHECK_TRUE_MSG(unify_format != nullptr, nullptr, "unify_format is nullptr.");
  if (!unify_format->Run(func_graph)) {
    MS_LOG(ERROR) << "Run insert transpose failed.";
    return nullptr;
  }

  return func_graph;
}
}  // namespace mindspore::lite
