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
#include "tools/converter/parser/parser_utils.h"
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

STATUS MindsporeImporter::RemoveUnusedGraphInput(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, RET_ERROR, "func_graph is nullptr");
  std::map<AnfNodePtr, bool> graph_input_map;
  for (auto &input : func_graph->get_inputs()) {
    graph_input_map[input] = false;
  }
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    for (size_t i = 0; i < cnode->inputs().size(); i++) {
      for (auto &input : func_graph->get_inputs()) {
        if (input == cnode->input(i) && graph_input_map.count(input) == 1) {
          graph_input_map[input] = true;
        }
      }
    }
  }
  for (auto &item : graph_input_map) {
    if (item.second == false) {
      func_graph->DropNode(item.first);
    }
  }
  return RET_OK;
}

FuncGraphPtr MindsporeImporter::ImportMindIR(const converter::Flags &flag) {
  FuncGraphPtr func_graph;
  if (flag.dec_key.size() != 0) {
    unsigned char key[32];
    const size_t key_len = Hex2ByteArray(flag.dec_key, key, 32);
    if (key_len == 0) {
      return nullptr;
    }
    func_graph = LoadMindIR(flag.modelFile, false, key, key_len, flag.dec_mode);
    auto ret = memset_s(key, sizeof(key), 0, key_len);
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "memset_s error";
    }
  } else {
    func_graph = LoadMindIR(flag.modelFile);
  }
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "get funcGraph failed for fmk:MINDIR";
    MS_LOG(ERROR)
      << "The model maybe an old model, Please download the package whose version is before 1.2 and then try again.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return nullptr;
  }
  func_graph->set_attr("graph_name", MakeValue("main_graph"));
  func_graph->set_attr("fmk", MakeValue(static_cast<int>(converter::kFmkTypeMs)));
  auto status = RemoveUnusedGraphInput(func_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "RemoveUnusedGraphInput failed.";
    return nullptr;
  }
  status = GetFuncGraphOutputName(func_graph->get_return());
  if (status != RET_OK) {
    MS_LOG(ERROR) << "GetFuncGraphOutputName failed.";
    return nullptr;
  }
  if (output_tensor_name_.empty()) {
    MS_LOG(ERROR) << "Can not find output name.";
    return nullptr;
  }
  ConverterContext::GetInstance()->SetGraphOutputTensorNames(output_tensor_name_);
#ifdef ENABLE_LITE_ACL
  MS_LOG(INFO) << "There is no need to adjust and pass graph when in Ascend310.";
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

  auto lstm_adjust_pass = std::make_shared<opt::LstmAdjustPass>();
  MS_CHECK_TRUE_MSG(lstm_adjust_pass != nullptr, nullptr, "lstm_adjust_pass is nullptr.");
  if (!lstm_adjust_pass->Run(func_graph)) {
    MS_LOG(ERROR) << "Run mindir lstm adjust failed.";
    return nullptr;
  }
  return func_graph;
}
}  // namespace mindspore::lite
