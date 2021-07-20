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
#include <vector>
#include <regex>
#include "tools/converter/parser/parser_utils.h"
#include "tools/converter/import/primitive_adjust.h"
#include "tools/converter/import/mindir_adjust.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/parser/insert_transpose.h"

namespace mindspore::lite {
namespace {
constexpr size_t kConvWeightIndex = 2;
}  // namespace
STATUS MindsporeImporter::Mindir2AnfAdjust(const FuncGraphPtr &func_graph, const converter::Flags &flag) {
  auto primitive_adjust_pass = std::make_shared<PrimitiveAdjust>();
  primitive_adjust_pass->SetFmkType(flag.fmk);
  if (!primitive_adjust_pass->Run(func_graph)) {
    MS_LOG(ERROR) << "primitive adjust failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return RET_ERROR;
  }
  auto mindir_adjust_pass = std::make_shared<MindirAdjust>();
  mindir_adjust_pass->SetFmkType(flag.fmk);
  mindir_adjust_pass->SetQuantType(flag.quantType);
  mindir_adjust_pass->SetTrainFlag(flag.trainModel);
  if (!mindir_adjust_pass->Run(func_graph)) {
    MS_LOG(ERROR) << "mindir adjust failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS MindsporeImporter::WeightFormatTransform(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  auto node_list = TopoSort(graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto conv_cnode = node->cast<CNodePtr>();
    if (!opt::CheckPrimitiveType(node, prim::kPrimConv2DFusion) &&
        !opt::CheckPrimitiveType(node, opt::kPrimConv2DBackpropInputFusion) &&
        !opt::CheckPrimitiveType(node, prim::kPrimConv2dTransposeFusion)) {
      continue;
    }
    MS_ASSERT(conv_cnode->inputs().size() > kConvWeightIndex);
    int status = HardCodeMindir(conv_cnode, graph);
    if (status != lite::RET_OK) {
      MS_LOG(ERROR) << "Format hard code failed: " << status << ", node: " << node->fullname_with_scope();
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS MindsporeImporter::HardCodeMindir(const CNodePtr &conv_node, const FuncGraphPtr &graph) {
  MS_ASSERT(conv_cnode != nullptr);
  auto prim = GetValueNode<PrimitivePtr>(conv_node->input(0));
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Invalid anfnode, which don't have primitive.";
    return lite::RET_ERROR;
  }
  int64_t format = prim->GetAttr(ops::kFormat) != nullptr ? GetValue<int64_t>(prim->GetAttr(ops::kFormat)) : 0;
  auto weight_node = conv_node->input(kConvWeightIndex);
  schema::Format weight_dst_format = schema::Format::Format_KHWC;
  STATUS status = RET_OK;
  schema::Format weight_src_format = schema::Format::Format_NUM_OF_FORMAT;
  switch (quant_type_) {
    case QuantType_AwareTraining:
    case QuantType_PostTraining:
    case QuantType_WeightQuant:
    case QuantType_QUANT_NONE: {
      if (format == schema::Format::Format_KHWC) {
        weight_src_format = schema::Format::Format_KHWC;
      } else {
        weight_src_format = schema::Format::Format_KCHW;
      }
    } break;
    default: {
      MS_LOG(ERROR) << "Unsupported quantType: " << EnumNameQuantType(quant_type_)
                    << ", node: " << conv_node->fullname_with_scope();
      return RET_ERROR;
    }
  }
  if (utils::isa<CNodePtr>(weight_node)) {
    status = HandleWeightConst(graph, conv_node, weight_node->cast<CNodePtr>(), weight_src_format, weight_dst_format);
    if (status != lite::RET_OK) {
      MS_LOG(ERROR) << "handle weight-const failed.";
      return RET_ERROR;
    }
  }
  weight_node = conv_node->input(kConvWeightIndex);
  auto weight_value = opt::GetTensorInfo(weight_node);
  if (weight_value != nullptr) {
    status = opt::TransFilterFormat(weight_value, weight_src_format, weight_dst_format);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "TransFilter " << EnumNameFormat(schema::EnumValuesFormat()[weight_dst_format]) << "To"
                    << EnumNameFormat(weight_dst_format) << " failed, node : " << conv_node->fullname_with_scope()
                    << "quant type:" << quant_type_;
      return RET_ERROR;
    }
    prim->AddAttr(ops::kFormat, MakeValue<int64_t>(weight_dst_format));
    auto type_id = static_cast<TypeId>(weight_value->data_type());
    auto shape = weight_value->shape();
    std::vector<int64_t> shape_vector(shape.begin(), shape.end());
    auto abstract = lite::CreateTensorAbstract(shape_vector, type_id);
    if (abstract == nullptr) {
      MS_LOG(ERROR) << "Create tensor abstarct failed";
      return RET_ERROR;
    }
    weight_node->set_abstract(abstract);
  }
  if (utils::isa<ParameterPtr>(weight_node)) {
    status = HandleWeightSharing(graph, KHWC, weight_node->cast<ParameterPtr>(), weight_src_format, weight_dst_format);
    if (status != lite::RET_OK) {
      MS_LOG(ERROR) << "handle weight-sharing failed.";
      return RET_ERROR;
    }
  }
  return lite::RET_OK;
}

size_t MindsporeImporter::Hex2ByteArray(std::string hex_str, unsigned char *byte_array, size_t max_len) {
  std::regex r("[0-9a-fA-F]+");
  if (!std::regex_match(hex_str, r)) {
    MS_LOG(ERROR) << "Some characters of dec_key not in [0-9a-fA-F]";
    return 0;
  }
  if (hex_str.size() % 2 == 1) {
    hex_str.push_back('0');
  }
  size_t byte_len = hex_str.size() / 2;
  if (byte_len > max_len) {
    MS_LOG(ERROR) << "the hexadecimal dec_key length exceeds the maximum limit: 64";
    return 0;
  }
  for (size_t i = 0; i < byte_len; ++i) {
    size_t p = i * 2;
    if (hex_str[p] >= 'a' && hex_str[p] <= 'f') {
      byte_array[i] = hex_str[p] - 'a' + 10;
    } else if (hex_str[p] >= 'A' && hex_str[p] <= 'F') {
      byte_array[i] = hex_str[p] - 'A' + 10;
    } else {
      byte_array[i] = hex_str[p] - '0';
    }
    if (hex_str[p + 1] >= 'a' && hex_str[p + 1] <= 'f') {
      byte_array[i] = (byte_array[i] << 4) | (hex_str[p + 1] - 'a' + 10);
    } else if (hex_str[p] >= 'A' && hex_str[p] <= 'F') {
      byte_array[i] = (byte_array[i] << 4) | (hex_str[p + 1] - 'A' + 10);
    } else {
      byte_array[i] = (byte_array[i] << 4) | (hex_str[p + 1] - '0');
    }
  }
  return byte_len;
}

FuncGraphPtr MindsporeImporter::ImportMindIR(const converter::Flags &flag) {
  quant_type_ = flag.quantType;
  FuncGraphPtr func_graph;
  if (flag.dec_key.size() != 0) {
    unsigned char key[32];
    const size_t key_len = Hex2ByteArray(flag.dec_key, key, 32);
    if (key_len == 0) {
      return nullptr;
    }
    func_graph = LoadMindIR(flag.modelFile, false, key, key_len, flag.dec_mode);
  } else {
    func_graph = LoadMindIR(flag.modelFile);
  }
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "get funcGraph failed for fmk:MINDIR";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return nullptr;
  }
  func_graph->set_attr("graph_name", MakeValue("main_graph"));
  func_graph->set_attr("fmk", MakeValue(static_cast<int>(converter::FmkType_MS)));
  STATUS status;
  if ((status = Mindir2AnfAdjust(func_graph, flag)) != RET_OK) {
    MS_LOG(ERROR) << "Mindir2AnfAdjust failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  auto insert_transpose = std::make_shared<InsertTranspose>(lite::converter::FmkType_MS, flag.trainModel);
  if (!insert_transpose->Run(func_graph)) {
    MS_LOG(ERROR) << "Run insert transpose failed.";
    return nullptr;
  }
  if ((status = WeightFormatTransform(func_graph)) != RET_OK) {
    MS_LOG(ERROR) << "WeightFormatTransform failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  return func_graph;
}
}  // namespace mindspore::lite
