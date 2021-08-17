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
#include <set>
#include <vector>
#include <regex>
#include "tools/converter/parser/parser_utils.h"
#include "tools/converter/import/primitive_adjust.h"
#include "tools/converter/import/mindir_adjust.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/parser/unify_format.h"
#include "tools/converter/parser/lstm_adjust_pass.h"

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
  STATUS status;
  if ((status = Mindir2AnfAdjust(func_graph, flag)) != RET_OK) {
    MS_LOG(ERROR) << "Mindir2AnfAdjust failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  auto unify_format = std::make_shared<UnifyFormatToNHWC>(converter::kFmkTypeMs, flag.trainModel, flag.quantType);
  if (!unify_format->Run(func_graph)) {
    MS_LOG(ERROR) << "Run insert transpose failed.";
    return nullptr;
  }

  auto lstm_adjust_pass = std::make_shared<opt::LstmAdjustPass>();
  if (!lstm_adjust_pass->Run(func_graph)) {
    MS_LOG(ERROR) << "Run mindir lstm adjust failed.";
    return nullptr;
  }
  return func_graph;
}
}  // namespace mindspore::lite
