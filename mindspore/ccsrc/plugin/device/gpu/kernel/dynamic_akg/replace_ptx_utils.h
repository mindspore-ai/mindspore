/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <sstream>
#include <regex>
#include <string>
#include <vector>
#include <cstdio>
#include <tuple>

using std::string;
using std::stringstream;
using std::vector;

constexpr auto kShouldRemove = -9999999;
constexpr auto kShouldKeep = -10000000;

string GetParam(const string &line, const string &value) {
  // get param number 0 from '.param .u64 Fused_xxx_kernel_param_0'
  string patternStr = "\\s*\\.param\\s+\\.u64\\s+" + value + "_param_(\\d+)\\s*";
  std::regex pattern(patternStr);
  std::smatch match;
  if (std::regex_search(line, match, pattern)) {
    return match[1].str();
  } else {
    return "";
  }
}

std::tuple<string, string> GetRegFromLoadParamGlobal(const string &line, const string &value) {
  // ld.param.u64   %rd2, [Fused_Reshape_Cast_Neg_Mul_fusion_18315353371220478878_kernel_param_18];
  string patternStr = "\\s*ld\\.param\\.u64\\s+(\\%\\w+), \\[" + value + "_param_(\\w+)\\]\\s*;";
  std::regex pattern(patternStr);
  std::smatch match;
  if (std::regex_search(line, match, pattern)) {
    return std::make_tuple(match[1].str(), match[2].str());
  } else {
    return std::make_tuple("", "");
  }
}

bool ContainsInstruction(const string &line, const string &instruction) {
  return line.find(instruction) != string::npos;
}

vector<int64_t> ParamsToValues(const vector<vector<int64_t>> shape_lists) {
  vector<int64_t> value_list;
  for (size_t i = 0; i < shape_lists.size(); i++) {
    value_list.push_back(kShouldRemove);
    value_list.push_back(kShouldKeep);  // real pointer
    for (size_t j = 0; j < shape_lists[i].size(); j++) {
      value_list.push_back(shape_lists[i][j]);
    }
  }
  return value_list;
}

string ReplacePTXFunction(const string &original_ptx, const vector<vector<int64_t>> shape_list,
                          const string &kernel_name) {
  stringstream original_ptx_stream(original_ptx);
  stringstream replaced_ptx_stream;
  string line;
  int step = 0;
  size_t currentTensor = 0;
  vector<int64_t> value_list = ParamsToValues(shape_list);
  while (std::getline(original_ptx_stream, line)) {
    if (step == 0 && ContainsInstruction(line, ".entry")) {
      step = 1;
    } else if (step == 1) {
      if (ContainsInstruction(line, ".param")) {
        auto numStr = GetParam(line, kernel_name);
        auto num = std::stoi(numStr);
        if (value_list[num] == kShouldKeep) {
          currentTensor++;
          if (currentTensor == shape_list.size()) {
            size_t pos = line.find(",");
            if (pos != string::npos) {
              line.erase(pos, 1);
            }
          }
          replaced_ptx_stream << line << '\n';
        }
        continue;
      } else {
        step = 2;
      }
    } else if (step == 2) {
      if (ContainsInstruction(line, kernel_name)) {
        string reg, numStr;
        std::tie(reg, numStr) = GetRegFromLoadParamGlobal(line, kernel_name);
        if (reg != "" && numStr != "") {
          int num = std::stoi(numStr);
          if (value_list[num] != kShouldKeep) {
            replaced_ptx_stream << "\tmov.u64 " << reg << ", " << std::to_string(value_list[num]) << ";\n";
            continue;
          }
        }
      }
    }
    replaced_ptx_stream << line << '\n';
  }
  return replaced_ptx_stream.str();
}
