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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_COMMON_FORMAT_UTILS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_COMMON_FORMAT_UTILS_H_

#include <vector>
#include <string>
#include <unordered_map>
#include "tools/optimizer/common/gllo_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace opt {
constexpr auto kOutputsFormat = "outputs_format";
enum FormatTransNodeType { kNCHW2NHWC, kNHWC2NCHW, kNONE };
struct TransTypePair {
  FormatTransNodeType pre_;
  FormatTransNodeType post_;
  TransTypePair() : pre_(kNONE), post_(kNONE) {}
};
const std::unordered_map<std::string, std::vector<size_t>> &GetNHWCOpMap();
const std::unordered_map<std::string, std::vector<size_t>> &GetNCHWOpMap();
const std::unordered_map<std::string, std::vector<size_t>> &GetToNCHWOpMap();
const std::vector<std::string> &GetDynamicFormatOpList();
bool IsDynamicFormatOp(const std::string &op_type);
bool IsDynamicFormatOpWithAxis(const std::string &op_type);
STATUS GetCastDstDataType(const CNodePtr &cnode, int *perm);
STATUS GetTransposePerm(const CNodePtr &cnode, std::vector<int> *perm);
void RemoveIfMonad(const CNodePtr &cnode);
bool IsMonadNode(const AnfNodePtr &node);
bool IsSpecialType(const CNodePtr &cnode);
int DetermineCertainOutputFormat(const CNodePtr &cnode, int index, Format *format);
int DetermineCertainVarInputFormat(const CNodePtr &cnode, size_t index, Format *format);
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_COMMON_FORMAT_UTILS_H_
