/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "tools/graph_kernel/converter/format_recognition.h"

#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include "utils/anf_utils.h"
#include "include/common/utils/utils.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"

namespace mindspore::graphkernel {
namespace {
using KernelWithIndex = std::pair<AnfNodePtr, size_t>;

std::pair<std::string, bool> GetLiteFormat(const CNodePtr &cnode) {
  auto prim = GetCNodePrimitive(cnode);
  if (prim == nullptr || !prim->HasAttr("format")) {
    return std::make_pair(kOpFormat_NCHW, false);
  }
  auto format_attr = prim->GetAttr("format");
  MS_EXCEPTION_IF_NULL(format_attr);
  auto format = GetValue<int64_t>(format_attr);
  if (format == 1) {
    return std::make_pair(kOpFormat_NHWC, false);
  } else {
    return std::make_pair(kOpFormat_NCHW, false);
  }
}

std::pair<std::string, bool> GetTransposeFormat(const CNodePtr &cnode) {
  constexpr size_t perm_idx = 2;
  std::vector<int64_t> perm_list;
  if (cnode->input(perm_idx)->isa<Parameter>()) {
    auto perm_para = cnode->input(perm_idx)->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(perm_para);
    auto perm_tensor = perm_para->default_param()->cast<tensor::TensorPtr>();
    auto perm = static_cast<int32_t *>(perm_tensor->data_ptr()->data());
    std::transform(perm, perm + perm_tensor->shape()[0], std::back_inserter(perm_list), IntToLong);
  } else {
    auto perm_value = cnode->input(perm_idx)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(perm_value);
    perm_list = GetValue<std::vector<int64_t>>(perm_value->value());
  }
  std::vector<int64_t> nh2nc_perm = {0, 3, 1, 2};
  std::vector<int64_t> nc2nh_perm = {0, 2, 3, 1};
  if (perm_list == nh2nc_perm) {
    return std::make_pair(kOpFormat_NCHW, true);
  } else if (perm_list == nc2nh_perm) {
    return std::make_pair(kOpFormat_NHWC, true);
  } else {
    return GetLiteFormat(cnode);
  }
}

std::pair<std::string, bool> ExtractOutputFormat(const CNodePtr &cnode) {
  if (IsPrimitiveCNode(cnode, prim::kPrimTranspose)) {
    return GetTransposeFormat(cnode);
  }
  return GetLiteFormat(cnode);
}

void SetOutputsFormat(const CNodePtr &cnode) {
  auto extract_res = ExtractOutputFormat(cnode);
  auto output_format = extract_res.first;
  if (!extract_res.second) {
    // fix output format when it can not be determined by transpose
    for (size_t i = 1; i < cnode->size(); i++) {
      if (cnode->input(i)->isa<CNode>()) {
        auto kernel_with_index = AnfUtils::VisitKernel(cnode->input(i), 0);
        auto prev_cnode = kernel_with_index.first->cast<CNodePtr>();
        if (prev_cnode != nullptr && prev_cnode->HasAttr(kOutputsFormat)) {
          output_format =
            GetValue<std::vector<std::string>>(prev_cnode->GetAttr(kOutputsFormat))[kernel_with_index.second];
          break;
        }
      }
    }
  }
  if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple)) {
    std::vector<std::string> outputs_format(cnode->size() - 1, output_format);
    cnode->AddAttr(kOutputsFormat, MakeValue(outputs_format));
  } else {
    std::vector<std::string> outputs_format(AnfUtils::GetOutputTensorNum(cnode), output_format);
    cnode->AddAttr(kOutputsFormat, MakeValue(outputs_format));
  }
}

void FixFormatsBeforeTranspose(const CNodePtr &cnode) {
  for (size_t i = 1; i < cnode->size(); i++) {
    auto prev_node = cnode->input(i);
    if (IsPrimitiveCNode(prev_node, prim::kPrimTranspose)) {
      continue;
    }
    auto outputs_format = GetValue<std::vector<std::string>>(cnode->GetAttr(kOutputsFormat));
    if (outputs_format.empty()) {
      MS_LOG(EXCEPTION) << "The outputs_format of node " << cnode->fullname_with_scope() << " is empty.";
    }
    if (prev_node->isa<CNode>()) {
      auto prev_cnode = prev_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(prev_cnode);
      auto prev_format = outputs_format[0];
      if (ExtractOutputFormat(cnode).second) {
        // input node need to fix format when current node is nhwc->nchw or nchw->nhwc
        if (prev_format == kOpFormat_NCHW) {
          prev_format = kOpFormat_NHWC;
        } else {
          prev_format = kOpFormat_NCHW;
        }
      }
      std::vector<std::string> outputs_formats(AnfUtils::GetOutputTensorNum(prev_cnode), prev_format);
      prev_cnode->AddAttr(kOutputsFormat, MakeValue(outputs_formats));
    } else if (prev_node->isa<Parameter>()) {
      // save parameter's format in callback instance
      inner::NodeBase nodebase = {{}, TypeId::kMetaTypeBegin, outputs_format[0]};
      Callback::Instance()->SetBasicNodeKernelInfo(prev_node, {nodebase});
    }
  }
}
}  // namespace

bool FormatRecognition::Run(const FuncGraphPtr &func_graph) {
  auto todos = TopoSort(func_graph->output());
  for (auto &node : todos) {
    if (node->isa<CNode>()) {
      SetOutputsFormat(node->cast<CNodePtr>());
    }
  }
  for (auto it = todos.rbegin(); it != todos.rend(); it++) {
    if ((*it)->isa<CNode>()) {
      FixFormatsBeforeTranspose((*it)->cast<CNodePtr>());
    }
  }
  return true;
}
}  // namespace mindspore::graphkernel
