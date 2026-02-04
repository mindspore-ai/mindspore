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
#ifndef MINDSPORE_CCSRC_TOOLS_SUMMARY_SUMMARY_H_
#define MINDSPORE_CCSRC_TOOLS_SUMMARY_SUMMARY_H_

#include <map>
#include <string>
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "ir/tensor.h"
#include "tools/visible.h"

namespace mindspore::tools {
using CallBackFunc = uint32_t (*)(uint32_t graph_id,
                                  const std::map<std::string, mindspore::tensor::TensorPtr> &params_list);
using mindspore::session::KernelGraph;

class TOOLS_EXPORT Summary {
 public:
  static Summary &GetInstance();
  void RecurseSetSummaryNodesForAllGraphs(KernelGraph *graph);
  void SummaryTensor(KernelGraph *graph);
  void RegisterSummaryCallBackFunc();
  void SetSummaryNodes(KernelGraph *graph);

 private:
  CallBackFunc summary_callback_;
};
void RecurseSetSummaryNodesForAllGraphs(KernelGraph *graph);
void SummaryTensor(KernelGraph *graph);
void RegisterSummaryCallBackFunc();
}  // namespace mindspore::tools
#endif  // MINDSPORE_CCSRC_TOOLS_SUMMARY_SUMMARY_H_
