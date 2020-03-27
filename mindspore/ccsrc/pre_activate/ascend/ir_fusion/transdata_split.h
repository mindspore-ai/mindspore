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
#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_IR_FUSION_TRANSDATA_SPLIT_H_
#define MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_IR_FUSION_TRANSDATA_SPLIT_H_
#include <vector>
#include <string>
#include <utility>
#include <memory>

#include "pre_activate/common/pass.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "pre_activate/common/helper.h"
#include "pre_activate/common/optimizer.h"
#include "pre_activate/ascend/ascend_helper.h"

namespace mindspore {
namespace opt {
class TransDataSplit : public Pass {
 public:
  TransDataSplit() : Pass("trans_data_split"), kernel_select_(std::make_shared<KernelSelect>()) {}
  ~TransDataSplit() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  bool DoSplit(const FuncGraphPtr &func_graph, const AnfNodePtr &node);
  bool IsFormatInvaild(const AnfNodePtr &node);
  KernelSelectPtr kernel_select_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_IR_FUSION_TRANSDATA_SPLIT_H_
