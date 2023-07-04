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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_LITE_TENSOR_EXTRACTOR_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_LITE_TENSOR_EXTRACTOR_H_

#include <vector>
#include "include/registry/converter_context.h"
#include "ir/anf.h"
#include "src/tensor.h"
#include "tools/lite_exporter/fetch_content.h"

namespace mindspore {
namespace opt {
class LiteTensorExtractor {
 public:
  LiteTensorExtractor() = default;
  ~LiteTensorExtractor() = default;
  static int GetCNodeInputAbstractLists(const CNodePtr &cnode, AbstractBasePtrList *abs_list);
  static int GetCNodeInputTensors(const CNodePtr &cnode, std::vector<TensorPtr> *inputs, converter::FmkType fmk_type,
                                  bool train_flag, bool copy_data);
  static int GetCNodeOutputTensors(const CNodePtr &cnode, std::vector<TensorPtr> *outputs, bool train_flag);
  static int GetCNodeConstInputs(const CNodePtr &cnode, const converter::FmkType &fmk_type, const bool &train_flag,
                                 const bool &copy_data, std::vector<TensorPtr> *const_ms_inputs);
  static int GetCNodeConstInput(const CNodePtr &cnode, const size_t &index, const converter::FmkType &fmk_type,
                                const bool &train_flag, const bool &copy_data, std::vector<TensorPtr> *const_ms_inputs);
  static int GetCNodeVarInput(const CNodePtr &cnode, const size_t &index, const converter::FmkType &fmk_type,
                              std::vector<TensorPtr> *var_ms_inputs);
  static int GetCNodeConstInputToAbstract(const CNodePtr &cnode, const AbstractBasePtrList &abs_list,
                                          converter::FmkType fmk_type, bool train_flag);
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_LITE_TENSOR_EXTRACTOR_H_
