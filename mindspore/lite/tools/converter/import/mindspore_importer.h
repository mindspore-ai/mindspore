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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_IMPORT_MINDSPORE_IMPORTER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_IMPORT_MINDSPORE_IMPORTER_H_

#include <set>
#include <string>
#include <vector>
#include <memory>
#include "load_mindir/load_model.h"
#include "tools/converter/cxx_api/converter_para.h"
#include "include/errorcode.h"

namespace mindspore::lite {
class MindsporeImporter {
 public:
  MindsporeImporter() = default;
  ~MindsporeImporter() = default;
  FuncGraphPtr ImportMindIR(const std::shared_ptr<ConverterPara> &param);
  FuncGraphPtr ImportMindIR(const std::shared_ptr<ConverterPara> &param, const void *buff, const size_t &size);
  FuncGraphPtr CheckAndUpdateFuncGraph(const std::shared_ptr<ConverterPara> &param, FuncGraphPtr func_graph);

 private:
  static void RemoveUnusedGraphInput(const FuncGraphPtr &func_graph);
  STATUS GetFuncGraphOutputName(const CNodePtr &cnode);
  STATUS TraceOutput(const AnfNodePtr &node);
  STATUS Mindir2AnfAdjust(const FuncGraphPtr &func_graph, const std::shared_ptr<ConverterPara> &param);
  std::vector<std::string> output_tensor_name_;
};

}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_IMPORT_MINDSPORE_IMPORTER_H_
