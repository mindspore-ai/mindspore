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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_CONVERTER_FUNCGRAPH_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_CONVERTER_FUNCGRAPH_H_

#include <memory>
#include <string>
#include <vector>
#include "include/converter.h"
#include "include/registry/model_parser.h"
#include "schema/inner/model_generated.h"
#include "tools/converter/graphdef_transform.h"
#include "include/registry/model_parser_registry.h"
#include "tools/converter/anf_transform.h"
#include "tools/converter/converter_context.h"
#include "tools/common/graph_util.h"
#include "tools/converter/preprocess/preprocess_param.h"
#include "tools/converter/quantizer/quant_params.h"
#include "tools/converter/adapter/acl/common/acl_types.h"
#include "micro/coder/config.h"
#include "tools/converter/cxx_api/converter_para.h"
#include "tools/converter/config_parser/config_file_parser.h"

namespace mindspore {
namespace lite {
class ConverterFuncGraph {
 public:
  static FuncGraphPtr Build(const std::shared_ptr<ConverterPara> &param);
  static STATUS Optimize(const std::shared_ptr<ConverterPara> &param, FuncGraphPtr func_graph);
  static STATUS Save(const std::shared_ptr<ConverterPara> &param, const FuncGraphPtr &func_graph, void **buff,
                     size_t *size);

 private:
  static STATUS UnifyFuncGraphForInfer(const std::shared_ptr<ConverterPara> &param, FuncGraphPtr func_graph,
                                       std::vector<std::string> *output_names);
  static STATUS UnifyFuncGraphInputFormat(const std::shared_ptr<ConverterPara> &param, FuncGraphPtr func_graph);
  static STATUS UnifyFuncGraphInputDataType(const std::shared_ptr<ConverterPara> &param, FuncGraphPtr func_graph);
  static FuncGraphPtr Load(const std::shared_ptr<ConverterPara> &param);
  static FuncGraphPtr Load3rdModelToFuncgraph(const std::shared_ptr<ConverterPara> &param);
};
}  // namespace lite
}  // namespace mindspore

#endif
