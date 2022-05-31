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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_EXPORT_MODEL_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_EXPORT_MODEL_H_

#include <map>
#include <memory>
#include "include/errorcode.h"
#include "ir/func_graph.h"
#include "tools/converter/cxx_api/converter_para.h"

namespace mindspore {
namespace lite {
FuncGraphPtr CloneFuncGraph(const FuncGraphPtr &graph, const std::shared_ptr<ConverterPara> &param,
                            std::map<FuncGraphPtr, FuncGraphPtr> *cloned_func_graph);
STATUS ExportModel(const FuncGraphPtr &graph, const std::shared_ptr<ConverterPara> &param);
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_EXPORT_MODEL_H_
