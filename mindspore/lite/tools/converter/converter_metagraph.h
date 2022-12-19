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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_CONVERTER_METAGRAPH_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_CONVERTER_METAGRAPH_H_

#include <memory>
#include <string>
#include <vector>
#include "tools/converter/graphdef_transform.h"
#include "tools/common/graph_util.h"

namespace mindspore {
namespace lite {
class ConverterToMetaGraph {
 public:
  static schema::MetaGraphT *Build(const std::shared_ptr<ConverterPara> &param, FuncGraphPtr func_graph);
  static STATUS Save(schema::MetaGraphT *meta_graph, const std::shared_ptr<ConverterPara> &param, void **model_data,
                     size_t *data_size, bool not_save);

 private:
  static STATUS UnifyFuncGraphFormat(const std::shared_ptr<ConverterPara> &param, const FuncGraphPtr &old_graph);
  static STATUS UpdateMetaGraphOutputName(schema::MetaGraphT *meta_graph, const std::vector<std::string> &output_names);
};
}  // namespace lite
}  // namespace mindspore

#endif
