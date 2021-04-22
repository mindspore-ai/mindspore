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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MODEL_PARSER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_MODEL_PARSER_H
#include <google/protobuf/message.h>
#include <string>
#include <memory>
#include "schema/inner/model_generated.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "tools/converter/converter_context.h"
#include "tools/converter/converter_flags.h"
#include "tools/converter/quant_param_holder.h"

namespace mindspore::lite {
using namespace schema;
class ModelParser {
 public:
  ModelParser() = default;

  virtual ~ModelParser() = default;

  FuncGraphPtr Parse(const std::string &model_file, const std::string &weight_file, const QuantType &quant_type) {
    auto ret = ParseToFuncGraph(model_file, weight_file, quant_type);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Parse to func graph failed : " << ret;
      return nullptr;
    }
    ret = PostAdjust();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Adjust func graph failed : " << ret;
      return nullptr;
    }
    return this->res_graph_;
  }

 protected:
  virtual int ParseToFuncGraph(const std::string &model_file, const std::string &weight_file,
                               const QuantType &quant_type) = 0;

  virtual int PostAdjust() = 0;

 protected:
  FuncGraphPtr res_graph_ = nullptr;
};
}  // namespace mindspore::lite

#endif
