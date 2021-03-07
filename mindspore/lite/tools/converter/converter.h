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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_CONVERTER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_CONVERTER_H

#include <memory>
#include <string>
#include "schema/inner/model_generated.h"
#include "tools/converter/graphdef_transform.h"
#include "tools/converter/model_parser.h"
#include "tools/converter/converter_flags.h"
#include "tools/converter/anf_transform.h"
#include "tools/converter/converter_context.h"
#include "load_mindir/load_model.h"

namespace mindspore {
namespace lite {
class Converter {
 public:
  static std::unique_ptr<Converter> CreateConverter(converter::FmkType fmk);

  virtual ~Converter() = default;

  virtual schema::MetaGraphT *Convert(const std::unique_ptr<converter::Flags> &flag);

  virtual FuncGraphPtr BuildFuncGraph(const std::string &model_file, const std::string &weight_file,
                                      schema::QuantType quant_type) = 0;

 protected:
  Converter() = default;

  std::unique_ptr<GraphDefTransform> metagraph_transform_ = std::make_unique<GraphDefTransform>();
  std::unique_ptr<AnfTransform> funcgraph_transform_ = std::make_unique<AnfTransform>();
};

class MindsporeImporter : public Converter {
 public:
  MindsporeImporter();

  ~MindsporeImporter() override = default;

  FuncGraphPtr BuildFuncGraph(const std::string &model_file, const std::string &weight_file,
                              schema::QuantType quant_type) override {
    auto func_graph = LoadMindIR(model_file);
    if (func_graph == nullptr) {
      MS_LOG(ERROR) << "get funcgraph failed.";
      return nullptr;
    }
    func_graph->set_attr("graph_name", MakeValue("main_graph"));
    func_graph->set_attr("fmk", MakeValue(static_cast<int>(converter::FmkType_MS)));
    return func_graph;
  }
};

int RunConverter(int argc, const char **argv);
}  // namespace lite
}  // namespace mindspore

#endif
