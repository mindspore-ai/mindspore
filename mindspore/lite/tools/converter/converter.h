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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_CONVERTER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_CONVERTER_H_

#include <memory>
#include <string>
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
constexpr auto kMaxSplitRatio = 10;
constexpr auto kComputeRate = "computeRate";
constexpr auto kSplitDevice0 = "device0";
constexpr auto kSplitDevice1 = "device1";

int RunConverter(const std::shared_ptr<ConverterPara> &param, void **model_data = nullptr, size_t *data_size = nullptr,
                 bool not_save = false);

class ConverterImpl {
 public:
  ConverterImpl() = default;
  ~ConverterImpl() {}

  int Convert(const std::shared_ptr<ConverterPara> &param, void **model_data, size_t *data_size, bool not_save);

 private:
  int InitConfigParam(const std::shared_ptr<ConverterPara> &param);
  int InitExtendedIntegrationInfo(const std::shared_ptr<ConverterPara> &param,
                                  const lite::ConfigFileParser &config_file_parser);
  bool CheckOfflineParallelConfig(const std::string &file, ParallelSplitConfig *parallel_split_config);
  std::string GetStrFromConfigFile(const std::string &file, const std::string &target_key);
  int SaveGraph(FuncGraphPtr graph, const std::shared_ptr<ConverterPara> &param, void **model_data, size_t *data_size,
                bool not_save);
  int SaveMindIRModel(FuncGraphPtr graph, const std::shared_ptr<ConverterPara> &param, void **model_data,
                      size_t *data_size);
  int LoadPluginLib(const std::shared_ptr<ConverterPara> &param);
};
}  // namespace lite
}  // namespace mindspore

#endif
