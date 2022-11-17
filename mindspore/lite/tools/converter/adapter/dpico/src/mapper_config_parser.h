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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_MAPPER_CONFIG_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_MAPPER_CONFIG_PARSER_H_

#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <map>
#include "ir/dtype/type_id.h"

namespace mindspore {
namespace dpico {
constexpr auto kInsertOpConf = "[insert_op_conf]";
constexpr auto kInstructionName = "[instruction_name]";
constexpr auto kImageList = "[image_list]";
constexpr auto kInputType = "[input_type]";
constexpr auto kInputShape = "[input_shape]";
constexpr auto kGfpqParamFile = "[gfpq_param_file]";
constexpr auto kOutNodes = "[out_nodes]";
constexpr auto kOutputType = "[output_type]";
constexpr auto kRelatedInputRank = "related_input_rank";
constexpr auto kInputFormat = "input_format";
constexpr auto kModelFormat = "model_format";
constexpr auto kMeanChn = "mean_chn";
constexpr auto kMeanChn0 = "mean_chn_0";
constexpr auto kMeanChn1 = "mean_chn_1";
constexpr auto kMeanChn2 = "mean_chn_2";
constexpr auto kMeanChn3 = "mean_chn_3";
constexpr auto kVarReciChn = "var_reci_chn";
constexpr auto kVarReciChn0 = "var_reci_chn_0";
constexpr auto kVarReciChn1 = "var_reci_chn_1";
constexpr auto kVarReciChn2 = "var_reci_chn_2";
constexpr auto kVarReciChn3 = "var_reci_chn_3";
constexpr size_t kNumPrecision = 10;

struct AippModule {
  std::string input_format;
  std::string model_format;
  std::map<int, double> mean_map;
  std::map<int, double> val_map;
};

class MapperConfigParser {
 public:
  static MapperConfigParser *GetInstance();
  int Parse(const std::string &cfg_file, const std::vector<std::string> &graph_input_names);
  int AddImageList(const std::string &op_name, const std::string &calib_data_path);
  const std::unordered_map<std::string, std::string> &GetCommonConfig() const { return mapper_config_; }
  const std::unordered_map<std::string, std::string> &GetImageLists() const { return image_lists_; }
  const std::unordered_map<std::string, struct AippModule> &GetAippModules() const { return aipp_; }
  const std::string &GetOriginConfigPath() const { return origin_config_file_path_; }
  const std::string &GetOutputPath() const { return tmp_generated_file_dir_; }
  void SetOriginConfigFilePath(const std::string &origin_config_file_path);

 private:
  MapperConfigParser() = default;
  ~MapperConfigParser() = default;
  int ParseInputType(const std::string &input_type_str, const std::vector<std::string> &graph_input_names);
  int ParseImageList(const std::string &image_list_str, const std::vector<std::string> &graph_input_names);
  int ParseRawLine(const std::string &raw_line, const std::vector<std::string> &graph_input_names,
                   size_t *graph_input_idx);
  int ParseAippModule(const std::string &aipp_cfg, const std::vector<std::string> &graph_input_names);

  std::unordered_map<std::string, std::string> mapper_config_;
  std::unordered_map<std::string, std::string> image_lists_;
  std::unordered_map<std::string, struct AippModule> aipp_;
  std::string origin_config_file_path_;
  std::string tmp_generated_file_dir_;
};
}  // namespace dpico
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_MAPPER_CONFIG_PARSER_H_
