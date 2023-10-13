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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_CONFIG_PARSER_CONFIG_FILE_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_CONFIG_PARSER_CONFIG_FILE_PARSER_H_
#include <string>
#include <map>
#include <vector>
#include <memory>
#include "tools/converter/cxx_api/converter_para.h"

namespace mindspore {
namespace lite {
struct DataPreProcessString {
  std::string calibrate_path;
  std::string calibrate_size;
  std::string input_type;
  std::string image_to_format;
  std::string normalize_mean;
  std::string normalize_std;
  std::string resize_width;
  std::string resize_height;
  std::string resize_method;
  std::string center_crop_width;
  std::string center_crop_height;
};

struct CommonQuantString {
  std::string quant_type;
  std::string bit_num;
  std::string min_quant_weight_size;
  std::string min_quant_weight_channel;
  std::string skip_quant_node;
  std::string debug_info_save_path;
  std::string enable_encode;
  std::string workspace;  // support for over 2G model.
};

struct MixedBitWeightQuantString {
  std::string init_scale;
  std::string auto_tune;
  std::string use_cv_data;
  std::string max_iterations;
};

struct WeightQuantString {
  std::string dequant_strategy;
  std::string quant_strategy;
  std::string update_mindir;
  std::string max_segments;  // Define how many segments to use for parallel FSE
                             // It's recommended to use the max number of cores of your device
  std::string per_channel;
  std::string bias_correction;
};

struct FullQuantString {
  std::string activation_quant_method;
  std::string bias_correction;
  std::string target_device;
  std::string per_channel;
  std::string smooth_alpha;
  std::string enable_smooth_shift;
};

struct RegistryInfoString {
  std::string plugin_path;
  std::string disable_fusion;
  std::string fusion_blacklists;
};

struct AclOptionCfgString {
  std::string device_id;
  std::string input_format;
  std::string input_shape_vector;
  std::string input_shape;
  std::string output_type;
  std::string precision_mode;
  std::string op_select_impl_mode;
  std::string fusion_switch_config_file_path;
  std::string dynamic_batch_size;
  std::string buffer_optimize;
  std::string insert_op_config_file_path;
  std::string dynamic_image_size;
  std::string dynamic_dims;
  std::string aoe_mode;
  std::string custom_opp_path;
  std::map<std::string, std::string> init_options_map;
  std::map<std::string, std::string> build_options_map;
  std::map<std::string, std::string> aoe_global_options_map;
  std::map<std::string, std::string> aoe_tuning_options_map;
};

struct MicroParamString {
  std::string codegen_mode;
  std::string target;
  std::string support_parallel;
  std::string debug_mode;
  std::string enable_micro;
  std::string save_path;
  std::string project_name;
  std::string keep_original_weight;
  std::string changeable_weights_name;
};

struct CpuOptionCfgString {
  std::string architecture;
  std::string instruction;
};

struct TransformQuantString {
  std::string export_precision_mode;
};

struct DynamicQuantString {
  std::string quant_strategy;
};

struct OMConverterString {
  std::string input_name_vector;
  std::string input_shape_vector;
  std::string input_data_type_vector;
  std::string output_name_vector;
  std::string output_shape_vector;
  std::string output_data_type_vector;
};
using GraphKernelString = std::vector<std::string>;

class ConfigFileParser {
 public:
  int ParseConfigFile(const std::string &config_file_path,
                      std::map<int, std::map<std::string, std::string>> *model_param_infos);
  int ParseConfigParam(std::map<std::string, std::map<std::string, std::string>> *maps);
  bool SetParamByConfigfile(const std::shared_ptr<mindspore::ConverterPara> &param,
                            const std::map<std::string, std::string> &ascend_map);
  DataPreProcessString GetDataPreProcessString() const { return this->data_pre_process_string_; }
  CommonQuantString GetCommonQuantString() const { return this->common_quant_string_; }
  MixedBitWeightQuantString GetMixedBitWeightQuantString() const { return this->mixed_bit_quant_string_; }
  FullQuantString GetFullQuantString() const { return this->full_quant_string_; }
  WeightQuantString GetWeightQuantString() const { return this->weight_quant_string_; }
  RegistryInfoString GetRegistryInfoString() const { return this->registry_info_string_; }
  AclOptionCfgString GetAclOptionCfgString() { return this->acl_option_cfg_string_; }
  MicroParamString GetMicroParamString() { return this->micro_param_string_; }
  CpuOptionCfgString GetCpuOptionCfgString() { return this->cpu_option_cfg_string_; }
  TransformQuantString GetTransformQuantString() const { return this->transform_quant_string_; }
  DynamicQuantString GetDynamicQuantString() const { return this->dynamic_quant_string_; }
  GraphKernelString GetGraphKernelString() const { return this->graph_kernel_string_; }

 private:
  int ParseDataPreProcessString(const std::map<std::string, std::map<std::string, std::string>> &maps);
  int ParseCommonQuantString(const std::map<std::string, std::map<std::string, std::string>> &maps);
  int ParseMixedBitQuantString(const std::map<std::string, std::map<std::string, std::string>> &maps);
  int ParseFullQuantString(const std::map<std::string, std::map<std::string, std::string>> &maps);
  int ParseWeightQuantString(const std::map<std::string, std::map<std::string, std::string>> &maps);
  int ParseRegistryInfoString(const std::map<std::string, std::map<std::string, std::string>> &maps);
  int ParseAclOptionCfgString(const std::map<std::string, std::map<std::string, std::string>> &maps);
  int SetMapData(const std::map<std::string, std::string> &input_map,
                 const std::map<std::string, std::string &> &parse_map, const std::string &section);
  int ParseMicroParamString(const std::map<std::string, std::map<std::string, std::string>> &maps);
  int ParseCpuOptionCfgString(const std::map<std::string, std::map<std::string, std::string>> &maps);
  int ParseTransformQuantString(const std::map<std::string, std::map<std::string, std::string>> &maps);
  int ParseDynamicQuantString(const std::map<std::string, std::map<std::string, std::string>> &maps);
  int ParseGraphKernelString(const std::map<std::string, std::map<std::string, std::string>> &maps);
  void SetVariableParams(const std::shared_ptr<mindspore::ConverterPara> &param,
                         const std::map<std::string, std::string> &ascend_map);
  int ProcessVariableParam(const std::vector<std::string> &variable_param, std::vector<int64_t> &variable_index);
  int CheckVariableParm(const std::vector<int64_t> &variable_index);
  STATUS ParseCustomPattern(const std::shared_ptr<mindspore::ConverterPara> &param, std::string custom_pattern_str);
  int ParseOMConverterString(const std::map<std::string, std::map<std::string, std::string>> &maps);
  bool CheckPluginCustomOps(const std::vector<std::string> &plugin_custom_ops);

 private:
  DataPreProcessString data_pre_process_string_;
  CommonQuantString common_quant_string_;
  MixedBitWeightQuantString mixed_bit_quant_string_;
  FullQuantString full_quant_string_;
  WeightQuantString weight_quant_string_;
  RegistryInfoString registry_info_string_;
  AclOptionCfgString acl_option_cfg_string_;
  MicroParamString micro_param_string_;
  CpuOptionCfgString cpu_option_cfg_string_;
  TransformQuantString transform_quant_string_;
  DynamicQuantString dynamic_quant_string_;
  GraphKernelString graph_kernel_string_;
  std::vector<int64_t> inputs_variable_index_;
  std::vector<int64_t> outputs_variable_index_;
  OMConverterString om_converter_string_;
};

}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_CONFIG_PARSER_CONFIG_FILE_PARSER_H_
