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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_CXX_API_CONVERTER_PARA_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_CXX_API_CONVERTER_PARA_H_

#include <map>
#include <string>
#include <vector>
#include <set>
#include "include/converter.h"
#include "tools/converter/quantizer/quant_params.h"
#include "tools/converter/preprocess/preprocess_param.h"
#include "tools/converter/adapter/acl/common/acl_types.h"
#include "tools/converter/micro/coder/config.h"

namespace mindspore {
enum ParallelSplitType { SplitNo = 0, SplitByUserRatio = 1, SplitByUserAttr = 2 };

struct ParallelSplitConfig {
  ParallelSplitType parallel_split_type_ = SplitNo;
  std::vector<int64_t> parallel_compute_rates_;
  std::vector<std::string> parallel_devices_;
};

struct CpuOptionCfg {
  std::string architecture;
  std::string instruction;
};

struct ConverterPara {
  converter::FmkType fmk_type;
  std::string model_file;
  std::string output_file;
  std::string weight_file;

  std::string config_file;
  std::map<std::string, std::map<std::string, std::string>> config_param;
  bool weight_fp16 = false;
  std::map<std::string, std::vector<int64_t>> input_shape;
  Format input_format = NHWC;
  Format spec_input_format = DEFAULT_FORMAT;
  DataType input_data_type = DataType::kNumberTypeFloat32;
  DataType output_data_type = DataType::kNumberTypeFloat32;
  ModelType export_mindir = kMindIR_Lite;
  std::string decrypt_key;
  std::string decrypt_mode = "AES-GCM";
  std::string encrypt_key;
  std::string encrypt_mode = "AES-GCM";  // inner
  bool enable_encryption = false;
  bool pre_infer = false;
  bool train_model = false;
  bool no_fusion = false;
  bool optimize_transformer = false;
  bool is_runtime_converter = false;
  std::set<std::string> fusion_blacklists;

  // inner
  std::vector<std::string> plugins_path;
  lite::quant::CommonQuantParam commonQuantParam;
  lite::quant::MixedBitWeightQuantParam mixedBitWeightQuantParam;
  lite::quant::FullQuantParam fullQuantParam;
  lite::quant::WeightQuantParam weightQuantParam;
  lite::preprocess::DataPreProcessParam dataPreProcessParam;
  lite::acl::AclModelOptionCfg aclModelOptionCfgParam;
  lite::micro::MicroParam microParam;
  ParallelSplitConfig parallel_split_config;
  std::string device;
  CpuOptionCfg cpuOptionCfgParam;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_CXX_API_CONVERTER_PARA_H_
