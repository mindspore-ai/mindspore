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

#define USE_DEPRECATED_API
#include "tools/converter/converter.h"
#include <memory>
#include <vector>
#include <set>
#include <map>
#include <tuple>
#include <algorithm>
#include <utility>
#include "src/common/log_adapter.h"
#include "tools/common/meta_graph_serializer.h"
#include "tools/lite_exporter/anf_exporter.h"
#include "tools/graph_kernel/converter/graph_kernel_optimization.h"
#ifdef SUPPORT_TRAIN
#include "src/train/train_populate_parameter.h"
#endif
#include "include/registry/model_parser_registry.h"
#include "include/api/format.h"
#include "src/common/dynamic_library_loader.h"
#include "src/common/log_util.h"
#include "tools/converter/parser/parser_utils.h"
#include "tools/converter/import/mindspore_importer.h"
#include "nnacl/op_base.h"
#include "tools/converter/micro/coder/coder.h"
#include "src/common/prim_util.h"
#include "src/common/version_manager.h"
#include "tools/common/tensor_util.h"
#include "include/api/model.h"
#include "tools/mindir_exporter/mindir_serializer.h"
#include "src/common/primitive_t_utils.h"
#include "tools/converter/config_parser/acl_option_param_parser.h"
#include "tools/converter/config_parser/micro_param_parser.h"
#include "tools/converter/config_parser/preprocess_parser.h"
#include "tools/converter/config_parser/quant_param_parser.h"
#include "tools/converter/converter_funcgraph.h"
#include "tools/converter/converter_metagraph.h"
#include "tools/common/string_util.h"
#include "src/common/file_utils.h"
#include "ops/dynamic_shape.h"
#include "tools/common/parse_config_utils.h"
#include "tools/converter/converter_packed_node.h"
#include "tools/converter/config_parser/cpu_option_param_parser.h"

namespace mindspore {
std::map<std::string, Format> StrToEnumFormatMap = {{"NHWC", Format::NHWC}, {"NCHW", Format::NCHW}};
extern "C" {
void mindspore_log_init();
}
namespace lite {
#define CONVERTER_LOG_ERROR(str)   \
  do {                             \
    MS_LOG(ERROR) << str;          \
    std::cout << str << std::endl; \
  } while (0);

namespace {
constexpr size_t kMaxNum1024 = 1024;
constexpr size_t kPluginPathMaxNum = 10;
constexpr int kPathLengthUpperLimit = 1024;
constexpr size_t kEncMaxLen = 16;
constexpr auto kFmk = "fmk";
constexpr auto kModelFile = "modelFile";
constexpr auto kOutputFile = "outputFile";
constexpr auto kWeightFile = "weightFile";
constexpr auto kFp16 = "fp16";
constexpr auto kInputshape = "inputShape";
constexpr auto kInputDataFormat = "inputDataFormat";
constexpr auto kEncryptKey = "encryptKey";
constexpr auto kEncryption = "encryption";
constexpr auto kInputDataType = "inputDataType";
constexpr auto kOutputDataType = "outputDataType";
constexpr auto kInfer = "infer";
std::map<std::string, FmkType> StrToEnumFmkTypeMap = {
  {"CAFFE", FmkType::kFmkTypeCaffe}, {"MINDIR", FmkType::kFmkTypeMs}, {"TFLITE", FmkType::kFmkTypeTflite},
  {"ONNX", FmkType::kFmkTypeOnnx},   {"TF", FmkType::kFmkTypeTf},     {"PYTORCH", FmkType::kFmkTypePytorch}};
std::map<std::string, DataType> StrToEnumDataTypeMap = {{"FLOAT", DataType::kNumberTypeFloat32},
                                                        {"INT8", DataType::kNumberTypeInt8},
                                                        {"UINT8", DataType::kNumberTypeUInt8},
                                                        {"DEFAULT", DataType::kTypeUnknown}};

#if defined(_WIN32) || defined(_WIN64)
static const char kSlash[] = "\\";
#else
static const char kSlash[] = "/";
#endif

// Deal with early release of 3rd-party plugin library.
static std::vector<std::shared_ptr<DynamicLibraryLoader>> dl_loaders;
bool FileExists(const std::string &path) {
  std::ifstream file(path);
  return file.good();
}

int InitModelFmk(const std::string &value, const std::shared_ptr<ConverterPara> &param) {
  if (StrToEnumFmkTypeMap.find(value) != StrToEnumFmkTypeMap.end()) {
    param->fmk_type = StrToEnumFmkTypeMap.at(value);
  } else {
    std::cerr << "INPUT ILLEGAL: fmk must be TF|TFLITE|CAFFE|MINDIR|ONNX" << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int InitModelFile(const std::string &value, const std::shared_ptr<ConverterPara> &param) {
  if (value.empty() || !FileExists(value)) {
    MS_LOG(ERROR) << "model file path is empty or invalid";
    return RET_INPUT_PARAM_INVALID;
  }
  param->model_file = value;
  return RET_OK;
}

int InitModelInputDataType(const std::string &value, const std::shared_ptr<ConverterPara> &param) {
  if (StrToEnumDataTypeMap.find(value) == StrToEnumDataTypeMap.end()) {
    std::cerr << "INPUT INVALID: inputDataType is invalid, supported inputDataType: FLOAT | INT8 | UINT8 | DEFAULT"
              << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  param->input_data_type = StrToEnumDataTypeMap.at(value);
  return RET_OK;
}

int InitModelOutputDataType(const std::string &value, const std::shared_ptr<ConverterPara> &param) {
  if (StrToEnumDataTypeMap.find(value) == StrToEnumDataTypeMap.end()) {
    std::cerr << "OUTPUT INVALID: outputDataType is invalid, supported outputDataType: FLOAT | INT8 | UINT8 | DEFAULT"
              << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  param->output_data_type = StrToEnumDataTypeMap.at(value);
  return RET_OK;
}

int InitModelSaveFP16(const std::string &value, const std::shared_ptr<ConverterPara> &param) {
  if (value == "on") {
    param->weight_fp16 = true;
  } else if (value.empty() || value == "off") {
    param->weight_fp16 = false;
  } else {
    std::cerr << "Init save_fp16 failed." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int InitModelTrainMode(const std::string &value, const std::shared_ptr<ConverterPara> &param) {
  if (value == "true") {
    param->train_model = true;
  } else if (value.empty() || value == "false") {
    param->train_model = false;
  } else {
    std::cerr << "INPUT ILLEGAL: trainModel must be true|false " << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int InitModelInputShape(const std::string &value, const std::shared_ptr<ConverterPara> &param) {
  if (value.empty()) {
    return RET_OK;
  }
  std::vector<int64_t> shape;
  auto shape_strs = lite::StrSplit(value, std::string(";"));
  for (const auto &shape_str : shape_strs) {
    if (shape_str.empty()) {
      continue;
    }
    shape.clear();
    auto string_split = lite::StrSplit(shape_str, std::string(":"));
    constexpr int kMinShapeSizeInStr = 2;
    if (string_split.size() < kMinShapeSizeInStr) {
      MS_LOG(ERROR) << "shape size must not be less than " << kMinShapeSizeInStr;
      return lite::RET_INPUT_PARAM_INVALID;
    }
    auto name = string_split[0];
    for (size_t i = 1; i < string_split.size() - 1; ++i) {
      name += ":" + string_split[i];
    }
    if (name.empty()) {
      MS_LOG(ERROR) << "input tensor name is empty";
      return lite::RET_INPUT_PARAM_INVALID;
    }
    auto dim_strs = string_split[string_split.size() - 1];
    if (dim_strs.empty()) {
      MS_LOG(ERROR) << "input tensor dim string is empty";
      return lite::RET_INPUT_PARAM_INVALID;
    }
    auto dims = lite::StrSplit(dim_strs, std::string(","));
    if (dims.empty()) {
      MS_LOG(ERROR) << "input tensor dim is empty";
      return lite::RET_INPUT_PARAM_INVALID;
    }
    for (const auto &dim : dims) {
      int64_t dim_value;
      try {
        dim_value = std::stoi(dim);
      } catch (const std::exception &e) {
        MS_LOG(ERROR) << "Get dim failed: " << e.what();
        return lite::RET_INPUT_PARAM_INVALID;
      }
      shape.push_back(dim_value);
    }
    param->input_shape[name] = shape;
  }
  return RET_OK;
}

int InitModelInputDataFormat(const std::string &value, const std::shared_ptr<ConverterPara> &param) {
  if (StrToEnumFormatMap.find(value) != StrToEnumFormatMap.end()) {
    param->input_format = StrToEnumFormatMap.at(value);
  } else if (!value.empty()) {
    MS_LOG(ERROR) << "Input format is invalid.";
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int InitModelInfer(const std::string &value, const std::shared_ptr<ConverterPara> &param) {
  if (value == "true") {
    param->pre_infer = true;
  } else if (value == "false" || value.empty()) {
    param->pre_infer = false;
  } else {
    std::cerr << "INPUT ILLEGAL: infer must be true|false " << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int InitModelNoFusion(const std::string &value, const std::shared_ptr<ConverterPara> &param) {
  if (value == "true") {
    param->no_fusion = true;
  } else if (value == "false") {
    param->no_fusion = false;
  } else if (!value.empty()) {
    std::cerr << "INPUT ILLEGAL: NoFusion must be true|false " << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

std::shared_ptr<ConverterPara> CreateConvertParam(const std::map<std::string, string> &model_params) {
  auto parm = std::make_shared<ConverterPara>();
  std::map<std::string, std::function<int(const std::string &, const std::shared_ptr<ConverterPara> &)>> parse_funcs = {
    {"fmk", InitModelFmk},
    {"modelFile", InitModelFile},
    {"inputDataType", InitModelInputDataType},
    {"outputDataType", InitModelOutputDataType},
    {"fp16", InitModelSaveFP16},
    {"trainModel", InitModelTrainMode},
    {"inputShape", InitModelInputShape},
    {"inputDataFormat", InitModelInputDataFormat},
    {"infer", InitModelInfer},
    {"NoFusion", InitModelNoFusion}};
  if (model_params.find("fmk") == model_params.end() || model_params.find("modelFile") == model_params.end()) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: fmk and modelFile must be set in [model_param].";
    return nullptr;
  }
  for (auto &pair : model_params) {
    if (parse_funcs.find(pair.first) == parse_funcs.end()) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: `" << pair.first << "` is not supported in [model_param]";
      return nullptr;
    }
    if (parse_funcs[pair.first](pair.second, parm) != RET_OK) {
      MS_LOG(ERROR) << pair.first << "'value is invalid";
      return nullptr;
    }
  }
  return parm;
}
}  // namespace

STATUS StoreConverterParameters(const std::shared_ptr<ConverterPara> &param) {
  if (param == nullptr) {
    MS_LOG(ERROR) << "Input param is nullptr";
    return RET_INPUT_PARAM_INVALID;
  }
  std::string param_input_shape;
  for (auto i = param->input_shape.cbegin(); i != param->input_shape.cend(); ++i) {
    std::stringstream input_shape_ss;
    string input_shape_str;
    (void)copy(i->second.begin(), i->second.end(), std::ostream_iterator<int>(input_shape_ss, ","));
    input_shape_str = input_shape_ss.str();
    input_shape_str.erase(input_shape_str.end() - 1);
    param_input_shape += i->first + ":" + input_shape_str + ";";
  }
  std::map<std::string, std::map<std::string, std::string>> conver_param_maps;
  conver_param_maps[mindspore::converter::KConverterParam][kFmk] = std::to_string(param->fmk_type);
  conver_param_maps[mindspore::converter::KConverterParam][kModelFile] = param->model_file;
  conver_param_maps[mindspore::converter::KConverterParam][kOutputFile] = param->output_file;
  conver_param_maps[mindspore::converter::KConverterParam][kWeightFile] = param->weight_file;
  std::stringstream weight_fp16_ss;
  weight_fp16_ss << std::boolalpha << param->weight_fp16;
  conver_param_maps[mindspore::converter::KConverterParam][kFp16] = weight_fp16_ss.str();
  conver_param_maps[mindspore::converter::KConverterParam][kInputshape] = param_input_shape;
  conver_param_maps[mindspore::converter::KConverterParam][kInputDataFormat] = std::to_string(param->input_format);
  conver_param_maps[mindspore::converter::KConverterParam][kEncryptKey] = param->encrypt_key;
  std::stringstream encryption_ss;
  encryption_ss << std::boolalpha << param->enable_encryption;
  conver_param_maps[mindspore::converter::KConverterParam][kEncryption] = encryption_ss.str();
  conver_param_maps[mindspore::converter::KConverterParam][kInputDataType] =
    std::to_string(static_cast<int>(param->input_data_type));
  conver_param_maps[mindspore::converter::KConverterParam][kOutputDataType] =
    std::to_string(static_cast<int>(param->output_data_type));
  std::stringstream pre_infer_ss;
  pre_infer_ss << std::boolalpha << param->pre_infer;
  conver_param_maps[mindspore::converter::KConverterParam][kInfer] = pre_infer_ss.str();
  for (const auto &config_info : conver_param_maps) {
    ConverterInnerContext::GetInstance()->SetExternalUsedConfigInfos(config_info.first, config_info.second);
  }
  return RET_OK;
}

int CheckExistCustomOps(const schema::MetaGraphT *meta_graph, bool *exist_custom_nodes) {
  MS_CHECK_TRUE_MSG(meta_graph != nullptr && exist_custom_nodes != nullptr, RET_ERROR, "input params contain nullptr.");
  flatbuffers::FlatBufferBuilder fbb(kMaxNum1024);
  for (const auto &node : meta_graph->nodes) {
    MS_CHECK_TRUE_RET(node != nullptr, RET_ERROR);
    auto prim = ConvertToPrimitive(node->primitive.get(), &fbb);
    if (prim == nullptr) {
      MS_LOG(ERROR) << "get primitive failed.";
      fbb.Clear();
      return RET_ERROR;
    }
    if (IsCustomNode(prim, static_cast<int>(SCHEMA_CUR))) {
      *exist_custom_nodes = true;
      break;
    }
  }
  fbb.Clear();
  return RET_OK;
}

int PreInference(const schema::MetaGraphT &meta_graph, bool train_model) {
  if (train_model) {
    MS_LOG(WARNING) << "train model dont support pre-infer.";
    return RET_OK;
  }

  bool exist_custom_nodes = false;
  auto check_ret = CheckExistCustomOps(&meta_graph, &exist_custom_nodes);
  if (check_ret == RET_ERROR) {
    MS_LOG(ERROR) << "CheckExistCustomOps failed.";
    return RET_ERROR;
  }
  if (exist_custom_nodes) {
    MS_LOG(WARNING) << "exist custom nodes and will not be pre-infer.";
    return RET_OK;
  }
  mindspore::Model model;
  flatbuffers::FlatBufferBuilder builder(kMaxNum1024);
  auto offset = schema::MetaGraph::Pack(builder, &meta_graph);
  builder.Finish(offset);
  schema::FinishMetaGraphBuffer(builder, offset);
  int size = builder.GetSize();
  auto content = builder.GetBufferPointer();
  if (content == nullptr) {
    MS_LOG(ERROR) << "GetBufferPointer nullptr";
    return RET_ERROR;
  }
  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    MS_LOG(ERROR) << "New context failed while running ";
    std::cerr << "New context failed while running " << std::endl;
    return RET_ERROR;
  }
  std::shared_ptr<CPUDeviceInfo> device_info = std::make_shared<CPUDeviceInfo>();
  auto &device_list = context->MutableDeviceInfo();
  device_list.push_back(device_info);

  auto ret = model.Build(content, size, kMindIR_Lite, context);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Build error ";
    std::cerr << "Build error " << std::endl;
    return RET_ERROR;
  }
  for (auto &tensor : model.GetInputs()) {
    if (tensor.Shape().empty() || tensor.DataSize() == 0 ||
        std::find(tensor.Shape().begin(), tensor.Shape().end(), -1) != tensor.Shape().end()) {
      MS_LOG(WARNING) << tensor.Name() << " is dynamic shape and will not be pre-infer.";
      return RET_OK;
    }
    auto status = GenerateRandomData(&tensor);
    if (status != RET_OK) {
      MS_LOG(ERROR) << tensor.Name() << "GenerateRandomData failed.";
      return status;
    }
  }
  std::vector<MSTensor> outputs;
  ret = model.Predict(model.GetInputs(), &outputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Inference error ";
    std::cerr << "Inference error " << std::endl;
    return RET_ERROR;
  }
  return RET_OK;
}

int ConverterImpl::InitConfigParam(const std::shared_ptr<ConverterPara> &param,
                                   std::map<int, std::map<std::string, std::string>> *model_param_infos) {
  model_param_infos->clear();
  lite::ConfigFileParser config_parser;
  std::map<std::string, std::map<std::string, std::string>> maps;
  auto ret = RET_OK;
  auto parse_map_ret = RET_OK;
  if (!param->config_file.empty()) {
    ret = config_parser.ParseConfigFile(param->config_file, nullptr);
    parse_map_ret = mindspore::lite::ParseConfigFile(param->config_file, &maps, model_param_infos);
  } else {
    ret = config_parser.ParseConfigParam(&param->config_param);
  }
  if (ret != RET_OK || parse_map_ret != RET_OK) {
    MS_LOG(ERROR) << "Parse config param failed.";
    return ret;
  }
  if (model_param_infos->empty()) {
    ret = lite::PreprocessParser::ParsePreprocess(config_parser.GetDataPreProcessString(), &param->dataPreProcessParam);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Parse preprocess failed.";
      return ret;
    }
    ret = lite::QuantParamParser::ParseCommonQuant(config_parser.GetCommonQuantString(), &param->commonQuantParam);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Parse common quant param failed.";
      return ret;
    }
    ret = lite::QuantParamParser::ParseFullQuant(config_parser.GetFullQuantString(), &param->fullQuantParam);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Parse full quant param failed.";
      return ret;
    }
    ret = lite::QuantParamParser::ParseWeightQuant(config_parser.GetWeightQuantString(), &param->weightQuantParam);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Parse full quant param failed.";
      return ret;
    }
    ret = lite::QuantParamParser::ParseMixedBitWeightQuant(config_parser.GetMixedBitWeightQuantString(),
                                                           &param->mixedBitWeightQuantParam);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Parse mixed bit weight quant param failed.";
      return ret;
    }
    ret = InitExtendedIntegrationInfo(param, config_parser);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Parse extended integration info failed.";
      return ret;
    }
    std::string output_file = param->output_file;
    param->aclModelOptionCfgParam.om_file_path = output_file;
    auto dir_pos = output_file.find_last_of('/');
    param->aclModelOptionCfgParam.dump_model_name =
      dir_pos != std::string::npos ? output_file.substr(dir_pos + 1) : output_file;
    lite::AclOptionParamParser acl_param_parser;
    ret = acl_param_parser.ParseAclOptionCfg(config_parser.GetAclOptionCfgString(), &param->aclModelOptionCfgParam);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Parse acl option param failed.";
      return ret;
    }
    // parse ascend_context in config file, the priority is higher
    if (maps.find("ascend_context") != maps.end()) {
      auto map = maps.at("ascend_context");
      config_parser.SetParamByConfigfile(param, map);
    }
    if (!param->config_file.empty()) {
      (void)CheckOfflineParallelConfig(param->config_file, &param->parallel_split_config);
    }

    lite::CpuOptionParamParser cpu_param_parser;
    ret = cpu_param_parser.ParseCpuOptionCfg(config_parser.GetCpuOptionCfgString(), &param->cpuOptionCfgParam);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Parse cpu option param failed.";
      return ret;
    }
  } else {
    MS_LOG(WARNING) << "Multi mode only support micro_param and model_param, other configure can not take effect";
  }
  lite::MicroParamParser micro_param_parser;
  ret = micro_param_parser.ParseMicroParam(config_parser.GetMicroParamString(), &param->microParam);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse micro param failed.";
    return ret;
  }

  return RET_OK;
}

int ConverterImpl::InitExtendedIntegrationInfo(const std::shared_ptr<ConverterPara> &param,
                                               const lite::ConfigFileParser &config_parser) {
  auto extended_info = config_parser.GetRegistryInfoString();
  if (!extended_info.plugin_path.empty()) {
    const char delimiter = ';';
    auto relative_path = lite::SplitStringToVector(extended_info.plugin_path, delimiter);
    if (relative_path.size() > kPluginPathMaxNum) {
      MS_LOG(ERROR) << "extended plugin library's num is too big, which shouldn't be larger than " << kPluginPathMaxNum;
      return RET_INPUT_PARAM_INVALID;
    }
    for (auto &i : relative_path) {
      param->plugins_path.push_back(lite::RealPath(i.c_str()));
    }
  }

  if (!extended_info.disable_fusion.empty()) {
    if (extended_info.disable_fusion == "on") {
      param->no_fusion = true;
    } else if (extended_info.disable_fusion == "off") {
      param->no_fusion = false;
    } else {
      std::cerr << "CONFIG SETTING ILLEGAL: disable_fusion should be on/off" << std::endl;
      return RET_INPUT_PARAM_INVALID;
    }
  }

  if (!extended_info.fusion_blacklists.empty()) {
    std::vector<std::string> fusions = SplitStringToVector(extended_info.fusion_blacklists, ",");
    for (const auto &fusion : fusions) {
      bool inserted = false;
      std::tie(std::ignore, inserted) = param->fusion_blacklists.insert(fusion);
      if (inserted) {
        MS_LOG(DEBUG) << "Value was inserted successfully.";
      }
    }
  }
  return RET_OK;
}

bool ConverterImpl::CheckOfflineParallelConfig(const std::string &file, ParallelSplitConfig *parallel_split_config) {
  // device: [device0 device1] ---> {cpu, gpu}
  // computeRate: [x: y] x >=0 && y >=0 && x/y < 10
  MS_ASSERT(parallel_split_config != nullptr);
  std::vector<std::string> config_devices = {"cpu", "gpu", "npu"};
  auto compute_rate_result = GetStrFromConfigFile(file, kComputeRate);
  if (compute_rate_result.empty()) {
    return false;
  }
  std::string device0_result = GetStrFromConfigFile(file, kSplitDevice0);
  if (device0_result.empty()) {
    return false;
  }
  std::string device1_result = GetStrFromConfigFile(file, kSplitDevice1);
  if (device1_result.empty()) {
    return false;
  }
  bool device0_flag = false;
  bool device1_flag = false;
  for (const auto &device : config_devices) {
    if (device == device0_result) {
      device0_flag = true;
    }
    if (device == device1_result) {
      device1_flag = true;
    }
  }
  if (!device0_flag || !device1_flag) {
    return false;
  }
  const char delimiter = ';';
  std::vector<std::string> device_rates = lite::SplitStringToVector(compute_rate_result, delimiter);
  const char colon = ':';
  for (const auto &device : device_rates) {
    std::vector<std::string> rate = lite::SplitStringToVector(device, colon);
    int64_t compute_rate = 0;
    try {
      compute_rate = std::stoi(rate.back());
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Get compute rate failed: " << e.what();
      return false;
    }
    parallel_split_config->parallel_compute_rates_.push_back(compute_rate);
  }
  const size_t support_rates_num = 2;
  if (parallel_split_config->parallel_compute_rates_.size() != support_rates_num) {
    return false;
  }
  int64_t bigger_rate = INT32_MIN;
  int64_t smaller_rate = INT32_MAX;
  for (const auto &rate : parallel_split_config->parallel_compute_rates_) {
    if (rate <= 0 || rate > INT32_MAX) {
      return false;
    }
    bigger_rate = std::max(rate, bigger_rate);
    smaller_rate = std::min(rate, smaller_rate);
  }
  parallel_split_config->parallel_devices_.push_back(device0_result);
  parallel_split_config->parallel_devices_.push_back(device1_result);
  // parall_split_type will extend by other user's attr
  parallel_split_config->parallel_split_type_ = SplitByUserRatio;
  if (smaller_rate == 0) {
    MS_LOG(ERROR) << "smaller_rate is zero";
    return false;
  }
  return bigger_rate / smaller_rate <= kMaxSplitRatio;
}

std::string ConverterImpl::GetStrFromConfigFile(const std::string &file, const std::string &target_key) {
  std::string res;
  if (file.empty()) {
    MS_LOG(ERROR) << "file is nullptr";
    return res;
  }
  auto resolved_path = std::make_unique<char[]>(PATH_MAX);
  if (resolved_path == nullptr) {
    MS_LOG(ERROR) << "new resolved_path failed";
    return "";
  }

#ifdef _WIN32
  auto *real_path = _fullpath(resolved_path.get(), file.c_str(), kPathLengthUpperLimit);
#else
  char *real_path = realpath(file.c_str(), resolved_path.get());
#endif
  if (real_path == nullptr || strlen(real_path) == 0) {
    MS_LOG(ERROR) << "file path is not valid : " << file;
    return "";
  }
  std::ifstream ifs(resolved_path.get());
  if (!ifs.good()) {
    MS_LOG(ERROR) << "file: " << real_path << " is not exist";
    return res;
  }
  if (!ifs.is_open()) {
    MS_LOG(ERROR) << "file: " << real_path << "open failed";
    return res;
  }
  std::string line;
  while (std::getline(ifs, line)) {
    lite::Trim(&line);
    if (line.empty() || line.at(0) == '#' || line.at(0) == '[') {
      continue;
    }
    auto index = line.find('=');
    if (index == std::string::npos) {
      MS_LOG(ERROR) << "the config file is invalid, can not find '=', please check";
      return "";
    }
    auto key = line.substr(0, index);
    auto value = line.substr(index + 1);
    lite::Trim(&key);
    lite::Trim(&value);
    if (key == target_key) {
      return value;
    }
  }
  return res;
}

int CheckFmkType(const std::shared_ptr<ConverterPara> &param) {
  if (param != nullptr) {
    std::set valid_values = {FmkType::kFmkTypeTf, FmkType::kFmkTypeCaffe,  FmkType::kFmkTypeOnnx,
                             FmkType::kFmkTypeMs, FmkType::kFmkTypeTflite, FmkType::kFmkTypePytorch};
    if (std::find(valid_values.begin(), valid_values.end(), param->fmk_type) == valid_values.end()) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: fmk_type must be kFmkTypeTf|kFmkTypeCaffe|kFmkTypeOnnx|kFmkTypeMs|kFmkTypeTflite"
                    << ", but got " << param->fmk_type;
      return RET_INPUT_PARAM_INVALID;
    }
    if (param->fmk_type != converter::kFmkTypeCaffe && !param->weight_file.empty()) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: weight_file is not a valid flag";
      return RET_INPUT_PARAM_INVALID;
    }
  }
  return RET_OK;
}

int CheckModelFile(const std::shared_ptr<ConverterPara> &param) {
  if (param != nullptr) {
    if (param->model_file.empty()) {
      MS_LOG(ERROR) << "INPUT MISSING: model file path is necessary";
      return RET_INPUT_PARAM_INVALID;
    }
  }
  return RET_OK;
}

int CheckOutputFile(const std::shared_ptr<ConverterPara> &param) {
  if (param != nullptr && param->aclModelOptionCfgParam.offline) {
    if (param->output_file.empty()) {
      MS_LOG(ERROR) << "INPUT MISSING: output file path is necessary";
      return RET_INPUT_PARAM_INVALID;
    }

#ifdef _WIN32
    replace(param->output_file.begin(), param->output_file.end(), '/', '\\');
#endif

    if (param->output_file.rfind('/') == param->output_file.length() - 1 ||
        param->output_file.rfind('\\') == param->output_file.length() - 1) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: output file must be a valid file path";
      return RET_INPUT_PARAM_INVALID;
    }
    param->aclModelOptionCfgParam.om_file_path = param->output_file;
  }
  return RET_OK;
}

int CheckInputShape(const std::shared_ptr<ConverterPara> &param) {
  if (param != nullptr) {
    if (param->input_shape.empty()) {
      return RET_OK;
    }
    for (const auto &elem : param->input_shape) {
      std::vector<int64_t> dims = elem.second;
      if (dims.empty()) {
        MS_LOG(ERROR) << "INPUT MISSING: input tensor dim is empty";
        return lite::RET_INPUT_PARAM_INVALID;
      }
      bool has_negative_dim = std::any_of(dims.begin(), dims.end(), [](int64_t dim) { return dim < 0; });
      if (has_negative_dim) {
        MS_LOG(ERROR) << "INPUT ILLEGAL: Unsupported dim < 0.";
        return lite::RET_INPUT_PARAM_INVALID;
      }
    }
  }
  return RET_OK;
}

int CheckInputFormat(const std::shared_ptr<ConverterPara> &param) {
  if (param != nullptr) {
    std::set valid_values = {NHWC, NCHW};
    if (std::find(valid_values.begin(), valid_values.end(), param->input_format) == valid_values.end()) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: input_format is not in {NHWC, NCHW}, but got " << param->input_format;
      return RET_INPUT_PARAM_INVALID;
    }
  }
  return RET_OK;
}

int CheckInputOutputDataType(const std::shared_ptr<ConverterPara> &param) {
  if (param != nullptr) {
    std::set input_valid_values = {DataType::kNumberTypeFloat32, DataType::kNumberTypeInt8,  DataType::kNumberTypeUInt8,
                                   DataType::kNumberTypeInt32,   DataType::kNumberTypeInt64, DataType::kTypeUnknown};
    if (std::find(input_valid_values.begin(), input_valid_values.end(), param->input_data_type) ==
        input_valid_values.end()) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: input_data_type is not in {kNumberTypeFloat32, kNumberTypeInt8, "
                    << "kNumberTypeUInt8, kTypeUnknown}, but got " << param->input_data_type;
      return RET_INPUT_PARAM_INVALID;
    }

    std::set output_valid_values = {DataType::kNumberTypeFloat32, DataType::kNumberTypeInt8, DataType::kNumberTypeUInt8,
                                    DataType::kTypeUnknown};
    if (std::find(output_valid_values.begin(), output_valid_values.end(), param->output_data_type) ==
        output_valid_values.end()) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: output_data_type is not in {kNumberTypeFloat32, kNumberTypeInt8, "
                    << "kNumberTypeUInt8, kTypeUnknown}, but got " << param->output_data_type;
      return RET_INPUT_PARAM_INVALID;
    }
  }
  return RET_OK;
}

int CheckSaveType(const std::shared_ptr<ConverterPara> &param) {
  if (param != nullptr) {
    std::set valid_values = {kMindIR, kMindIR_Lite};
    if (std::find(valid_values.begin(), valid_values.end(), param->save_type) == valid_values.end()) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: save_type is not in {kMindIR, kMindIR_Lite}, but got " << param->save_type;
      return RET_INPUT_PARAM_INVALID;
    }
  }
  return RET_OK;
}

int CheckEncrypt(const std::shared_ptr<ConverterPara> &param) {
  if (param != nullptr) {
    if (param->enable_encryption) {
      if (param->encrypt_key.empty()) {
        MS_LOG(ERROR) << "encryption param is true and encrypt_key param must be set. If you don't "
                         "need to use model encryption, please set encryption param to false.";
        return RET_INPUT_PARAM_INVALID;
      }
    }
  }
  return RET_OK;
}

int CheckTrainModel(const std::shared_ptr<ConverterPara> &param) {
  if (param != nullptr) {
    if (param->train_model) {
      if (param->fmk_type != converter::FmkType::kFmkTypeMs) {
        MS_LOG(ERROR) << "INPUT ILLEGAL: train model converter supporting only MINDIR format";
        return RET_INPUT_PARAM_INVALID;
      }
      if ((param->input_data_type != DataType::kNumberTypeFloat32) &&
          (param->input_data_type != DataType::kTypeUnknown)) {
        MS_LOG(ERROR) << "INPUT ILLEGAL: train model converter supporting only FP32 input tensors";
        return RET_INPUT_PARAM_INVALID;
      }
      if ((param->output_data_type != DataType::kNumberTypeFloat32) &&
          (param->output_data_type != DataType::kTypeUnknown)) {
        MS_LOG(ERROR) << "INPUT ILLEGAL: train model converter supporting only FP32 output tensors";
        return RET_INPUT_PARAM_INVALID;
      }
    }
  }
  return RET_OK;
}

int CheckDevice(const std::shared_ptr<ConverterPara> &param) {
  if (param != nullptr && !(param->device.empty())) {
    std::set valid_values = {"Ascend310", "Ascend310P", "Ascend"};
    if (std::find(valid_values.begin(), valid_values.end(), param->device) == valid_values.end()) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: device is not in {Ascend310, Ascend310P}, but got " << param->device;
      return RET_INPUT_PARAM_INVALID;
    }
  }
  return RET_OK;
}

int CheckValueParam(const std::shared_ptr<ConverterPara> &param, bool is_multi_model) {
  if (param == nullptr) {
    MS_LOG(ERROR) << "INPUT MISSING: param is nullptr.";
    return RET_INPUT_PARAM_INVALID;
  }

  auto ret = CheckFmkType(param);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Check value of fmk_type failed.";
    return RET_INPUT_PARAM_INVALID;
  }

  ret = CheckModelFile(param);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Check value of model_file failed.";
    return RET_INPUT_PARAM_INVALID;
  }

  if (!is_multi_model) {
    ret = CheckOutputFile(param);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Check value of output_file failed.";
      return RET_INPUT_PARAM_INVALID;
    }
  }

  ret = CheckInputShape(param);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Check value of input_shape failed.";
    return RET_INPUT_PARAM_INVALID;
  }

  ret = CheckInputFormat(param);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Check value of input_format failed.";
    return RET_INPUT_PARAM_INVALID;
  }

  ret = CheckInputOutputDataType(param);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Check value of input_data_type or output_data_type failed.";
    return RET_INPUT_PARAM_INVALID;
  }

  ret = CheckSaveType(param);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Check value of save_type failed.";
    return RET_INPUT_PARAM_INVALID;
  }

  ret = CheckEncrypt(param);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Check value of encrypt failed.";
    return RET_INPUT_PARAM_INVALID;
  }

  ret = CheckTrainModel(param);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Check value of train model failed.";
    return RET_INPUT_PARAM_INVALID;
  }

  ret = CheckDevice(param);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Check device failed.";
    return RET_INPUT_PARAM_INVALID;
  }

  return RET_OK;
}

int ConverterImpl::LoadPluginLib(const std::shared_ptr<ConverterPara> &param) {
  if (!param->plugins_path.empty()) {
    for (auto &path : param->plugins_path) {
      auto dl_loader = std::make_shared<DynamicLibraryLoader>();
      MS_CHECK_TRUE_RET(dl_loader != nullptr, RET_ERROR);
      auto status = dl_loader->Open(path);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "open dynamic library failed. " << path;
        return RET_ERROR;
      }
      dl_loaders.emplace_back(dl_loader);
    }
  }
  return RET_OK;
}

int ConverterImpl::Convert(const std::shared_ptr<ConverterPara> &param, void **model_data, size_t *data_size,
                           bool not_save) {
  if (param == nullptr) {
    MS_LOG(ERROR) << "Input param is nullptr";
    return RET_ERROR;
  }
  std::map<int, std::map<std::string, std::string>> model_param_infos;  // {model_index, {param_key:param_value}}
  auto ret = InitConfigParam(param, &model_param_infos);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init config file failed: " << ret << " " << GetErrorInfo(ret);
    return ret;
  }
  ret = StoreConverterParameters(param);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get converter parameter failed: " << ret << " " << GetErrorInfo(ret);
    return ret;
  }

  ret = LoadPluginLib(param);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Load plugin lib failed: " << ret << " " << GetErrorInfo(ret);
    return ret;
  }

  if (model_param_infos.empty()) {
    ret = CheckValueParam(param, false);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Converter input parameters check valid failed";
      return ret;
    }
    ret = HandleGraphCommon(param, model_data, data_size, not_save, false);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Handle graph failed: " << ret << " " << GetErrorInfo(ret);
      return ret;
    }
  } else {
    size_t model_index = 0;
    for (auto pair : model_param_infos) {
      auto convert_param = CreateConvertParam(pair.second);
      convert_param->microParam = param->microParam;
      ret = CheckValueParam(convert_param, true);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "For model" << pair.first << ", converter input parameters check valid failed";
        return ret;
      }
      if (model_index == model_param_infos.size() - 1) {
        convert_param->microParam.is_last_model = true;
      }
      ret = HandleGraphCommon(convert_param, model_data, data_size, not_save, true);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Handle graph failed: " << ret << " " << GetErrorInfo(ret);
        return ret;
      }
      model_index++;
    }
  }

  return RET_OK;
}

int ConverterImpl::HandleGraphCommon(const std::shared_ptr<ConverterPara> &param, void **model_data, size_t *data_size,
                                     bool not_save, bool is_multi_model) {
  auto graph = ConverterFuncGraph::Build(param);
  if (graph == nullptr) {
    MS_LOG(ERROR) << "Build func graph failed";
    return RET_ERROR;
  }

  int ret = ConverterFuncGraph::Optimize(param, graph);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Optimize func graph failed: " << ret << " " << GetErrorInfo(ret);
    return ret;
  }

  ret = SaveGraph(graph, param, model_data, data_size, not_save, is_multi_model);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Save graph failed: " << ret << " " << GetErrorInfo(ret);
    return ret;
  }

  return RET_OK;
}

int ConverterImpl::SaveGraph(FuncGraphPtr graph, const std::shared_ptr<ConverterPara> &param, void **model_data,
                             size_t *data_size, bool not_save, bool is_multi_model) {
  int status = RET_ERROR;
  if (param->save_type == kMindIR) {
    status = SaveMindIRModel(graph, param, model_data, data_size);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Save mindir model failed :" << status << " " << GetErrorInfo(status);
      return RET_ERROR;
    }
    return RET_OK;
  }

  auto meta_graph = ConverterToMetaGraph::Build(param, graph);
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "Convert to meta graph failed";
    return RET_ERROR;
  }

  if (!param->cpuOptionCfgParam.architecture.empty()) {
    std::string cpu_option = param->cpuOptionCfgParam.architecture + param->cpuOptionCfgParam.instruction;
    status = ConverterPackedNode(meta_graph, cpu_option);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "save pack info failed.";
      return status;
    }
  }

  meta_graph->version = Version();

  if (param->pre_infer) {
    status = PreInference(*meta_graph, param->train_model);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Preinference failed: " << status << " " << GetErrorInfo(status);
      delete meta_graph;
      return status;
    }
  }

  if (param->microParam.enable_micro) {
    if (!is_multi_model) {
      status = micro::Coder::MicroSourceCodeGeneration(*meta_graph, param->output_file, param->microParam.codegen_mode,
                                                       param->microParam.target, param->microParam.support_parallel,
                                                       param->microParam.debug_mode, true);
    } else {
      if (param->microParam.save_path.empty() || param->microParam.project_name.empty()) {
        MS_LOG(ERROR) << "Micro param for invalid: save_path or project name is needed";
        return RET_ERROR;
      }
      auto output_path = param->microParam.save_path + param->microParam.project_name;
      if (param->microParam.save_path[param->microParam.save_path.size() - 1] != '/' ||
          param->microParam.save_path[param->microParam.save_path.size() - 1] != '\\') {
        output_path = param->microParam.save_path + kSlash + param->microParam.project_name;
      }
      status = micro::Coder::MicroSourceCodeGeneration(*meta_graph, output_path, param->microParam.codegen_mode,
                                                       param->microParam.target, param->microParam.support_parallel,
                                                       param->microParam.debug_mode, param->microParam.is_last_model);
    }
  } else {
    status = ConverterToMetaGraph::Save(meta_graph, param, model_data, data_size, not_save);
  }
  delete meta_graph;
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Save failed:" << status << " " << GetErrorInfo(status);
    return status;
  }

  return RET_OK;
}

int ConverterImpl::SaveMindIRModel(FuncGraphPtr graph, const std::shared_ptr<ConverterPara> &param, void **model_data,
                                   size_t *data_size) {
  int status = RET_OK;
  if (param->pre_infer) {
    schema::MetaGraphT *meta_graph = nullptr;
    auto new_param = std::make_shared<ConverterPara>();
    new_param->fmk_type = converter::kFmkTypeMs;
    new_param->save_type = kMindIR;
    meta_graph = lite::ConverterToMetaGraph::Build(new_param, graph);
    if (meta_graph == nullptr) {
      MS_LOG(ERROR) << "FuncGraph convert to meta graph failed";
      return false;
    }
    status = PreInference(*meta_graph, param->train_model);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "PreInferenceMindIR failed: " << status << " " << GetErrorInfo(status);
      return status;
    }
  }
  status = ConverterFuncGraph::Save(param, graph, model_data, data_size);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Export to mindir failed: " << status << " " << GetErrorInfo(status);
    return RET_ERROR;
  }
  return RET_OK;
}

int RunConverter(const std::shared_ptr<ConverterPara> &param, void **model_data, size_t *data_size, bool not_save) {
  mindspore::mindspore_log_init();

  param->aclModelOptionCfgParam.offline = !not_save;
  int status = RET_OK;
  ConverterImpl converter_impl;
  status = converter_impl.Convert(param, model_data, data_size, not_save);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Convert model failed";
    NotSupportOp::GetInstance()->PrintOps();
    return status;
  }

  MS_LOG(INFO) << "CONVERT RESULT SUCCESS:" << status;
  std::cout << "CONVERT RESULT SUCCESS:" << status << std::endl;
  return status;
}
}  // namespace lite
}  // namespace mindspore
