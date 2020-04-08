/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "kernel/common_utils.h"
#include <unordered_map>
#include <map>
#include <iostream>
#include <fstream>
#include "nlohmann/json.hpp"
#include "session/anf_runtime_algorithm.h"
#include "common/utils.h"

namespace mindspore {
namespace kernel {
const std::unordered_map<std::string, TypeId> type_id_maps = {
  {"float", TypeId::kNumberTypeFloat32},   {"float16", TypeId::kNumberTypeFloat16},
  {"float32", TypeId::kNumberTypeFloat32}, {"float64", TypeId::kNumberTypeFloat64},
  {"int", TypeId::kNumberTypeInt},         {"int8", TypeId::kNumberTypeInt8},
  {"int16", TypeId::kNumberTypeInt16},     {"int32", TypeId::kNumberTypeInt32},
  {"int64", TypeId::kNumberTypeInt64},     {"uint", TypeId::kNumberTypeUInt},
  {"uint8", TypeId::kNumberTypeUInt8},     {"uint16", TypeId::kNumberTypeUInt16},
  {"uint32", TypeId::kNumberTypeUInt32},   {"uint64", TypeId::kNumberTypeUInt64},
  {"bool", TypeId::kNumberTypeBool},
};

const std::map<TypeId, std::string> type_id_str_map = {
  {TypeId::kNumberTypeFloat32, "float32"}, {TypeId::kNumberTypeFloat16, "float16"},
  {TypeId::kNumberTypeFloat, "float"},     {TypeId::kNumberTypeFloat64, "float64"},
  {TypeId::kNumberTypeInt, "int"},         {TypeId::kNumberTypeInt8, "int8"},
  {TypeId::kNumberTypeInt16, "int16"},     {TypeId::kNumberTypeInt32, "int32"},
  {TypeId::kNumberTypeInt64, "int64"},     {TypeId::kNumberTypeUInt, "uint"},
  {TypeId::kNumberTypeUInt8, "uint8"},     {TypeId::kNumberTypeUInt16, "uint16"},
  {TypeId::kNumberTypeUInt32, "uint32"},   {TypeId::kNumberTypeUInt64, "uint64"},
  {TypeId::kNumberTypeBool, "bool"},
};

const std::map<std::string, std::string> DATATYPE_STRING_MAP{
  {"Float32", "float32"}, {"Float16", "float16"}, {"Int8", "int8"},   {"Int16", "int16"},
  {"UInt16", "uint16"},   {"UInt8", "uint8"},     {"Int32", "int32"}, {"UInt32", "uint32"},
  {"Int64", "int64"},     {"UInt64", "uint64"},   {"Bool_", "bool"},  {"Float64", "double"},
};

const std::unordered_map<std::string, std::string> dtype_shortdtype_map_ = {
  {"float16", "f16"}, {"float32", "f32"}, {"float64", "f64"}, {"int8", "i8"},    {"int16", "i16"},  {"int32", "i32"},
  {"int64", "i64"},   {"uint8", "u8"},    {"uint16", "u16"},  {"uint32", "u32"}, {"uint64", "u64"}, {"bool", "bool"},
};

const std::unordered_map<std::string, size_t> dtype_nbyte_map = {
  {"float16", sizeof(float) / 2}, {"float32", sizeof(float)},  {"float64", sizeof(float) * 2},
  {"int8", sizeof(int) / 4},      {"int16", sizeof(int) / 2},  {"int32", sizeof(int)},
  {"int64", sizeof(int) * 2},     {"uint8", sizeof(int) / 4},  {"uint16", sizeof(int) / 2},
  {"uint32", sizeof(int)},        {"uint64", sizeof(int) * 2}, {"bool", sizeof(char)},
};

const std::unordered_map<std::string, FusionType> fusion_type_maps = {
  {"CONVLUTION", FusionType::CONVLUTION}, {"ELEMWISE", FusionType::ELEMWISE}, {"COMMREDUCE", FusionType::COMMREDUCE},
  {"SEGMENT", FusionType::SEGMENT},       {"OPAQUE", FusionType::OPAQUE},
};

bool IsAtomicNode(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto parameters_indexs = kernel_mod->GenParameters();
  if (parameters_indexs.empty()) {
    return false;
  }
  auto atomic_flag = false;
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  auto workspace_size_list = kernel_mod->GetWorkspaceSizeList();
  size_t workspace_num = kernel_mod->GetWorkspaceSizeList().size();
  if (input_num + workspace_num + output_num > parameters_indexs.size()) {
    size_t lossNum = (input_num + workspace_num + output_num) - parameters_indexs.size();
    for (size_t i = 0; i < lossNum; i++) {
      parameters_indexs.push_back(0);
    }
  }
  std::vector<size_t> clean_output_indexs;
  // in parameters data sort as input->workspace->output
  size_t index = 0;
  while (index < output_num) {
    if (parameters_indexs[input_num + workspace_num + index] == 1) {
      atomic_flag = true;
      clean_output_indexs.push_back(index);
    }
    index++;
  }
  if (atomic_flag) {
    AnfAlgo::SetNodeAttr(kAttrAutomicOutputIndexs, MakeValue(clean_output_indexs), kernel_node);
  }
  for (size_t i = 0; i < workspace_num; ++i) {
    if (parameters_indexs[input_num + i] == 1) {
      atomic_flag = true;
      AnfAlgo::SetNodeAttr(kAttrAutomicWorkspaceSize,
                           MakeValue(std::accumulate(workspace_size_list.begin(), workspace_size_list.end(), 0)),
                           kernel_node);
      break;
    }
  }
  return atomic_flag;
}

bool KernelMeta::ReadIndex(const std::string &bin_dir) {
  DIR *dir = opendir(bin_dir.c_str());
  if (dir == nullptr) {
    auto ret = mkdir(bin_dir.c_str(), S_IRWXG | S_IRWXU);
    if (ret != 0) {
      MS_LOG(INFO) << "kernel dir not exist[" << bin_dir << "].";
      return false;
    }
    dir = opendir(bin_dir.c_str());
  }

  struct dirent *entry;
  while ((entry = readdir(dir)) != nullptr) {
    string bin_dir_tmp = bin_dir;
    std::string cce_json = entry->d_name;
    if (cce_json.length() <= 5) {
      continue;
    }

    std::string suffix = cce_json.substr(cce_json.length() - 5);
    if (suffix != kJsonSuffix) {
      continue;
    }

    auto sp = cce_json.rfind('/');
    if (sp != std::string::npos) {
      continue;
    }

    sp = cce_json.rfind('.');
    if (sp == std::string::npos) {
      continue;
    }
    auto kernel_name = cce_json.substr(0, sp);
    (void)bin_dir_tmp.append("/");
    (void)bin_dir_tmp.append(cce_json);
    kernel_meta_map_[kernel_name] = bin_dir_tmp;
  }
  (void)closedir(dir);

  MS_LOG(INFO) << "Cache kernel initialized, kernel size[" << kernel_meta_map_.size() << "].";
  initialized_ = true;
  return true;
}

std::string KernelMeta::Search(const std::string &kernel_name) const {
  if (!initialized_) {
    return "";
  }

  auto iter = kernel_meta_map_.find(kernel_name);
  if (iter == kernel_meta_map_.end()) {
    return "";
  } else {
    return iter->second;
  }
}

bool KernelMeta::Insert(const std::string &kernel_name, const std::string &cce_json) {
  if (!initialized_) {
    return false;
  }
  kernel_meta_map_[kernel_name] = cce_json;
  return true;
}

bool CheckCache(const std::string &kernel_name) {
  // check cache.
  KernelMeta *bin_map = KernelMeta::GetInstance();
  if (bin_map == nullptr) {
    MS_LOG(DEBUG) << "kernel cache is invalid.";
    return false;
  }
  std::string cce_json = bin_map->Search(kernel_name);
  bool ret = (!cce_json.empty());
  if (ret) {
    MS_LOG(INFO) << "Kernel name:" << kernel_name << " has registed.";
  } else {
    MS_LOG(INFO) << "Kernel name:" << kernel_name << " will been registed.";
  }
  return ret;
}

KernelPackPtr SearchCache(const std::string &kernel_name, const std::string &processor) {
  // search cache.
  KernelMeta *bin_map = KernelMeta::GetInstance();
  if (bin_map == nullptr) {
    MS_LOG(DEBUG) << "kernel cache is invalid.";
    return nullptr;
  }

  std::string cce_json = bin_map->Search(kernel_name);
  if (!cce_json.empty()) {
    KernelPackPtr kernel_pack = std::make_shared<KernelPack>();
    // just a tmp solution.
    if (!kernel_pack->ReadFromJsonFile(cce_json, processor)) {
      MS_LOG(DEBUG) << "Read cache json and bin file failed[" << cce_json << "].";
      return nullptr;
    } else {
      return kernel_pack;
    }
  } else {
    MS_LOG(INFO) << "cache kernel not found[" << kernel_name << "].";
    return nullptr;
  }
}

KernelPackPtr InsertCache(const std::string &kernel_name, const std::string &processor) {
  MS_LOG(INFO) << "kernel name:" << kernel_name << ", processr:" << processor;
  std::string cce_json;
  if (processor == kProcessorAiCore || processor == kProcessorAiCpu) {
    cce_json = kCceKernelMeta;
  } else {
    cce_json = kGpuKernelMeta;
  }
  (void)cce_json.append(kernel_name).append(kJsonSuffix);
  KernelPackPtr kernel_pack = std::make_shared<KernelPack>();
  if (!kernel_pack->ReadFromJsonFile(cce_json, processor)) {
    MS_LOG(DEBUG) << "Read json and bin file failed[" << cce_json << "].";
    return nullptr;
  }

  KernelMeta *bin_map = KernelMeta::GetInstance();
  if (bin_map == nullptr) {
    MS_LOG(DEBUG) << "kernel cache is invalid.";
    return nullptr;
  }
  if (bin_map->Insert(kernel_name, cce_json)) {
    MS_LOG(INFO) << "Insert to cache success[" << cce_json << "], kernelname[" << kernel_name << "].";
  }
  return kernel_pack;
}

TypeId DtypeToTypeId(const std::string &dtypes) {
  auto iter = type_id_maps.find(dtypes);
  if (iter != type_id_maps.end()) {
    return iter->second;
  } else {
    MS_EXCEPTION(ArgumentError) << "Illegal input device dtype:" << dtypes;
  }
}

std::string Dtype2String(const std::string &dtypes) {
  auto iter = DATATYPE_STRING_MAP.find(dtypes);
  if (iter == DATATYPE_STRING_MAP.end()) {
    MS_EXCEPTION(ArgumentError) << "Illegal input dtype:" << dtypes;
  }
  return iter->second;
}

std::string TypeId2String(TypeId type_id) {
  auto iter = type_id_str_map.find(type_id);
  if (iter == type_id_str_map.end()) {
    MS_EXCEPTION(ArgumentError) << "Illegal input dtype." << TypeIdLabel(type_id);
  }
  return iter->second;
}

std::string Dtype2ShortType(const std::string &dtypes) {
  auto iter = dtype_shortdtype_map_.find(dtypes);
  if (iter != dtype_shortdtype_map_.end()) {
    return iter->second;
  } else {
    MS_EXCEPTION(ArgumentError) << "Illegal input dtype:" << dtypes;
  }
}

size_t GetDtypeNbyte(const std::string &dtypes) {
  auto iter = dtype_nbyte_map.find(dtypes);
  if (iter != dtype_nbyte_map.end()) {
    return iter->second;
  } else {
    MS_EXCEPTION(ArgumentError) << "Illegal input dtype:" << dtypes;
  }
}

bool SetInputKernelBuilderInfo(const std::vector<std::shared_ptr<OpIOInfo>> &inputs, size_t real_input_num,
                               size_t builder_idex, const std::vector<int> &dyn_input_sizes,
                               const std::shared_ptr<KernelBuildInfo::KernelBuildInfoBuilder> &builder) {
  MS_EXCEPTION_IF_NULL(builder);

  std::vector<TypeId> inputs_device_type;
  std::vector<std::string> inputs_format;
  size_t dyn_input_idx = 0;
  size_t kernel_info_index = 0;
  MS_EXCEPTION_IF_NULL(inputs[0]);
  size_t kernel_info_cnt = inputs[0]->dtypes().size();

  for (const auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    std::string param_type = input->param_type();
    std::vector<std::string> dtypes = input->dtypes();
    std::vector<std::string> formats = input->formats();
    if (dtypes.size() != kernel_info_cnt || formats.size() != kernel_info_cnt) {
      MS_LOG(DEBUG) << "Set input kernel builder info, dtyps size != formats size.";
      return false;
    }

    if (param_type == "dynamic") {
      if (dyn_input_sizes.empty()) {
        MS_LOG(DEBUG) << "Set input kernel builder info, dyn_input_sizes's size is 0 when param_type is dynamic";
        return false;
      }

      for (int t = 0; t < dyn_input_sizes[dyn_input_idx]; t++) {
        kernel_info_index++;
        auto type_id = DtypeToTypeId(dtypes[builder_idex]);
        inputs_device_type.push_back(type_id);
        inputs_format.push_back(formats[builder_idex]);
      }
      dyn_input_idx++;
    } else if (param_type == "required") {
      kernel_info_index++;
      auto type_id = DtypeToTypeId(dtypes[builder_idex]);
      inputs_device_type.push_back(type_id);
      inputs_format.push_back(formats[builder_idex]);
    } else {
      if (kernel_info_index < real_input_num) {
        MS_LOG(INFO) << "Set input kernel builder info, input type is optional, input index is :" << kernel_info_index;
        kernel_info_index++;
        auto type_id = DtypeToTypeId(dtypes[builder_idex]);
        inputs_device_type.push_back(type_id);
        inputs_format.push_back(formats[builder_idex]);
      }
    }
  }

  builder->SetInputsDeviceType(inputs_device_type);
  builder->SetInputsFormat(inputs_format);
  return true;
}

bool SetOutputKernelBuilderInfo(const std::vector<std::shared_ptr<OpIOInfo>> &outputs, size_t builder_idex,
                                const size_t &real_output_num,
                                const std::shared_ptr<KernelBuildInfo::KernelBuildInfoBuilder> &builder) {
  // not now but in the next we need to support dynamic output case
  MS_EXCEPTION_IF_NULL(builder);

  size_t output_idx = 0;
  std::vector<TypeId> outputs_device_type;
  std::vector<std::string> outputs_format;
  MS_EXCEPTION_IF_NULL(outputs[0]);
  size_t kernel_info_cnt = outputs[0]->dtypes().size();

  for (const auto &output : outputs) {
    MS_EXCEPTION_IF_NULL(output);
    if (output_idx >= real_output_num) {
      MS_LOG(DEBUG) << "real_output_num:" << real_output_num << ", output_idx:" << output_idx << " is out of limit!";
      continue;
    }
    size_t output_num = 0;
    if (output->param_type() == "dynamic") {
      if (outputs.size() > 1) {
        MS_EXCEPTION(ArgumentError) << "Dynamic output is unsupported multi output!";
      }
      output_num = real_output_num;
    } else if (output->param_type() == "required") {
      output_num = 1;
    } else {
      if (output_idx < real_output_num) {
        MS_LOG(INFO) << "Set output kernel builder info, output type is optional, output index is :" << output_idx;
        output_num = 1;
      }
    }

    for (size_t i = 0; i < output_num; i++) {
      std::vector<std::string> dtypes = output->dtypes();
      std::vector<std::string> formats = output->formats();
      if (dtypes.size() != kernel_info_cnt || formats.size() != kernel_info_cnt) {
        MS_LOG(DEBUG) << "Set output kernel builder info, dtyps size != formats size.";
        return false;
      }
      auto type_id = DtypeToTypeId(dtypes[builder_idex]);
      outputs_device_type.push_back(type_id);
      outputs_format.push_back(formats[builder_idex]);
      output_idx++;
    }
  }

  builder->SetOutputsFormat(outputs_format);
  builder->SetOutputsDeviceType(outputs_device_type);
  return true;
}

void SetKernelBuildInfo(const std::shared_ptr<KernelBuildInfo::KernelBuildInfoBuilder> &builder, Processor processor,
                        const std::shared_ptr<const OpInfo> &op_info_ptr) {
  MS_EXCEPTION_IF_NULL(builder);
  MS_EXCEPTION_IF_NULL(op_info_ptr);

  auto imply_type = op_info_ptr->imply_type();
  builder->SetProcessor(processor);
  std::string fusion_type = op_info_ptr->fusion_type();
  auto iter = fusion_type_maps.find(fusion_type);
  if (iter != fusion_type_maps.end()) {
    builder->SetFusionType(iter->second);
  } else {
    if (imply_type == kAKG) {
      MS_EXCEPTION(NotExistsError) << "Illegal fusion type from dsl register:" << fusion_type;
    }
  }

  if (imply_type == kAKG) {
    builder->SetKernelType(AUTO_DIFF_KERNEL);
  } else {
    builder->SetKernelType(TBE_KERNEL);
  }
}

bool ParseMetadata(const CNodePtr &kernel_node, const std::shared_ptr<const OpInfo> &op_info_ptr, Processor processor,
                   std::vector<std::shared_ptr<KernelBuildInfo>> *const kernel_info_list) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  size_t real_input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  size_t real_output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  std::vector<std::shared_ptr<OpIOInfo>> inputs = op_info_ptr->inputs_ptr();
  std::vector<std::shared_ptr<OpIOInfo>> outputs = op_info_ptr->outputs_ptr();
  std::vector<int> dyn_input_sizes;
  auto primitive = AnfAlgo::GetCNodePrimitive(kernel_node);
  MS_EXCEPTION_IF_NULL(primitive);
  if (primitive->GetAttr("dyn_input_sizes") != nullptr) {
    dyn_input_sizes = GetValue<std::vector<int>>(primitive->GetAttr("dyn_input_sizes"));
  }
  if (inputs.size() > 0) {
    MS_EXCEPTION_IF_NULL(inputs[0]);
    size_t kernel_info_cnt = inputs[0]->dtypes().size();
    for (size_t j = 0; j < kernel_info_cnt; j++) {
      auto builder = std::make_shared<KernelBuildInfo::KernelBuildInfoBuilder>();
      MS_EXCEPTION_IF_NULL(builder);
      SetKernelBuildInfo(builder, processor, op_info_ptr);

      if (!SetInputKernelBuilderInfo(inputs, real_input_num, j, dyn_input_sizes, builder)) {
        MS_LOG(DEBUG) << "Parse kernel metadata, set inputs kernel builder info failed.";
        return false;
      }

      if (outputs.size() > 0) {
        if (!SetOutputKernelBuilderInfo(outputs, j, real_output_num, builder)) {
          MS_LOG(DEBUG) << "Parse kernel metadata, set outputs kernel builder info failed.";
          return false;
        }
      }

      kernel_info_list->push_back(builder->Build());
    }
  } else if (outputs.size() > 0) {
    MS_EXCEPTION_IF_NULL(outputs[0]);
    size_t kernel_info_cnt = outputs[0]->dtypes().size();
    for (size_t j = 0; j < kernel_info_cnt; j++) {
      auto builder = std::make_shared<KernelBuildInfo::KernelBuildInfoBuilder>();
      MS_EXCEPTION_IF_NULL(builder);
      SetKernelBuildInfo(builder, processor, op_info_ptr);

      if (!SetOutputKernelBuilderInfo(outputs, j, real_output_num, builder)) {
        MS_LOG(DEBUG) << "Parse kernel metadata, set outputs kernel builder info failed.";
        return false;
      }

      kernel_info_list->push_back(builder->Build());
    }
  }
  return true;
}

void SaveJsonInfo(const std::string &json_name, const std::string &info) {
  char real_path[PATH_MAX] = {0};
  std::string path = kCceKernelMeta + json_name + kInfoSuffix;
  if (path.size() > PATH_MAX) {
    MS_LOG(DEBUG) << "file path " << path << " is too long.";
    return;
  }
  std::ofstream filewrite;
  filewrite.open(path);
  if (!filewrite.is_open()) {
    return;
  }
  filewrite << info << std::endl;
  filewrite.close();
  if (nullptr == realpath(path.c_str(), real_path)) {
    MS_LOG(DEBUG) << "dir " << path << " does not exit.";
    return;
  }
  MS_LOG(INFO) << "real path is :" << real_path;
  if (chmod(real_path, S_IRUSR) == -1) {
    MS_LOG(DEBUG) << "modify file:" << real_path << " to read only fail.";
  }
}

std::string GetProcessor(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::string device;
  switch (AnfAlgo::GetProcessor(anf_node)) {
    case Processor::AICORE:
      device = kProcessorAiCore;
      break;

    case Processor::AICPU:
      device = kProcessorAiCpu;
      break;

    case Processor::CUDA:
      device = kProcessorCuda;
      break;

    default:
      MS_LOG(DEBUG) << "Unknown processor type.";
      break;
  }
  return device;
}
}  // namespace kernel
}  // namespace mindspore
