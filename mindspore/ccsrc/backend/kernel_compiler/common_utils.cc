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

#include "backend/kernel_compiler/common_utils.h"
#include <unordered_map>
#include <map>
#include <iostream>
#include <utility>
#include <fstream>
#include <algorithm>
#include <thread>
#include "nlohmann/json.hpp"
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/ms_utils.h"
#include "ir/manager.h"
#include "ir/meta_tensor.h"
#include "ir/func_graph.h"
#include "frontend/operator/ops.h"
#include "ir/graph_utils.h"

namespace mindspore {
namespace kernel {
constexpr char kAxis[] = "axis";
constexpr char kTypeInt32[] = "Int32";
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

void KernelMeta::Initialize(int pid) {
  if (pid == -1) {
    kernel_meta_path_ = std::string(kGpuKernelMeta) + "_" + std::to_string(getpid()) + "/";
  } else {
    kernel_meta_path_ = std::string(kGpuKernelMeta) + "_" + std::to_string(pid) + "/";
  }
  // remove old kernel cache
  RemoveKernelCache();

#if defined(_WIN32) || defined(_WIN64)
  auto ret = mkdir(kernel_meta_path_.c_str());
#else
  auto ret = mkdir(kernel_meta_path_.c_str(), S_IRWXG | S_IRWXU);
#endif
  if (ret != 0) {
    MS_LOG(INFO) << "kernel dir [" << kernel_meta_path_ << "], will be created later";
  }
  initialized_ = true;
}

void KernelMeta::RemoveKernelCache() {
  DIR *dir = opendir(kernel_meta_path_.c_str());
  if (dir == nullptr) {
    return;
  }
  struct dirent *entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string kernel_file = entry->d_name;
    std::string kernel_file_realpath = kernel_meta_path_ + kernel_file;
    (void)remove(kernel_file_realpath.c_str());
  }
  (void)closedir(dir);
  (void)rmdir(kernel_meta_path_.c_str());
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

bool KernelMeta::Insert(const std::string &kernel_name, const std::string &kernel_json) {
  if (!initialized_) {
    return false;
  }
  kernel_meta_map_[kernel_name] = kernel_json;
  return true;
}

bool CheckCache(const std::string &kernel_name) {
  // check cache.
  KernelMeta *bin_map = KernelMeta::GetInstance();
  if (bin_map == nullptr) {
    MS_LOG(DEBUG) << "kernel cache is invalid.";
    return false;
  }
  std::string kernel_json = bin_map->Search(kernel_name);
  bool ret = (!kernel_json.empty());
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

  std::string kernel_json = bin_map->Search(kernel_name);
  if (!kernel_json.empty()) {
    KernelPackPtr kernel_pack = std::make_shared<KernelPack>();
    // just a tmp solution.
    if (!kernel_pack->ReadFromJsonFile(kernel_json, processor)) {
      MS_LOG(DEBUG) << "Read cache json and bin file failed[" << kernel_json << "].";
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
  KernelMeta *bin_map = KernelMeta::GetInstance();
  std::string kernel_json;
  if (processor == kProcessorAiCore || processor == kProcessorAiCpu) {
    kernel_json = kCceKernelMeta;
  } else {
    kernel_json = bin_map->GetKernelMetaPath();
  }
  (void)kernel_json.append(kernel_name).append(kJsonSuffix);
  KernelPackPtr kernel_pack = std::make_shared<KernelPack>();
  if (!kernel_pack->ReadFromJsonFile(kernel_json, processor)) {
    MS_LOG(DEBUG) << "Read json and bin file failed[" << kernel_json << "].";
    return nullptr;
  }

  if (bin_map == nullptr) {
    MS_LOG(DEBUG) << "kernel cache is invalid.";
    return nullptr;
  }
  if (bin_map->Insert(kernel_name, kernel_json)) {
    MS_LOG(INFO) << "Insert to cache success[" << kernel_json << "], kernelname[" << kernel_name << "].";
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

std::string TypeId2String(TypeId type_id) {
  auto iter = type_id_str_map.find(type_id);
  if (iter == type_id_str_map.end()) {
    return std::string(TypeIdLabel(type_id));
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
        MS_LOG(DEBUG) << "Set output kernel builder info, output type is optional, output index is :" << output_idx;
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
    builder->SetKernelType(AKG_KERNEL);
  } else if (imply_type == kAICPU) {
    builder->SetKernelType(AICPU_KERNEL);
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
  } else {
    if (processor == AICPU) {
      auto builder = std::make_shared<KernelBuildInfo::KernelBuildInfoBuilder>();
      MS_EXCEPTION_IF_NULL(builder);
      SetKernelBuildInfo(builder, processor, op_info_ptr);
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
#if defined(_WIN32) || defined(_WIN64)
  if (nullptr == _fullpath(real_path, path.c_str(), PATH_MAX)) {
    MS_LOG(DEBUG) << "dir " << path << " does not exit.";
    return;
  }
#else
  if (nullptr == realpath(path.c_str(), real_path)) {
    MS_LOG(DEBUG) << "dir " << path << " does not exit.";
    return;
  }
#endif
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

bool IsSameShape(const std::vector<size_t> &shape_a, const std::vector<size_t> &shape_b) {
  if (shape_a.size() != shape_b.size()) {
    return false;
  }
  for (size_t i = 0; i < shape_a.size(); ++i) {
    if (shape_a[i] != shape_b[i]) {
      return false;
    }
  }
  return true;
}

int Sign(float x) {
  if (x > 0) {
    return 1;
  }
  if (x < 0) {
    return -1;
  }
  return 0;
}

namespace {
struct BucketSparseGradient {
  float *value_;
  int *indices_;
  int *global_indices_;
  size_t indices_size_;
};

struct MultiThreadReduceSparseGradientParam {
  SparseGradient *input_grad_{nullptr};
  SparseGradient *workspace_grad_{nullptr};
  SparseGradient *output_grad_{nullptr};
  size_t max_index_{0};
  size_t value_stride_{0};
  size_t thread_num_{0};
  bool use_sort_reduce_{false};
};

void CalculateEachBucketSize(const std::shared_ptr<SparseGradient> &sparse_grad, size_t max_index,
                             std::vector<size_t> *each_bucket_size) {
  MS_LOG(DEBUG) << "Start";
  MS_EXCEPTION_IF_NULL(sparse_grad);
  MS_EXCEPTION_IF_NULL(sparse_grad->indices_);
  MS_EXCEPTION_IF_NULL(each_bucket_size);
  size_t bucket_num = each_bucket_size->size();
  for (size_t i = 0; i < sparse_grad->indices_size_; ++i) {
    int index = sparse_grad->indices_[i];
    if (index >= 0 && IntToSize(index) < max_index) {
      auto bucket_id = index % bucket_num;
      each_bucket_size->at(bucket_id)++;
    }
  }
  MS_LOG(DEBUG) << "End";
}

void SplitAndCalculateSegmentBucketSize(const MultiThreadReduceSparseGradientParam &param,
                                        std::vector<std::shared_ptr<SparseGradient>> *segments_ptr,
                                        std::vector<std::shared_ptr<std::vector<size_t>>> *segment_bucket_sizes_ptr) {
  MS_EXCEPTION_IF_NULL(param.input_grad_);
  MS_EXCEPTION_IF_NULL(segment_bucket_sizes_ptr);
  MS_EXCEPTION_IF_NULL(segments_ptr);
  auto &segments = *segments_ptr;
  auto &segment_bucket_sizes = *segment_bucket_sizes_ptr;
  auto input_grad = param.input_grad_;
  if (param.thread_num_ < 1) {
    MS_EXCEPTION(ArgumentError) << "Input param thread num must > 0!";
  }
  size_t thread_indices_size = input_grad->indices_size_ / param.thread_num_;
  size_t left_indices_size = input_grad->indices_size_ % param.thread_num_;
  std::vector<std::thread> threads;
  threads.reserve(param.thread_num_);
  segments.reserve(param.thread_num_);

  size_t current_indices_offset = 0;
  for (size_t i = 0; i < param.thread_num_; ++i) {
    segment_bucket_sizes.emplace_back(std::make_shared<std::vector<size_t>>(param.thread_num_, 0));
    size_t indices_size = thread_indices_size;
    if (i < left_indices_size) {
      indices_size += 1;
    }
    segments.emplace_back(std::make_shared<SparseGradient>());
    segments[i]->value_ = input_grad->value_ + current_indices_offset * param.value_stride_;
    segments[i]->indices_ = input_grad->indices_ + current_indices_offset;
    segments[i]->indices_size_ = indices_size;
    threads.emplace_back(
      std::thread(CalculateEachBucketSize, segments[i], param.max_index_, segment_bucket_sizes[i].get()));
    current_indices_offset += indices_size;
  }

  for (size_t i = 0; i < param.thread_num_; ++i) {
    threads[i].join();
  }
}

void CopySegmentIndicesToBucket(const MultiThreadReduceSparseGradientParam &param,
                                const std::shared_ptr<SparseGradient> &segment, size_t bucket_offset,
                                const std::vector<std::shared_ptr<BucketSparseGradient>> &buckets) {
  MS_LOG(DEBUG) << "Start";
  MS_EXCEPTION_IF_NULL(segment);
  MS_EXCEPTION_IF_NULL(segment->indices_);
  std::vector<size_t> bucket_data_num(param.thread_num_, 0);
  for (size_t i = 0; i < segment->indices_size_; ++i) {
    int index = segment->indices_[i];
    if (index >= 0 && IntToSize(index) < param.max_index_) {
      auto bucket_id = index % param.thread_num_;
      auto bucket_index = bucket_data_num[bucket_id];
      buckets[bucket_id]->indices_[bucket_index] = index;
      buckets[bucket_id]->global_indices_[bucket_index] = bucket_offset + i;
      bucket_data_num[bucket_id]++;
    }
  }
  MS_LOG(DEBUG) << "End";
}

void GatherSegmentIndicesToOutputBucket(const MultiThreadReduceSparseGradientParam &param,
                                        const std::vector<std::shared_ptr<SparseGradient>> &segments,
                                        const std::vector<std::shared_ptr<std::vector<size_t>>> &segment_bucket_sizes,
                                        std::vector<std::shared_ptr<BucketSparseGradient>> *buckets_ptr) {
  MS_EXCEPTION_IF_NULL(param.output_grad_);
  MS_EXCEPTION_IF_NULL(param.output_grad_->value_);
  MS_EXCEPTION_IF_NULL(param.output_grad_->indices_);
  MS_EXCEPTION_IF_NULL(buckets_ptr);
  auto &buckets = *buckets_ptr;
  size_t thread_num = param.thread_num_;
  if (thread_num != segment_bucket_sizes.size()) {
    MS_EXCEPTION(ArgumentError) << "Input param thread num not equal to segment size!";
  }
  std::vector<size_t> bucket_data_size(thread_num, 0);
  for (size_t i = 0; i < thread_num; ++i) {
    for (size_t j = 0; j < thread_num; ++j) {
      bucket_data_size[j] += segment_bucket_sizes[i]->at(j);
    }
  }
  size_t current_indices_offset = 0;
  for (size_t i = 0; i < thread_num; ++i) {
    buckets.emplace_back(std::make_shared<BucketSparseGradient>());
    buckets[i]->value_ = param.output_grad_->value_ + current_indices_offset * param.value_stride_;
    buckets[i]->indices_ = param.output_grad_->indices_ + current_indices_offset;
    buckets[i]->global_indices_ = param.workspace_grad_->indices_ + current_indices_offset;
    buckets[i]->indices_size_ = bucket_data_size[i];
    current_indices_offset += bucket_data_size[i];
  }
  std::vector<size_t> tmp_bucket_data_size(thread_num, 0);
  std::vector<std::vector<std::shared_ptr<BucketSparseGradient>>> each_thread_buckets;
  for (size_t i = 0; i < thread_num; ++i) {
    std::vector<std::shared_ptr<BucketSparseGradient>> thread_buckets;
    for (size_t j = 0; j < thread_num; ++j) {
      thread_buckets.emplace_back(std::make_shared<BucketSparseGradient>());
      thread_buckets[j]->indices_ = buckets[j]->indices_ + tmp_bucket_data_size[j];
      thread_buckets[j]->global_indices_ = buckets[j]->global_indices_ + tmp_bucket_data_size[j];
      thread_buckets[j]->value_ = buckets[j]->value_ + tmp_bucket_data_size[j] * param.value_stride_;
      thread_buckets[j]->indices_size_ = segment_bucket_sizes[i]->at(j);
      tmp_bucket_data_size[j] += segment_bucket_sizes[i]->at(j);
    }
    each_thread_buckets.emplace_back(thread_buckets);
  }
  std::vector<std::thread> threads;
  threads.reserve(thread_num);
  current_indices_offset = 0;
  for (size_t i = 0; i < thread_num; ++i) {
    threads.emplace_back(
      std::thread(CopySegmentIndicesToBucket, param, segments[i], current_indices_offset, each_thread_buckets[i]));
    current_indices_offset += segments[i]->indices_size_;
  }
  for (size_t i = 0; i < thread_num; ++i) {
    threads[i].join();
  }
}

void SortAndReduceBucketSparseGradient(const MultiThreadReduceSparseGradientParam &param,
                                       const std::shared_ptr<BucketSparseGradient> &bucket,
                                       const std::shared_ptr<SparseGradient> &reduced_bucket) {
  MS_LOG(DEBUG) << "Start";
  MS_EXCEPTION_IF_NULL(bucket);
  MS_EXCEPTION_IF_NULL(bucket->value_);
  MS_EXCEPTION_IF_NULL(bucket->indices_);
  MS_EXCEPTION_IF_NULL(reduced_bucket);
  MS_EXCEPTION_IF_NULL(reduced_bucket->value_);
  MS_EXCEPTION_IF_NULL(reduced_bucket->indices_);
  std::vector<std::pair<int, int>> sorted_indices;
  sorted_indices.reserve(bucket->indices_size_);
  for (size_t i = 0; i < bucket->indices_size_; ++i) {
    int index = bucket->indices_[i];
    int global_index = bucket->global_indices_[i];
    sorted_indices.emplace_back(std::pair<int, int>(index, global_index));
  }
  std::sort(sorted_indices.begin(), sorted_indices.end());

  float *global_value = param.input_grad_->value_;
  size_t unique_indices_size = 0;
  size_t max_length = reduced_bucket->indices_size_ * param.value_stride_;
  int last_index{0};
  size_t value_offset{0};
  for (size_t i = 0; i < sorted_indices.size(); ++i) {
    int index = sorted_indices[i].first;
    int global_index = sorted_indices[i].second;
    int global_value_offset = global_index * param.value_stride_;
    if (i == 0 || index != last_index) {
      if (i != 0) {
        unique_indices_size++;
      }
      reduced_bucket->indices_[unique_indices_size] = index;
      value_offset = unique_indices_size * param.value_stride_;
      auto ret_code = memcpy_s(reduced_bucket->value_ + value_offset, (max_length - value_offset) * sizeof(float),
                               global_value + global_value_offset, param.value_stride_ * sizeof(float));
      if (ret_code != EOK) {
        MS_LOG(EXCEPTION) << "Failed to copy data!";
      }
    } else {
      for (size_t j = 0; j < param.value_stride_; ++j) {
        reduced_bucket->value_[value_offset + j] += global_value[global_value_offset + j];
      }
    }
    last_index = index;
  }
  reduced_bucket->indices_size_ = unique_indices_size;
  MS_LOG(DEBUG) << "End";
}

void ReduceBucketSparseGradient(const MultiThreadReduceSparseGradientParam &param,
                                const std::shared_ptr<BucketSparseGradient> &bucket,
                                const std::shared_ptr<SparseGradient> &reduced_bucket) {
  MS_LOG(DEBUG) << "Start";
  MS_EXCEPTION_IF_NULL(bucket);
  MS_EXCEPTION_IF_NULL(bucket->value_);
  MS_EXCEPTION_IF_NULL(bucket->indices_);
  MS_EXCEPTION_IF_NULL(reduced_bucket);
  MS_EXCEPTION_IF_NULL(reduced_bucket->value_);
  MS_EXCEPTION_IF_NULL(reduced_bucket->indices_);

  float *global_value = param.input_grad_->value_;
  std::unordered_map<int, size_t> index_map;
  size_t unique_indices_size = 0;
  size_t max_length = reduced_bucket->indices_size_ * param.value_stride_;
  for (size_t i = 0; i < bucket->indices_size_; ++i) {
    int index = bucket->indices_[i];
    int global_index = bucket->global_indices_[i];
    auto iter = index_map.find(index);
    if (iter == index_map.end()) {
      reduced_bucket->indices_[unique_indices_size] = index;
      size_t start_index = unique_indices_size * param.value_stride_;
      index_map[index] = start_index;
      auto ret_code = memcpy_s(reduced_bucket->value_ + start_index, (max_length - start_index) * sizeof(float),
                               global_value + global_index * param.value_stride_, param.value_stride_ * sizeof(float));
      if (ret_code != EOK) {
        MS_LOG(EXCEPTION) << "Failed to copy data!";
      }
      unique_indices_size++;
    } else {
      size_t start_index = iter->second;
      size_t end_index = start_index + param.value_stride_;
      for (size_t j = start_index, k = global_index * param.value_stride_; j < end_index; ++j, ++k) {
        reduced_bucket->value_[j] += global_value[k];
      }
    }
  }
  reduced_bucket->indices_size_ = unique_indices_size;
  MS_LOG(DEBUG) << "End";
}

void ReduceBucketSparseGradientToWorkspace(const MultiThreadReduceSparseGradientParam &param,
                                           const std::vector<std::shared_ptr<BucketSparseGradient>> &buckets,
                                           std::vector<std::shared_ptr<SparseGradient>> *reduced_buckets_ptr) {
  MS_EXCEPTION_IF_NULL(param.workspace_grad_);
  MS_EXCEPTION_IF_NULL(param.workspace_grad_->value_);
  MS_EXCEPTION_IF_NULL(param.workspace_grad_->indices_);
  MS_EXCEPTION_IF_NULL(reduced_buckets_ptr);
  auto &reduced_buckets = *reduced_buckets_ptr;
  size_t thread_num = buckets.size();
  std::vector<std::thread> threads;
  threads.reserve(thread_num);

  size_t current_indices_offset = 0;
  for (size_t i = 0; i < thread_num; ++i) {
    reduced_buckets.emplace_back(std::make_shared<SparseGradient>());
    reduced_buckets[i]->value_ = param.workspace_grad_->value_ + current_indices_offset * param.value_stride_;
    reduced_buckets[i]->indices_ = param.workspace_grad_->indices_ + current_indices_offset;
    reduced_buckets[i]->indices_size_ = buckets[i]->indices_size_;
    if (param.use_sort_reduce_) {
      threads.emplace_back(std::thread(SortAndReduceBucketSparseGradient, param, buckets[i], reduced_buckets[i]));
    } else {
      threads.emplace_back(std::thread(ReduceBucketSparseGradient, param, buckets[i], reduced_buckets[i]));
    }
    current_indices_offset += buckets[i]->indices_size_;
  }
  for (size_t i = 0; i < thread_num; ++i) {
    threads[i].join();
  }
}

void MergeReduceSparseGradient(const MultiThreadReduceSparseGradientParam &param,
                               const std::vector<std::shared_ptr<SparseGradient>> &reduced_buckets) {
  MS_EXCEPTION_IF_NULL(param.output_grad_);
  auto output_grad = param.output_grad_;
  MS_EXCEPTION_IF_NULL(output_grad->value_);
  MS_EXCEPTION_IF_NULL(output_grad->indices_);
  size_t stride_data_size = param.value_stride_ * sizeof(float);
  size_t unique_indices_size = 0;
  for (size_t i = 0; i < reduced_buckets.size(); ++i) {
    auto &bucket = reduced_buckets[i];
    MS_EXCEPTION_IF_NULL(bucket);
    if (bucket->indices_size_ == 0) {
      continue;
    }
    auto ret_code = memcpy_s(output_grad->value_ + unique_indices_size * param.value_stride_,
                             (output_grad->indices_size_ - unique_indices_size) * stride_data_size, bucket->value_,
                             bucket->indices_size_ * stride_data_size);
    if (ret_code != EOK) {
      MS_LOG(EXCEPTION) << "Failed to copy data!";
    }
    ret_code = memcpy_s(output_grad->indices_ + unique_indices_size,
                        (output_grad->indices_size_ - unique_indices_size) * sizeof(int), bucket->indices_,
                        bucket->indices_size_ * sizeof(int));
    if (ret_code != EOK) {
      MS_LOG(EXCEPTION) << "Failed to copy data!";
    }
    unique_indices_size += bucket->indices_size_;
  }
  output_grad->indices_size_ = unique_indices_size;
}
}  // namespace

void BucketReduceSparseGradient(const ReduceSparseGradientParam &param) {
  MS_LOG(DEBUG) << "Start";
  MS_EXCEPTION_IF_NULL(param.input_grad_);
  size_t thread_num = 23;
  if (param.input_grad_->indices_size_ < thread_num) {
    thread_num = param.input_grad_->indices_size_;
  }
  MultiThreadReduceSparseGradientParam multi_thread_param({param.input_grad_, param.workspace_grad_, param.output_grad_,
                                                           param.max_index_, param.value_stride_, thread_num,
                                                           param.use_sort_reduce_});
  std::vector<std::shared_ptr<SparseGradient>> segments;
  std::vector<std::shared_ptr<std::vector<size_t>>> segment_bucket_sizes;
  SplitAndCalculateSegmentBucketSize(multi_thread_param, &segments, &segment_bucket_sizes);

  std::vector<std::shared_ptr<BucketSparseGradient>> buckets;
  GatherSegmentIndicesToOutputBucket(multi_thread_param, segments, segment_bucket_sizes, &buckets);

  std::vector<std::shared_ptr<SparseGradient>> reduced_buckets;
  ReduceBucketSparseGradientToWorkspace(multi_thread_param, buckets, &reduced_buckets);

  MergeReduceSparseGradient(multi_thread_param, reduced_buckets);
  MS_LOG(DEBUG) << "End";
}

std::pair<AnfNodePtr, size_t> GetKernelInput(const AnfNodePtr &anf_node, size_t index) {
  MS_EXCEPTION_IF_NULL(anf_node);

  if (index >= AnfAlgo::GetInputTensorNum(anf_node)) {
    MS_EXCEPTION(ArgumentError) << "Index is out of the size of anf_node inputs.";
  }

  auto cnode = anf_node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return AnfAlgo::VisitKernel(anf_node, 0);
  } else {
    return AnfAlgo::VisitKernel(anf_node->cast<CNodePtr>()->input(index + 1), 0);
  }
}

std::vector<std::pair<AnfNodePtr, std::pair<size_t, size_t>>> GetInputIndex(const std::vector<AnfNodePtr> &node_list,
                                                                            const std::vector<AnfNodePtr> &input_list) {
  std::vector<std::pair<AnfNodePtr, std::pair<size_t, size_t>>> input_index;
  for (size_t i = 0; i < input_list.size(); ++i) {
    auto const &input = input_list[i];
    MS_EXCEPTION_IF_NULL(input);
    bool found = false;
    // using NodeUsersMap = std::unordered_map<AnfNodePtr, std::set<std::pair<AnfNodePtr, int>>>;
    auto mng = input->func_graph()->manager();
    MS_EXCEPTION_IF_NULL(mng);
    const NodeUsersMap &users = mng->node_users();
    auto input_users = users.find(input);
    if (input_users == users.end() || input_users->second.empty()) {
      MS_EXCEPTION(ArgumentError) << "Input [" << i << "][" << input->DebugString(2) << "] of ["
                                  << input->func_graph()->ToString() << "] has no users.";
    }

    for (auto const &input_user : input_users->second) {
      for (auto const &anf_node : node_list) {
        if (anf_node != input_user.first) {
          continue;
        }

        std::vector<int> dyn_input_sizes;
        auto prim = AnfAlgo::GetCNodePrimitive(anf_node);
        MS_EXCEPTION_IF_NULL(prim);
        if (prim->GetAttr(kAttrDynInputSizes) != nullptr) {
          dyn_input_sizes = GetValue<const std::vector<int>>(prim->GetAttr(kAttrDynInputSizes));
        }

        if (dyn_input_sizes.empty()) {
          input_index.push_back(std::make_pair(anf_node, std::make_pair(IntToSize(input_user.second - 1), 0)));
          found = true;
          break;
        } else {
          int used_as_idx = input_user.second - 1;
          int accum_idx = 0;
          size_t dyn_i = 0;
          for (; dyn_i < dyn_input_sizes.size(); ++dyn_i) {
            accum_idx += dyn_input_sizes[dyn_i];
            if (used_as_idx < accum_idx) {
              input_index.push_back(std::make_pair(
                anf_node, std::make_pair(dyn_i, IntToSize(used_as_idx - (accum_idx - dyn_input_sizes[dyn_i])))));
              break;
            }
          }
          if (dyn_i != dyn_input_sizes.size()) {
            found = true;
            break;
          }
        }
      }
      if (found) {
        break;
      }
    }

    if (!found) {
      MS_EXCEPTION(ArgumentError) << "Input [" << i << "][" << input->DebugString(2) << "] of ["
                                  << input->func_graph()->ToString() << "] found no related kernel info.";
    }
  }
  return input_index;
}

std::vector<std::pair<AnfNodePtr, size_t>> GetOutputIndex(const std::vector<AnfNodePtr> &node_list,
                                                          const std::vector<AnfNodePtr> &input_list,
                                                          const std::vector<AnfNodePtr> &output_list) {
  std::vector<std::pair<AnfNodePtr, size_t>> output_index;
  for (size_t i = 0; i < output_list.size(); ++i) {
    auto const &output = output_list[i];
    MS_EXCEPTION_IF_NULL(output);
    bool found = false;
    auto pree_node = AnfAlgo::VisitKernel(output, 0);
    auto pos = std::find(std::begin(node_list), std::end(node_list), pree_node.first);
    if (pos != std::end(node_list)) {
      output_index.push_back(pree_node);
      continue;
    }
    auto ret = std::find(std::begin(input_list), std::end(input_list), pree_node.first);
    if (ret != std::end(input_list)) {
      output_index.push_back(std::make_pair(pree_node.first, 0));
      found = true;
    }
    if (!found) {
      MS_EXCEPTION(ArgumentError) << "Output [" << i << "][" << output->DebugString(2) << "] of ["
                                  << output->func_graph()->ToString() << "] found no related kernel info.";
    }
  }
  return output_index;
}

void GetValidKernelNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *node_list) {
  MS_EXCEPTION_IF_NULL(node_list);
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_lists = TopoSort(func_graph->get_return());
  for (auto const &node : node_lists) {
    if (!AnfAlgo::IsRealKernel(node) || !node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (IsValueNode<Primitive>(cnode->input(kAnfPrimitiveIndex))) {
      node_list->push_back(node);
    }
  }
}

void GetValidKernelNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *node_list,
                         std::vector<AnfNodePtr> *input_list, std::vector<AnfNodePtr> *output_list) {
  MS_EXCEPTION_IF_NULL(node_list);
  MS_EXCEPTION_IF_NULL(input_list);
  MS_EXCEPTION_IF_NULL(output_list);
  MS_EXCEPTION_IF_NULL(func_graph);

  GetValidKernelNodes(func_graph, node_list);

  auto parameters = func_graph->parameters();
  input_list->insert(input_list->begin(), parameters.begin(), parameters.end());

  auto func_output = func_graph->output();
  MS_EXCEPTION_IF_NULL(func_output);
  if (func_output->isa<CNode>()) {
    // multi output.
    auto cnode = func_output->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto input0 = cnode->input(kAnfPrimitiveIndex);
    MS_EXCEPTION_IF_NULL(input0);
    if (IsPrimitive(input0, prim::kPrimMakeTuple)) {
      for (size_t input_idx = 1; input_idx < cnode->inputs().size(); ++input_idx) {
        auto input_node = cnode->input(input_idx);
        MS_EXCEPTION_IF_NULL(input_node);
        output_list->push_back(AnfAlgo::VisitKernel(input_node, 0).first);
      }
    } else {
      // single output.
      output_list->push_back(AnfAlgo::VisitKernel(func_output, 0).first);
    }
  } else {
    // single output.
    output_list->push_back(AnfAlgo::VisitKernel(func_output, 0).first);
  }
}

bool GetInputTensorValue(const AnfNodePtr &anf_node, size_t input_idx, nlohmann::json *const node_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(node_json);
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (input_idx + 1 >= cnode->size()) {
    MS_EXCEPTION(ArgumentError) << "input_idx [" << input_idx << "] is out of index of inputs of ["
                                << cnode->inputs().size() << "][" << cnode->DebugString() << "]";
  }

  auto input_node = cnode->input(input_idx + 1);
  if (!IsValueNode<tensor::Tensor>(input_node)) {
    return false;
  }

  auto tensor = GetValueNode<tensor::TensorPtr>(input_node);
  if (tensor == nullptr) {
    return false;
  }

  auto type_id = tensor->data_type();
  auto *data = tensor->data_c();
  MS_EXCEPTION_IF_NULL(data);
  if (tensor->DataDim() > 1 || tensor->DataSize() != 1) {
    // not const tensor.
    MS_LOG(WARNING) << "We take first value of tensor whose datasize != 1, [" << input_node->DebugString(2) << "]";
  }

  if (type_id == kFloat32->type_id()) {
    float *val = static_cast<float *>(data);
    MS_EXCEPTION_IF_NULL(val);
    (*node_json)["value"] = val[0];
    MS_LOG(DEBUG) << "Value of tensor[" << cnode->DebugString() << "] is [float32][" << *val << "].";
    return true;
  } else if (type_id == kFloat16->type_id()) {
    float16 *val = static_cast<float16 *>(data);
    MS_EXCEPTION_IF_NULL(val);
    (*node_json)["value"] = static_cast<float>(val[0]);
    MS_LOG(INFO) << "Value of tensor[" << cnode->DebugString() << "] is [float16][" << *val << "].";
    return true;
  } else if (type_id == kInt32->type_id()) {
    int *val = static_cast<int *>(data);
    MS_EXCEPTION_IF_NULL(val);
    (*node_json)["value"] = val[0];
    MS_LOG(INFO) << "Value of tensor[" << cnode->DebugString() << "] is [int32][" << *val << "].";
    return true;
  }
  MS_LOG(ERROR) << "Unknown value type of tensor[" << cnode->DebugString() << "]";
  return false;
}

void GetGraphRealOutput(const FuncGraphPtr &func_graph, std::vector<std::pair<AnfNodePtr, size_t>> *node_list) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node_list);
  auto output = func_graph->output();
  MS_EXCEPTION_IF_NULL(output);
  if (AnfAlgo::IsRealKernel(output)) {
    // single output.
    node_list->push_back(std::make_pair(output, 0));
    return;
  } else if (IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
    auto output_cnode = output->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(output_cnode);
    // multi output.
    auto &inputs = output_cnode->inputs();
    for (size_t i = 1; i < inputs.size(); ++i) {
      auto in_with_idx = AnfAlgo::VisitKernel(inputs[i], 0);
      node_list->push_back(in_with_idx);
    }
    return;
  }
  MS_EXCEPTION(ArgumentError) << "Unknown  output type: " << output->DebugString(2)
                              << " of graph: " << func_graph->ToString();
}

bool IsWeightBoundary(const AnfNodePtr &node) {
  if (node->isa<ValueNode>()) {
    return true;
  }
  if (node->isa<Parameter>() && AnfAlgo::IsParameterWeight(node->cast<ParameterPtr>())) {
    return true;
  }
  return false;
}

void MultiThreadCompute(const MultiThreadComputeFunc &func, MultiThreadComputeParams *params,
                        size_t total_compute_size) {
  const size_t kThreadNum = 24;
  std::vector<std::thread> threads;
  threads.reserve(kThreadNum);
  size_t start = 0;
  size_t once_compute_size = (total_compute_size + kThreadNum - 1) / kThreadNum;
  while (start < total_compute_size) {
    size_t end = (start + once_compute_size) > total_compute_size ? total_compute_size : (start + once_compute_size);
    threads.emplace_back(std::thread(func, params, start, end));
    start += once_compute_size;
  }
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }
}

std::vector<int> GetReduceAttrAxis(const CNodePtr &cnode) {
  if (AnfAlgo::GetInputTensorNum(cnode) != AnfAlgo::GetOutputTensorNum(cnode) &&
      AnfAlgo::GetInputTensorNum(cnode) != 1) {
    MS_LOG(EXCEPTION) << "the kind of reduce node [" << cnode->DebugString()
                      << "] is not single input or single output ";
  }
  std::vector<int> axis;
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(cnode, 0);
  auto primitive = AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);
  auto axis_attr = primitive->GetAttr(kAxis);
  if (axis_attr == nullptr) {
    MS_LOG(ERROR) << "This node does't have axie attr.";
    return std::vector<int>();
  }
  auto type = axis_attr->type();
  MS_EXCEPTION_IF_NULL(type);
  std::vector<int> axis_list;
  if (type->ToString() == kTypeInt32) {
    axis_list.emplace_back(GetValue<int>(axis_attr));
  } else {
    axis_list = GetValue<std::vector<int>>(axis_attr);
  }
  for (const auto &elem : axis_list) {
    if (elem < 0) {
      axis.emplace_back(input_shape.size() + elem);
    } else {
      axis.emplace_back(elem);
    }
  }
  AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(axis), cnode);
  return axis;
}
}  // namespace kernel
}  // namespace mindspore
