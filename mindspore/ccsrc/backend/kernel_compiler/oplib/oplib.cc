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

#include "backend/kernel_compiler/oplib/oplib.h"
#include <memory>
#include <map>
#include <fstream>
#include "utils/log_adapter.h"
#include "utils/overload.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace kernel {
constexpr auto kImplyType = "imply_type";
constexpr auto kOpName = "op_name";
constexpr auto kFusionType = "fusion_type";
constexpr auto kAsyncFlag = "async_flag";
constexpr auto kBinfileName = "binfile_name";
constexpr auto kComputeCost = "compute_cost";
constexpr auto kKernelName = "kernel_name";
constexpr auto kPartialFlag = "partial_flag";
constexpr auto kReshapeType = "reshape_type";
constexpr auto kOpPattern = "op_pattern";
constexpr auto kIsDynamicFormat = "is_dynamic_format";
constexpr auto kDynamicFormat = "dynamicFormat";
constexpr auto kFormatAgnostic = "formatAgnostic";
constexpr auto kNeedCheckSupported = "need_check_supported";
constexpr auto kBroadcast = "broadcast";
constexpr auto kReduce = "reduce";
constexpr auto kDynamicShape = "dynamic_shape";
constexpr auto kDtypeFormat = "dtype_format";
constexpr auto kAttr = "attr";
constexpr auto kIputs = "inputs";
constexpr auto kOutputs = "outputs";
constexpr auto kAiCPU = "AiCPU";
constexpr auto kAiCore = "AiCore";
constexpr auto kCUDA = "CUDA";
constexpr auto kTbe = "TBE";
constexpr auto kAkg = "AKG";
constexpr auto kName = "name";
constexpr auto kParamType = "param_type";
constexpr auto kDtype = "dtype";
constexpr auto kType = "type";
constexpr auto kValue = "value";
constexpr auto kDefaultValue = "default_value";
constexpr auto kIndex = "index";
constexpr auto kFormat = "format";
constexpr auto kNeedCompile = "need_compile";
constexpr auto kShape = "shape";
constexpr auto kProcessor = "processor";
std::multimap<std::string, std::shared_ptr<OpInfo>> OpLib::op_info_;

static std::string ImplTypeToStr(OpImplyType impl_type) {
  switch (impl_type) {
    case kTBE:
      return kTbe;
    case kAKG:
      return kAkg;
    case kAICPU:
      return kAiCPU;
    default:
      return "unknown";
  }
}
bool OpLib::RegOp(const std::string &json_string, const std::string &impl_path) {
  bool ret = false;
  try {
    auto op_json = nlohmann::json::parse(json_string);
    std::string imply_type_string = op_json.at(kImplyType);
    std::string op_name = op_json.at(kOpName);
    if (imply_type_string == kTbe) {
      OpImplyType imply_type = kTBE;
      ret = DecodeOpInfo(op_json, imply_type, impl_path);
    } else if (imply_type_string == kAkg) {
      OpImplyType imply_type = kAKG;
      ret = DecodeOpInfo(op_json, imply_type, impl_path);
    } else if (imply_type_string == kAiCPU) {
      OpImplyType imply_type = kAICPU;
      ret = DecodeOpInfo(op_json, imply_type, impl_path);
    } else {
      MS_LOG(ERROR) << "Not support imply_type";
    }
    if (!ret) {
      MS_LOG(ERROR) << "RegOp failed: op_name: " << op_name << " imply_type " << imply_type_string;
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "get op json elements failed: " << e.what();
  }
  return ret;
}

void OpLib::DecodeTBESpecificInfo(const nlohmann::json &obj, const std::shared_ptr<OpInfo> &op_info) {
  const std::map<std::string, kernel::OpPattern> kOpPatternMap = {
    {kFormatAgnostic, kFormatAgnosticPattern}, {kBroadcast, kBroadcastPattern}, {kReduce, kReducePattern}};
  MS_EXCEPTION_IF_NULL(op_info);
  op_info->set_async_flag(obj.at(kAsyncFlag));
  op_info->set_binfile_name(obj.at(kBinfileName));
  op_info->set_compute_cost(obj.at(kComputeCost));
  op_info->set_kernel_name(obj.at(kKernelName));
  op_info->set_partial_flag(obj.at(kPartialFlag));
  op_info->set_need_check_supported(obj.at(kNeedCheckSupported));

  if (obj.find(kDynamicShape) != obj.end()) {
    op_info->set_dynamic_shape(obj.at(kDynamicShape));
  }

  if (obj.find(kIsDynamicFormat) != obj.end()) {
    op_info->set_is_dynamic_format(obj.at(kIsDynamicFormat));
  }

  if (obj.find(kOpPattern) != obj.end()) {
    std::string op_pattern = obj.at(kOpPattern);
    auto find_iter = kOpPatternMap.find(op_pattern);
    if (find_iter == kOpPatternMap.end()) {
      if (!op_pattern.empty()) {
        MS_LOG(WARNING) << "Op pattern set value error: " << op_pattern;
      }
      op_info->set_op_pattern(kCommonPattern);
    } else {
      op_info->set_op_pattern(find_iter->second);
    }
  }
}

void OpLib::DecodeAKGSpecificInfo(const nlohmann::json &obj, const std::shared_ptr<OpInfo> &op_info) {
  MS_EXCEPTION_IF_NULL(op_info);
  op_info->set_processor(obj.at(kProcessor));
}

bool OpLib::RegOpFromLocalInfo() {
  static bool has_load = false;
  if (has_load) {
    return true;
  }
  MS_LOG(INFO) << "Start";
  has_load = true;
  std::string dir = common::GetEnv("MINDSPORE_OP_INFO_PATH");
  if (dir.empty()) {
    MS_LOG(INFO) << "MindSpore op info path does not been set. use op info from python pass.";
    return true;
  }
  char real_path[PATH_MAX] = {0};
  if (dir.size() >= PATH_MAX) {
    MS_LOG(ERROR) << "Op info path is invalid: " << dir;
    return false;
  }
#if defined(_WIN32) || defined(_WIN64)
  if (_fullpath(real_path, common::SafeCStr(dir), PATH_MAX) == nullptr) {
    MS_LOG(ERROR) << "Op info path is invalid: " << dir;
    return false;
  }
#else
  if (realpath(common::SafeCStr(dir), real_path) == nullptr) {
    MS_LOG(ERROR) << "Op info path is invalid: " << dir;
    return false;
  }
  if (strlen(real_path) >= PATH_MAX) {
    MS_LOG(ERROR) << "Op info path is invalid, the absolute path length is greater than PATH_MAX";
    return false;
  }
#endif
  MS_LOG(INFO) << "Start to read op info from local file.";
  std::ifstream file(real_path);
  if (!file.is_open()) {
    MS_LOG(ERROR) << "Find op info file failed.";
    return false;
  }
  std::string line;
  while (getline(file, line)) {
    if (!line.empty()) {
      (void)OpLib::RegOp(line, "");
    }
  }
  MS_LOG(INFO) << "End";
  return true;
}

bool OpLib::DecodeOpInfo(const nlohmann::json &obj, const mindspore::kernel::OpImplyType imply_type,
                         const std::string &impl_path) {
  std::shared_ptr<OpInfo> op_info = std::make_shared<OpInfo>();
  MS_EXCEPTION_IF_NULL(op_info);
  op_info->set_op_name(obj.at(kOpName));
  op_info->set_impl_path(impl_path);
  op_info->set_imply_type(imply_type);
  op_info->set_fusion_type(obj.at(kFusionType));
  if (imply_type == kTBE) {
    DecodeTBESpecificInfo(obj, op_info);
  } else if (imply_type == kAKG) {
    DecodeAKGSpecificInfo(obj, op_info);
  }
  auto attrs = obj.at(kAttr);
  for (const auto &attr : attrs) {
    if (!DecodeAttr(attr, imply_type, op_info)) {
      MS_LOG(ERROR) << "DecodeAttr Failed";
      return false;
    }
  }
  nlohmann::json dtype_format;
  if (obj.find(kDtypeFormat) != obj.end()) {
    dtype_format = obj.at(kDtypeFormat);
  }
  auto inputs = obj.at(kIputs);
  for (const auto &input : inputs) {
    if (!DecodeInputOutput(input, imply_type, kInput, op_info, dtype_format)) {
      MS_LOG(ERROR) << "DecodeInputOutput Failed";
      return false;
    }
  }
  auto outputs = obj.at(kOutputs);
  for (const auto &output : outputs) {
    if (!DecodeInputOutput(output, imply_type, kOutput, op_info, dtype_format)) {
      MS_LOG(ERROR) << "DecodeInputOutput Failed";
      return false;
    }
  }
  if (CheckRepetition(op_info)) {
    MS_LOG(WARNING) << "This op info has been already registered. op name: " << op_info->op_name()
                    << ", impl type: " << ImplTypeToStr(op_info->imply_type())
                    << ", impl path: " << op_info->impl_path();
    return true;
  }
  if (!GetRefInfo(op_info)) {
    MS_LOG(ERROR) << "GetRefInfo Failed";
    return false;
  }
  op_info_.emplace(op_info->op_name(), op_info);
  return true;
}

bool OpLib::DecodeAttr(const nlohmann::json &obj, const OpImplyType imply_type,
                       const std::shared_ptr<OpInfo> &op_info) {
  MS_EXCEPTION_IF_NULL(op_info);
  bool ret = true;
  try {
    std::shared_ptr<OpAttr> op_attr = std::make_shared<OpAttr>();
    MS_EXCEPTION_IF_NULL(op_attr);
    op_attr->set_name(obj.at(kName));
    if (imply_type != kAICPU) {
      op_attr->set_param_type(obj.at(kParamType));
    }
    op_attr->set_type(obj.at(kType));
    if (imply_type == kTBE) {
      op_attr->set_value(obj.at(kValue));
    }
    if (obj.find(kDefaultValue) != obj.end()) {
      op_attr->set_default_value(obj.at(kDefaultValue));
    }
    op_info->add_attrs_ptr(op_attr);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "DecodeAttr failed:" << e.what();
    ret = false;
  }
  return ret;
}

bool OpLib::DecodeDtypeFormat(const nlohmann::json &dtype_format, const std::shared_ptr<OpIOInfo> &op_io,
                              size_t index) {
  MS_EXCEPTION_IF_NULL(op_io);
  bool ret = true;
  try {
    std::vector<std::string> dtype;
    std::vector<std::string> format;
    for (const auto &it : dtype_format) {
      dtype.emplace_back(it[index][0]);
      format.emplace_back(it[index][1]);
    }
    op_io->set_dtypes(dtype);
    op_io->set_formats(format);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "DecodeDtypeFormat failed" << e.what();
    ret = false;
  }
  return ret;
}

bool OpLib::DecodeInputOutput(const nlohmann::json &obj, const OpImplyType imply_type, const OpIOType io_type,
                              const std::shared_ptr<OpInfo> &op_info, const nlohmann::json &dtype_format) {
  MS_EXCEPTION_IF_NULL(op_info);
  bool ret = true;
  try {
    std::shared_ptr<OpIOInfo> op_io = std::make_shared<OpIOInfo>();
    MS_EXCEPTION_IF_NULL(op_io);
    op_io->set_index(obj.at(kIndex));
    op_io->set_name(obj.at(kName));
    if (!dtype_format.empty()) {
      if (!DecodeDtypeFormat(dtype_format, op_io, op_info->inputs_ptr().size() + op_info->outputs_ptr().size())) {
        MS_LOG(ERROR) << "Decode dtype format failed";
        return false;
      }
    } else {
      op_io->set_dtypes(obj.at(kDtype));
      op_io->set_formats(obj.at(kFormat));
    }
    if (op_io->dtypes().size() != op_io->formats().size()) {
      MS_LOG(ERROR) << "op " << op_io->name() << " dtype size: " << op_io->dtypes()
                    << " is not equal to format size: " << op_io->formats();
      return false;
    }
    if (obj.find(kParamType) != obj.end()) {
      op_io->set_param_type(obj.at(kParamType));
    }
    if (imply_type == kTBE) {
      if (obj.find(kNeedCompile) != obj.end()) {
        op_io->set_need_compile(obj.at(kNeedCompile));
      }
      if (obj.find(kShape) != obj.end()) {
        op_io->set_shape(obj.at(kShape));
      }
      if (obj.find(kReshapeType) != obj.end()) {
        op_io->set_reshape_type(obj.at(kReshapeType));
      }
    }

    if (io_type == kInput) {
      op_info->add_inputs_ptr(op_io);
    } else if (io_type == kOutput) {
      op_info->add_outputs_ptr(op_io);
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "DecodeInputOutput failed" << e.what();
    ret = false;
  }
  return ret;
}

std::shared_ptr<OpInfo> OpLib::FindOp(const std::string &op_name, OpImplyType imply_type, bool is_dynamic_shape) {
  if (!OpLib::RegOpFromLocalInfo()) {
    MS_LOG(INFO) << "Warning reg local op info failed.";
  }
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool is_gpu = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice);
  if (is_gpu && (imply_type == kTBE || imply_type == kAICPU)) {
    MS_LOG(ERROR) << "FindOp failed: opname: " << op_name << ", imply_type: " << ImplTypeToStr(imply_type)
                  << ", current op num: " << op_info_.size();
    return nullptr;
  }
  std::string target_processor = is_gpu ? kCUDA : kAiCore;
  for (auto [iter, end] = op_info_.equal_range(op_name); iter != end; ++iter) {
    auto &op_info = iter->second;
    MS_EXCEPTION_IF_NULL(op_info);

    if (op_info->imply_type() != imply_type) {
      continue;
    }
    if (imply_type == kAKG && op_info->processor() != target_processor) {
      continue;
    }
    if (is_dynamic_shape && !op_info->dynamic_shape()) {
      continue;
    }
    return op_info;
  }
  MS_LOG(INFO) << "FindOp failed: opname: " << op_name << ", imply_type: " << ImplTypeToStr(imply_type)
               << ", current op num: " << op_info_.size() << " is_dynamic_shape:" << is_dynamic_shape;
  return nullptr;
}

bool OpLib::GetRefInfo(const std::shared_ptr<OpInfo> &op_info) {
  MS_EXCEPTION_IF_NULL(op_info);
  const auto &output_infos = op_info->outputs_ptr();
  const auto &input_infos = op_info->inputs_ptr();
  for (size_t out_index = 0; out_index < output_infos.size(); out_index++) {
    MS_EXCEPTION_IF_NULL(output_infos[out_index]);
    const auto &out_name = output_infos[out_index]->name();
    for (size_t in_index = 0; in_index < input_infos.size(); in_index++) {
      MS_EXCEPTION_IF_NULL(input_infos[in_index]);
      const auto &in_name = input_infos[in_index]->name();
      if (out_name == in_name) {
        if (op_info->has_ref_index(out_index)) {
          MS_LOG(ERROR) << "The out_index " << out_index << " is already in ref_info";
          return false;
        }
        op_info->add_ref_pair(out_index, in_index);
      }
    }
  }
  return true;
}

bool OpLib::CheckRepetition(const std::shared_ptr<OpInfo> &op_info) {
  MS_EXCEPTION_IF_NULL(op_info);
  for (auto [iter, end] = op_info_.equal_range(op_info->op_name()); iter != end; ++iter) {
    auto &exist_op_info = iter->second;
    MS_EXCEPTION_IF_NULL(exist_op_info);
    if (exist_op_info->equals_to(op_info)) {
      return true;
    }
  }
  return false;
}
}  // namespace kernel
}  // namespace mindspore
