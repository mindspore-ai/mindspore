/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "kernel/oplib/oplib.h"

#include <memory>
#include <map>
#include <utility>
#include <fstream>
#include "kernel/oplib/op_info_keys.h"
#include "kernel/oplib/opinfo.h"
#include "utils/log_adapter.h"
#include "utils/overload.h"
#include "utils/ms_context.h"
#include "kernel/oplib/super_bar.h"
#include "utils/file_utils.h"

namespace mindspore::kernel {
std::vector<std::string> SplitStrToVec(const std::string &input) {
  static const std::map<std::string, std::string> kSpecFormat = {{kOpFormat_NCHW, kOpFormat_DEFAULT},
                                                                 {kOpFormat_ND, kOpFormat_DEFAULT}};
  if (input.empty()) {
    MS_LOG(INFO) << "Input string is empty.";
    return {};
  }
  // remove blank elem
  std::string input_tmp = input;
  (void)input_tmp.erase(remove(input_tmp.begin(), input_tmp.end(), ' '), input_tmp.end());
  (void)input_tmp.append(",");
  // split
  const char sep = ',';
  std::vector<std::string> result = {};
  auto begin = 0U;
  auto end = input_tmp.find(sep);
  while (end != std::string::npos) {
    auto format = input_tmp.substr(begin, end - begin);
    auto find_iter = kSpecFormat.find(format);
    if (find_iter != kSpecFormat.end()) {
      format = find_iter->second;
    }
    (void)result.emplace_back(format);
    begin = end + 1;
    end = input_tmp.find(sep, begin);
  }
  return result;
}

bool OpLib::RegOp(const std::string &json_string, const std::string &impl_path) {
  try {
    auto op_json = nlohmann::json::parse(json_string);
    std::string op_name = op_json.at(kOpName);
    std::string imply_type_str = op_json.at(kImplyType);
    auto iter = kImplyTypeStrToEnumMap.find(imply_type_str);
    if (iter == kImplyTypeStrToEnumMap.end()) {
      MS_LOG(ERROR) << "Not support imply_type: " << imply_type_str;
      return false;
    }
    auto imply_type = iter->second;
    std::string key_suffix = imply_type_str;
    if (imply_type_str == kImplyAKGStr) {
      key_suffix = op_json.at(kProcessor);
    }
    auto key = op_name + key_suffix;
    auto &op_infos = GetOpInfoMap();
    auto op_infos_iter = op_infos.find(imply_type);
    if (op_infos_iter != op_infos.end()) {
      auto op_info_iter = op_infos_iter->second.find(key);
      bool is_custom_op = (!impl_path.empty() || op_name.find(".so:") != std::string::npos);
      if (op_info_iter != op_infos_iter->second.end() && !is_custom_op) {
        MS_LOG(ERROR) << "Op: " << op_name << ", processor: " << key_suffix << ", imply type: " << imply_type_str
                      << "input json: " << json_string << "has been registered.";
        return false;
      }
    }
    auto op_info = DecodeOpInfo(op_json, imply_type, impl_path);
    if (op_info == nullptr) {
      MS_LOG(ERROR) << "RegOp failed: op_name: " << op_name << " imply_type " << imply_type_str;
      return false;
    }
    op_info->set_processor(key_suffix);
    (void)op_infos[imply_type].insert(std::pair<std::string, OpInfoPtr>(key, op_info));
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get op json elements failed: " << e.what();
  }
  return true;
}

void OpLib::DecodeTBESpecificInfo(const nlohmann::json &obj, const std::shared_ptr<OpInfo> &op_info) {
  const std::map<std::string, kernel::OpPattern> kOpPatternMap = {
    {kFormatAgnostic, kFormatAgnosticPattern}, {kBroadcast, kBroadcastPattern}, {kReduce, kReducePattern}};
  MS_EXCEPTION_IF_NULL(op_info);
  op_info->set_async(obj.at(kAsyncFlag));
  op_info->set_bin_file(obj.at(kBinfile));
  op_info->set_compute(obj.at(kComputeCost));
  op_info->set_op_interface(obj.at(kKernel));
  op_info->set_partial(obj.at(kPartialFlag));
  op_info->set_need_check_supported(obj.at(kNeedCheckSupport));
  if (obj.find(kDynamincRankSupport) != obj.end()) {
    op_info->set_dynamic_rank_support(obj.at(kDynamincRankSupport));
  }

  if (obj.find(kDynamicShapeSupport) != obj.end()) {
    op_info->set_dynamic_shape_support(obj.at(kDynamicShapeSupport));
  }

  if (obj.find(kDynamicCompileStatic) != obj.end()) {
    op_info->set_dynamic_compile_static(obj.at(kDynamicCompileStatic));
  }

  auto dynamic_iter = obj.find(kDynamicFormat);
  if (dynamic_iter != obj.end()) {
    bool is_dynamic_format = dynamic_iter->get<bool>();
    if (is_dynamic_format) {
      op_info->set_op_pattern(kDynamicFormatPattern);
    }
    op_info->set_dynamic_format(is_dynamic_format);
  }
  if (obj.find(kOpPattern) != obj.end()) {
    std::string op_pattern = obj.at(kOpPattern);
    auto find_iter = kOpPatternMap.find(op_pattern);
    if (find_iter != kOpPatternMap.end()) {
      op_info->set_op_pattern(find_iter->second);
    } else {
      if (!op_pattern.empty()) {
        MS_LOG(WARNING) << "The error pattern: " << op_pattern;
      }
    }
  }
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
    MS_LOG(INFO) << "MINDSPORE_OP_INFO_PATH has not been set, return.";
    return true;
  }
  auto real_path = FileUtils::GetRealPath(dir.c_str());
  if (!real_path.has_value()) {
    MS_LOG(INFO) << "Invalid environment variable 'MINDSPORE_OP_INFO_PATH', the path is: " << dir
                 << ". Please check (1) whether the path exists, (2) whether the path has the access permission, "
                 << "(3) whether the path is too long. ";
    return false;
  }
  std::ifstream file(real_path.value());
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
  file.close();
  MS_LOG(INFO) << "End";
  return true;
}

std::shared_ptr<OpInfo> OpLib::DecodeOpInfo(const nlohmann::json &obj, const mindspore::kernel::OpImplyType &imply_type,
                                            const std::string &impl_path) {
  std::shared_ptr<OpInfo> op_info = std::make_shared<OpInfo>();
  MS_EXCEPTION_IF_NULL(op_info);
  op_info->set_op_name(obj.at(kOpName));
  op_info->set_impl_path(impl_path);
  op_info->set_imply_type(imply_type);
  if (imply_type == kImplyTBE) {
    DecodeTBESpecificInfo(obj, op_info);
  }
  auto attrs = obj.at(kAttr);
  for (const auto &attr : attrs) {
    if (!DecodeAttr(attr, imply_type, op_info)) {
      MS_LOG(ERROR) << "DecodeAttr Failed";
      return nullptr;
    }
  }
  nlohmann::json dtype_format;
  if (obj.find(kDtypeFormat) != obj.end()) {
    dtype_format = obj.at(kDtypeFormat);
  }
  auto inputs = obj.at(kIputs);
  for (const auto &input : inputs) {
    if (!DecodeInputOutput(input, imply_type, true, op_info, dtype_format)) {
      MS_LOG(ERROR) << "DecodeInputOutput Failed";
      return nullptr;
    }
  }
  auto outputs = obj.at(kOutputs);
  for (const auto &output : outputs) {
    if (!DecodeInputOutput(output, imply_type, false, op_info, dtype_format)) {
      MS_LOG(ERROR) << "DecodeInputOutput Failed";
      return nullptr;
    }
  }
  if (!GetRefInfo(op_info)) {
    MS_LOG(ERROR) << "GetRefInfo Failed";
    return nullptr;
  }
  return op_info;
}

bool OpLib::DecodeAttr(const nlohmann::json &obj, const OpImplyType &imply_type,
                       const std::shared_ptr<OpInfo> &op_info) {
  MS_EXCEPTION_IF_NULL(op_info);
  bool ret = true;
  try {
    std::shared_ptr<OpAttr> op_attr = std::make_shared<OpAttr>();
    MS_EXCEPTION_IF_NULL(op_attr);
    op_attr->set_name(obj.at(kName));
    if (imply_type != kImplyAICPU) {
      op_attr->set_param_type(obj.at(kParamType));
    }
    op_attr->set_type(obj.at(kType));
    if (imply_type == kImplyTBE) {
      op_attr->set_value(obj.at(kValue));
    }
    if (obj.find(kDefaultValue) != obj.end()) {
      op_attr->set_default_value(obj.at(kDefaultValue));
    }
    op_info->add_attrs_ptr(op_attr);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "DecodeAttr failed:" << e.what() << ", input: " << obj.dump();
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
    std::vector<std::string> object_type;
    for (const auto &it : dtype_format) {
      dtype.emplace_back(it[index][kIndex0]);
      format.emplace_back(it[index][kIndex1]);
      if (it[index].size() == kIndex3) {
        object_type.emplace_back(it[index][kIndex2]);
      } else {
        object_type.emplace_back("tensor");
      }
    }
    op_io->set_dtypes(dtype);
    op_io->set_formats(format);
    op_io->set_object_types(object_type);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "DecodeDtypeFormat failed" << e.what();
    ret = false;
  }
  return ret;
}

bool OpLib::DecodeInputOutput(const nlohmann::json &obj, OpImplyType imply_type, bool is_input,
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
      if (op_info->dynamic_shape_support()) {
        op_io->set_unknown_shape_formats(op_io->formats());
      }
    } else {
      op_io->set_dtypes(obj.at(kDtype));
      op_io->set_formats(obj.at(kFormat));
      op_io->set_object_types(std::vector<std::string>(op_io->dtypes().size(), "tensor"));
      if (op_info->dynamic_shape_support()) {
        op_io->set_unknown_shape_formats(obj.at(kFormat));
      }
    }
    if (op_io->dtypes().size() != op_io->formats().size()) {
      MS_LOG(ERROR) << "op " << op_io->name() << " dtype size: " << op_io->dtypes()
                    << " is not equal to format size: " << op_io->formats();
      return false;
    }
    if (obj.find(kParamType) != obj.end()) {
      op_io->set_param_type(obj.at(kParamType));
    }
    if (imply_type == kImplyTBE) {
      if (obj.find(kNeedCompile) != obj.end()) {
        op_io->set_need_compile(obj.at(kNeedCompile));
      }
      if (obj.find(kShape) != obj.end()) {
        op_io->set_shape(obj.at(kShape));
      }
      if (obj.find(kReshape_Type) != obj.end()) {
        op_io->set_reshape_type(obj.at(kReshape_Type));
      }
      if (obj.find(kValueDepend) != obj.end()) {
        op_io->set_value_depend(obj.at(kValueDepend));
      }
    }

    if (is_input) {
      op_info->add_inputs_ptr(op_io);
    } else {
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
  auto &op_infos = GetOpInfoMap();
  auto op_infos_iter = op_infos.find(imply_type);
  if (op_infos_iter == op_infos.end()) {
    MS_LOG(INFO) << "FindOp failed: opname: " << op_name << ", imply_type: " << imply_type
                 << ", current imply type num: " << op_infos.size() << " is_dynamic_shape:" << is_dynamic_shape;
    return nullptr;
  }
  auto impl_type_iter = kImplyTypeEnumToStrMap.find(imply_type);
  if (impl_type_iter == kImplyTypeEnumToStrMap.end()) {
    MS_LOG(ERROR) << "FindOp failed: opname: " << op_name << ", imply_type: " << imply_type
                  << ", current imply type num: " << op_infos.size() << " is_dynamic_shape:" << is_dynamic_shape;
    return nullptr;
  }
  auto key_suffix = impl_type_iter->second;
  if (key_suffix == kImplyAKGStr) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    auto device = context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    auto processor_iter = kProcessorMap.find(device);
    if (processor_iter == kProcessorMap.end()) {
      return nullptr;
    }
    key_suffix = processor_iter->second;
  }
  auto key = op_name + key_suffix;
  auto op_info_iter = op_infos_iter->second.find(key);
  if (op_info_iter == op_infos_iter->second.end()) {
    MS_LOG(INFO) << "Note: Op: " << op_name << ", processor: " << key_suffix << ", imply type: " << imply_type
                 << " is not exist, op info num: " << op_infos_iter->second.size();
    return nullptr;
  }

  return op_info_iter->second;
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

std::map<mindspore::kernel::OpImplyType, std::map<std::string, std::shared_ptr<OpInfo>>> &OpLib::GetOpInfoMap() {
  static std::map<mindspore::kernel::OpImplyType, std::map<std::string, std::shared_ptr<OpInfo>>> op_infos;
  return op_infos;
}
}  // namespace mindspore::kernel
