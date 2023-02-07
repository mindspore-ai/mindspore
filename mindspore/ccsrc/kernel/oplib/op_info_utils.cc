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
#include "kernel/oplib/op_info_utils.h"

#include <fstream>
#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <memory>
#include <nlohmann/json.hpp>
#include "kernel/oplib/op_info_keys.h"
#include "kernel/oplib/super_bar.h"
#include "kernel/oplib/oplib.h"
#include "utils/file_utils.h"
#include "utils/dlopen_macro.h"

namespace mindspore::kernel {
namespace {
constexpr auto kAscend910A = "Ascend910A";
constexpr auto kAscend910ProA = "Ascend910ProA";
constexpr auto kAscend910PremiumA = "Ascend910PremiumA";
constexpr auto kAscend920A = "Ascend920A";
constexpr auto kAscend910SuffixPath = "/opp/built-in/op_impl/ai_core/tbe/config/ascend910/aic-ascend910-ops-info.json";
constexpr auto kAscend920SuffixPath = "/opp/built-in/op_impl/ai_core/tbe/config/ascend920/aic-ascend920-ops-info.json";
const std::map<std::string, std::string> kAscendSuffixPathMap = {{kAscend910A, kAscend910SuffixPath},
                                                                 {kAscend910ProA, kAscend910SuffixPath},
                                                                 {kAscend910PremiumA, kAscend910SuffixPath},
                                                                 {kAscend920A, kAscend920SuffixPath}};

#define MAKE_SHARE_CHECK(expr1)                                 \
  try {                                                         \
    expr1;                                                      \
  } catch (std::exception & e) {                                \
    MS_LOG(EXCEPTION) << "Make share ptr failed: " << e.what(); \
  }

#if !defined(_WIN32) && !defined(_WIN64)
bool LoadSuperBarFile(nlohmann::json *super_bar) {
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void *>(LoadSuperBarFile), &dl_info) == 0) {
    MS_LOG(ERROR) << "Get dladdr error";
    return false;
  }
  std::string cur_so_path = dl_info.dli_fname;
  auto pos = cur_so_path.find_last_of(PATH_SEPARATOR);
  if (cur_so_path.empty() || pos == std::string::npos) {
    MS_LOG(ERROR) << "Current so path is empty or the path [" << cur_so_path << "] is invalid.";
    return false;
  }

  auto super_bar_path = cur_so_path.substr(0, pos) + "/../config/super_bar_config.json";
  if (super_bar_path.size() >= PATH_MAX) {
    MS_LOG(ERROR) << "Current path [" << super_bar_path << "] length " << super_bar_path.size() << " exceeds the limit "
                  << PATH_MAX;
    return false;
  }

  char real_path_mem[PATH_MAX] = {0};
  if (realpath(common::SafeCStr(super_bar_path), real_path_mem) == nullptr) {
    super_bar_path = cur_so_path.substr(0, pos) + "/../../../../config/super_bar_config.json";
    if (super_bar_path.size() >= PATH_MAX) {
      MS_LOG(ERROR) << "Current path [" << super_bar_path << "] length " << super_bar_path.size()
                    << " exceeds the limit " << PATH_MAX;
      return false;
    }

    if (realpath(common::SafeCStr(super_bar_path), real_path_mem) == nullptr) {
      MS_LOG(ERROR) << "The file of super_bar_config.json is not exists in path: [" << super_bar_path << "], ";
      return false;
    }
  }

  auto file_path = std::string(real_path_mem);
  std::ifstream file(file_path);
  if (!file.is_open()) {
    MS_LOG(ERROR) << "Open op info file failed, real_path: " << file_path;
    return false;
  }
  file >> (*super_bar);
  file.close();
  return true;
}
#endif
}  // namespace

bool OpInfoUtils::GenerateOpInfos(const std::string &version, const std::string &ascend_path) {
  // Step1: Load json file
  nlohmann::json js;
  if (!LoadOpInfoJson(version, ascend_path, &js)) {
    MS_LOG(ERROR) << "Load op info json failed, version: " << version;
    return false;
  }
  std::map<std::string, OpInfoPtr> op_infos;
  for (auto item = js.begin(); item != js.end(); ++item) {
    const std::string &op_name = item.key();
    // hard code
    if (SuperBar::IsSkipNode(op_name)) {
      MS_LOG(INFO) << "Note: Skip node: " << op_name;
      continue;
    }
    // Step2. Parse common item
    OpInfoPtr op_info_ptr = nullptr;
    MAKE_SHARE_CHECK(op_info_ptr = std::make_shared<OpInfo>())
    op_info_ptr->set_op_name(op_name);
    op_info_ptr->set_imply_type(kImplyTBE);
    op_info_ptr->set_processor(kProAICORE);
    try {
      if (!ParseCommonItem(*item, op_info_ptr)) {
        MS_LOG(ERROR) << "Parse common item failed, js: " << item->dump() << ", op_type" << op_name;
        return false;
      }
      if (op_info_ptr->op_file() == kNull) {
        continue;
      }
      // Step3. Parse op attr
      if (!ParseAttrs(*item, op_info_ptr)) {
        MS_LOG(ERROR) << "Parse attr failed, js: " << item->dump() << ", op_type" << op_name;
        return false;
      }
      // Step4. Parse op io info
      if (!ParseOpIOInfo(*item, op_info_ptr)) {
        MS_LOG(ERROR) << "Parse op info failed, js: " << item->dump() << ", op_type" << op_name;
        return false;
      }
      // Step5: Set kernel name
      SetKernelName(op_info_ptr);
    } catch (std::exception &e) {
      MS_LOG(ERROR) << "Parse op info failed. node: " << op_name << " , json info: " << item->dump() << ", because of "
                    << e.what();
      return false;
    }
    auto key = op_name + kImplyTBEStr;
    op_infos.emplace(key, op_info_ptr);
  }

  (void)OpLib::GetOpInfoMap()[kImplyTBE].insert(op_infos.begin(), op_infos.end());
  return true;
}

bool OpInfoUtils::LoadOpInfoJson(const std::string &version, const std::string &ascend_path, nlohmann::json *js_) {
#if !defined(_WIN32) && !defined(_WIN64)
  nlohmann::json super_bar;
  if (!LoadSuperBarFile(&super_bar)) {
    MS_LOG(ERROR) << "Load super bar file failed.";
    return false;
  }
  SuperBar::LoadSBConfig(super_bar);
#endif
  auto ascend_suffix_path = kAscendSuffixPathMap.find(version);
  if (ascend_suffix_path == kAscendSuffixPathMap.end()) {
    MS_LOG(ERROR) << "Get op info json suffix path failed, soc_version: " << version;
    return false;
  }
  auto op_info_json_path = ascend_path + ascend_suffix_path->second;
  auto real_path = FileUtils::GetRealPath(op_info_json_path.c_str());
  if (!real_path.has_value()) {
    MS_LOG(ERROR)
      << "Get the path:" << op_info_json_path
      << "failed. Please check (1) whether the path exists, (2) whether the path has the access permission, "
      << "(3) whether the path is too long. ";
    return false;
  }
  std::ifstream file(real_path.value());
  if (!file.is_open()) {
    MS_LOG(ERROR) << "Open op info file failed, real_path: " << real_path.value();
    return false;
  }
  file >> (*js_);
  file.close();
  return true;
}

namespace {
#define CHECK_VALUE(json, value)              \
  if ((json).find((value)) == (json).end()) { \
    return false;                             \
  }

bool GetJsonValue(const nlohmann::json &item, const std::string &key, std::string *value) {
  if (item.find(key) == item.end()) {
    MS_LOG(WARNING) << "Get value failed, item: " << item.dump() << ", key: " << key;
    return false;
  }
  *value = item.at(key);
  return true;
}

bool GetJsonSubItem(const nlohmann::json &item, const std::string &key, nlohmann::json *ret) {
  auto iter = item.find(key);
  if (iter == item.end()) {
    return false;
  }
  *ret = *iter;
  return true;
}

bool ParseCommonDynamicShapeSupport(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  CHECK_VALUE(item, kFlag)
  op_info_ptr->set_dynamic_shape_support(item.at(kFlag) == kTrue);
  return true;
}

bool ParseCommonOp(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  CHECK_VALUE(item, kPattern)
  auto iter = kPatternMap.find(item.at(kPattern));
  auto pattern = iter != kPatternMap.end() ? iter->second : kCommonPattern;
  op_info_ptr->set_op_pattern(pattern);
  return true;
}

bool ParseCommonSlicePattern(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  CHECK_VALUE(item, kValue)
  op_info_ptr->set_slice_pattern(item.at(kValue));
  return true;
}
bool ParseCommonNeedCheckSupport(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  CHECK_VALUE(item, kFlag)
  op_info_ptr->set_need_check_supported(item.at(kFlag) == kTrue);
  return true;
}

bool ParseCommonRangeLimit(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  CHECK_VALUE(item, kValue)
  op_info_ptr->set_range_limit(item.at(kValue));
  return true;
}

bool ParseCommonSgatKeyAttr(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  CHECK_VALUE(item, kList)
  std::string attr = item.at(kList);
  std::vector<std::string> sgat_key_attrs = SplitStrToVec(attr);
  op_info_ptr->set_sagt_key_attrs(sgat_key_attrs);
  return true;
}

bool ParseCommonOpFile(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  CHECK_VALUE(item, kValue)
  op_info_ptr->set_op_file(item.at(kValue));
  return true;
}

bool ParseCommonOpInterface(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  CHECK_VALUE(item, kValue)
  op_info_ptr->set_op_interface(item.at(kValue));
  return true;
}

bool ParseCommonBinfile(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  CHECK_VALUE(item, kName)
  op_info_ptr->set_bin_file(item.at(kName));
  return true;
}

bool ParseCommonKernel(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  CHECK_VALUE(item, kName)
  op_info_ptr->set_kernel(item.at(kName));
  return true;
}

bool ParseCommonAsync(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  CHECK_VALUE(item, kFlag)
  op_info_ptr->set_async(item.at(kFlag) == kTrue);
  return true;
}

bool ParseCommonCompute(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  CHECK_VALUE(item, kCost)
  std::string value = item.at(kCost);
  int compute = 0;
  try {
    compute = atoi(value.c_str());
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Parse compute failed, value: " << value << ", msg: " << e.what();
  }
  op_info_ptr->set_compute(compute);
  return true;
}

bool ParseCommonDynamicFormat(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  CHECK_VALUE(item, kFlag)
  bool is_dynamic_format = (item.at(kFlag) == "True" || item.at(kFlag) == kTrue);
  if (is_dynamic_format) {
    op_info_ptr->set_op_pattern(kDynamicFormatPattern);
  }
  op_info_ptr->set_dynamic_format(is_dynamic_format);
  return true;
}

bool ParseCommonPartial(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  CHECK_VALUE(item, kFlag)
  op_info_ptr->set_partial(item.at(kFlag) == kTrue);
  return true;
}

bool ParseCommonPrecisionReduce(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  CHECK_VALUE(item, kFlag)
  op_info_ptr->set_precision_reduce(item.at(kFlag) == kTrue);
  return true;
}

bool ParseCommonDynamicRankSupport(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  CHECK_VALUE(item, kFlag)
  op_info_ptr->set_dynamic_rank_support(item.at(kFlag) == kTrue);
  return true;
}

bool ParseCommonDynamicCompileStatic(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  CHECK_VALUE(item, kFlag)
  auto op_name = op_info_ptr->op_name();
  if (SuperBar::IsSkipDynamicCompileStaticNode(op_name)) {
    op_info_ptr->set_dynamic_compile_static(false);
  } else {
    op_info_ptr->set_dynamic_compile_static(item.at(kFlag) == kTrue);
  }
  return true;
}

bool ParseCommonHeavyOp(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  CHECK_VALUE(item, kFlag)
  op_info_ptr->set_heavy_op(item.at(kFlag) == kTrue);
  return true;
}

bool ParseCommonJitCompile(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  CHECK_VALUE(item, kFlag)
  op_info_ptr->set_jit_compile(item.at(kFlag) == kTrue);
  return true;
}

bool ParseCommonSoftSync(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  CHECK_VALUE(item, kFlag)
  op_info_ptr->set_soft_sync(item.at(kFlag) == kTrue);
  return true;
}

bool ParseCommonOpImplSwitch(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  CHECK_VALUE(item, kValue)
  op_info_ptr->set_op_impl_switch(item.at(kValue));
  return true;
}

bool ParseCommonPreBuildPattern(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  CHECK_VALUE(item, kValue)
  op_info_ptr->set_prebuild_pattern(item.at(kValue));
  return true;
}

bool ParseCommonCube(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  CHECK_VALUE(item, kFlag)
  op_info_ptr->set_cube_op(item.at(kFlag) == kTrue);
  return true;
}

const std::map<std::string, std::function<bool(const nlohmann::json &item, const OpInfoPtr &op_info_ptr)>>
  parse_common_funcs = {
    {kDynamicShapeSupport, ParseCommonDynamicShapeSupport},
    {kOp, ParseCommonOp},
    {kSlicePattern, ParseCommonSlicePattern},
    {kNeedCheckSupport, ParseCommonNeedCheckSupport},
    {kRangeLimit, ParseCommonRangeLimit},
    {kGgatKeyAttr, ParseCommonSgatKeyAttr},
    {kOpFile, ParseCommonOpFile},
    {kOpInterface, ParseCommonOpInterface},
    {kBinfile, ParseCommonBinfile},
    {kKernel, ParseCommonKernel},
    {kAsync, ParseCommonAsync},
    {kCompute, ParseCommonCompute},
    {kDynamicFormat, ParseCommonDynamicFormat},
    {kPartial, ParseCommonPartial},
    {kPrecisionReduce, ParseCommonPrecisionReduce},
    {kDynamincRankSupport, ParseCommonDynamicRankSupport},
    {kDynamicCompileStatic, ParseCommonDynamicCompileStatic},
    {kHeavyOp, ParseCommonHeavyOp},
    {kCubeOp, ParseCommonCube},
    {kJitCompile, ParseCommonJitCompile},
    {kSoftSync, ParseCommonSoftSync},
    {kOpImplSwitch, ParseCommonOpImplSwitch},
    {kPreBuildPattern, ParseCommonPreBuildPattern},
};

bool ParseAttrParamType(const std::string &value, const OpAttrPtr &op_attr_ptr) {
  op_attr_ptr->set_param_type(value);
  return true;
}

bool ParseAttrType(const std::string &value, const OpAttrPtr &op_attr_ptr) {
  op_attr_ptr->set_type(value);
  return true;
}

bool ParseAttrDefaultValue(const std::string &value, const OpAttrPtr &op_attr_ptr) {
  std::string res = value;
  auto pos = value.find('[');
  if (pos != std::string::npos) {
    const size_t remove_len = 2;
    res = value.substr(1, value.length() - remove_len);
  }
  op_attr_ptr->set_default_value(res);
  return true;
}

bool ParseAttrValue(const std::string &value, const OpAttrPtr &op_attr_ptr) {
  op_attr_ptr->set_value(value);
  return true;
}

std::map<std::string, std::function<bool(const std::string &value, const OpAttrPtr &op_attr_ptr)>> parse_attr_funcs = {
  {kParamType, ParseAttrParamType},
  {kType, ParseAttrType},
  {kDefaultValue, ParseAttrDefaultValue},
  {kValue, ParseAttrValue}};

bool ParseIODtype(const std::string &value, const OpIOInfoPtr &op_io_info_ptr) {
  std::vector<std::string> dtypes = SplitStrToVec(value);
  op_io_info_ptr->set_dtypes(dtypes);
  return true;
}

bool ParseIOFormat(const std::string &value, const OpIOInfoPtr &op_io_info_ptr) {
  std::vector<std::string> formats = SplitStrToVec(value);
  op_io_info_ptr->set_formats(formats);
  return true;
}

bool ParseIOName(const std::string &value, const OpIOInfoPtr &op_io_info_ptr) {
  op_io_info_ptr->set_name(value);
  return true;
}

bool ParseIOParamType(const std::string &value, const OpIOInfoPtr &op_io_info_ptr) {
  op_io_info_ptr->set_param_type(value);
  return true;
}

bool ParseIOUnKnownShapeFormat(const std::string &value, const OpIOInfoPtr &op_io_info_ptr) {
  std::vector<std::string> formats = SplitStrToVec(value);
  op_io_info_ptr->set_unknown_shape_formats(formats);
  return true;
}

bool ParseIOReshapeType(const std::string &value, const OpIOInfoPtr &op_io_info_ptr) {
  op_io_info_ptr->set_reshape_type(value);
  return true;
}

bool ParseIOShape(const std::string &value, const OpIOInfoPtr &op_io_info_ptr) {
  op_io_info_ptr->set_shape(value);
  return true;
}

bool ParseIONeedCompile(const std::string &value, const OpIOInfoPtr &op_io_info_ptr) {
  op_io_info_ptr->set_need_compile(value == kTrue);
  return true;
}

bool ParseIOValueDepend(const std::string &value, const OpIOInfoPtr &op_io_info_ptr) {
  op_io_info_ptr->set_value_depend(value);
  return true;
}

bool ParseIOShapesType(const std::string &value, const OpIOInfoPtr &op_io_info_ptr) {
  op_io_info_ptr->set_shapes_type(value);
  return true;
}

std::map<std::string, std::function<bool(const std::string &value, const OpIOInfoPtr &op_io_info_ptr)>>
  parse_io_info_funcs = {{kDtype, ParseIODtype},
                         {kFormat, ParseIOFormat},
                         {kName, ParseIOName},
                         {kParamType, ParseIOParamType},
                         {kUnknownShapeFormat, ParseIOUnKnownShapeFormat},
                         {kReshapeType, ParseIOReshapeType},
                         {kShape, ParseIOShape},
                         {kNeedCompile, ParseIONeedCompile},
                         {kValueDepend, ParseIOValueDepend},
                         {kShapesType, ParseIOShapesType}};
}  // namespace

bool OpInfoUtils::ParseCommonItem(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  for (auto iter = item.begin(); iter != item.end(); ++iter) {
    const std::string &key = iter.key();
    if (key.find(kInput) != std::string::npos || key.find(kOutput) != std::string::npos ||
        key.find(kAttr) != std::string::npos) {
      continue;
    }
    auto func = parse_common_funcs.find(key);
    if (func == parse_common_funcs.end()) {
      MS_LOG(WARNING) << "Parse common key: " << iter.key() << " not found, json: " << item.dump();
      continue;
    }
    if (!func->second(*iter, op_info_ptr)) {
      MS_LOG(ERROR) << "Parse common failed: " << iter->dump();
      return false;
    }
  }
  return true;
}

bool OpInfoUtils::ParseAttrs(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  nlohmann::json attr;
  if (!GetJsonSubItem(item, kAttr, &attr)) {
    return true;
  }
  std::string attr_name_value;
  if (!GetJsonValue(attr, kList, &attr_name_value)) {
    MS_LOG(ERROR) << "Parse attr failed, attr: " << attr.dump();
    return false;
  }
  std::vector<std::string> attr_names = SplitStrToVec(attr_name_value);
  std::string attr_name_prefix = "attr_";
  for (auto &name : attr_names) {
    auto attr_name = attr_name_prefix + name;
    nlohmann::json attr_suffix;
    if (!GetJsonSubItem(item, attr_name, &attr_suffix)) {
      MS_LOG(ERROR) << "Parse attr failed, attr: " << attr.dump();
      return false;
    }
    OpAttrPtr op_attr_ptr = nullptr;
    MAKE_SHARE_CHECK(op_attr_ptr = std::make_shared<OpAttr>())
    op_attr_ptr->set_name(name);
    for (auto iter_attr = attr_suffix.begin(); iter_attr != attr_suffix.end(); iter_attr++) {
      auto func = parse_attr_funcs.find(iter_attr.key());
      if (func == parse_attr_funcs.end()) {
        MS_LOG(WARNING) << "Parse attr key: " << iter_attr.key() << " not found, json: " << attr_suffix.dump();
        continue;
      }
      if (!func->second(attr_suffix.at(iter_attr.key()), op_attr_ptr)) {
        MS_LOG(ERROR) << "Parse attr failed, attr: " << attr.dump() << ", item: " << iter_attr->dump();
        return false;
      }
    }
    op_info_ptr->add_attrs_ptr(op_attr_ptr);
  }

  return true;
}

bool OpInfoUtils::ParseOpIOInfo(const nlohmann::json &item, const OpInfoPtr &op_info_ptr) {
  // parse input
  if (!ParseOpIOInfoImpl(item, true, op_info_ptr)) {
    MS_LOG(ERROR) << "Parse input failed, js: " << item.dump();
    return false;
  }
  // parse output
  if (!ParseOpIOInfoImpl(item, false, op_info_ptr)) {
    MS_LOG(ERROR) << "Parse output failed, js: " << item.dump();
    return false;
  }
  // Updata input orders
  UpdateInputOrders(op_info_ptr);
  // update ref map
  UpdateRefInfo(op_info_ptr);
  return true;
}

bool OpInfoUtils::ParseOpIOInfoImpl(const nlohmann::json &item, bool is_input, const OpInfoPtr &op_info_ptr) {
  int index = 0;
  std::string prefix = is_input ? kInput : kOutput;
  std::string key = prefix + std::to_string(index);
  nlohmann::json ret;
  // input
  while (GetJsonSubItem(item, key, &ret)) {
    OpIOInfoPtr op_io_info_ptr = nullptr;
    MAKE_SHARE_CHECK(op_io_info_ptr = std::make_shared<OpIOInfo>())
    for (auto iter = ret.begin(); iter != ret.end(); iter++) {
      auto func = parse_io_info_funcs.find(iter.key());
      if (func == parse_io_info_funcs.end()) {
        MS_LOG(WARNING) << "Parse IO key: " << iter.key() << " not found, json: " << ret.dump();
        continue;
      }
      if (!func->second(ret.at(iter.key()), op_io_info_ptr)) {
        MS_LOG(ERROR) << "Parse input/output failed, item: " << iter->dump();
        return false;
      }
    }
    op_io_info_ptr->set_index(index);
    if (is_input) {
      op_info_ptr->add_inputs_ptr(op_io_info_ptr);
    } else {
      op_info_ptr->add_outputs_ptr(op_io_info_ptr);
    }
    key = prefix + std::to_string(++index);
  }
  return true;
}

void OpInfoUtils::UpdateInputOrders(const OpInfoPtr &op_info_ptr) {
  auto op_name = op_info_ptr->op_name();
  auto orders = SuperBar::GetGraphIdxToKernelIdx(op_name);
  if (!orders.has_value()) {
    return;
  }
  auto graph_idx_to_kernel_idx = orders.value();
  std::vector<std::shared_ptr<OpIOInfo>> tmp = op_info_ptr->inputs_ptr();
  if (tmp.size() != graph_idx_to_kernel_idx.size()) {
    MS_LOG(EXCEPTION) << "Op input size not equal order size, (" << tmp.size() << ", " << graph_idx_to_kernel_idx.size()
                      << ")";
  }
  for (size_t i = 0; i < graph_idx_to_kernel_idx.size(); ++i) {
    tmp[i] = op_info_ptr->inputs_ptr().at(graph_idx_to_kernel_idx[i]);
    tmp[i]->set_index(SizeToInt(i));
  }
  op_info_ptr->set_inputs_ptr(tmp);
}

void OpInfoUtils::UpdateRefInfo(const OpInfoPtr &op_info_ptr) {
  auto op_io_info_inputs = op_info_ptr->inputs_ptr();
  auto op_io_info_outputs = op_info_ptr->outputs_ptr();
  for (const auto &input : op_io_info_inputs) {
    for (const auto &output : op_io_info_outputs) {
      if (input->name() == output->name()) {
        op_info_ptr->add_ref_pair(output->index(), input->index());
      }
    }
  }
}

void OpInfoUtils::SetKernelName(const OpInfoPtr &op_info_ptr) {
  std::string kernel_name = op_info_ptr->op_file();
  if (kernel_name.empty() || kernel_name == kNull) {
    kernel_name = NormalizeKernelName(op_info_ptr->op_name());
  }
  op_info_ptr->set_kernel(kernel_name);
}

std::string OpInfoUtils::NormalizeKernelName(const std::string &op_name) {
  std::string tmp;
  static const std::map<std::string, std::string> func_names = {{"ApplyAdagradDAD", "apply_adagrad_da_d"}};
  auto find_iter = func_names.find(op_name);
  if (find_iter != func_names.end()) {
    return find_iter->second;
  }
  bool head = false;
  for (auto iter = op_name.begin(); iter != op_name.end(); ++iter) {
    if (std::islower(*iter)) {
      head = false;
    }
    if (std::isdigit(*iter)) {
      head = true;
    }
    if (std::isupper(*iter) && iter != op_name.begin()) {
      if (!head) {
        tmp.insert(tmp.end(), '_');
        head = true;
      } else {
        if (iter + 1 != op_name.end()) {
          if (std::islower(*(iter + 1))) {
            tmp.insert(tmp.end(), '_');
          }
        }
      }
    }
    tmp.insert(tmp.end(), *iter);
  }
  std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::tolower);
  return tmp;
}
}  // namespace mindspore::kernel
