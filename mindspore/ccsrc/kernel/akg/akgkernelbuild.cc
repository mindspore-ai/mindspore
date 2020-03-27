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

#include "kernel/akg/akgkernelbuild.h"
#include <Python.h>
#include <sys/types.h>
#include <signal.h>
#include <unistd.h>
#include <dirent.h>
#include <cctype>
#include <cstdint>
#include <memory>
#include <map>
#include <utility>
#include <algorithm>
#include <functional>
#include <sstream>
#include <iterator>
#include <numeric>
#include <unordered_set>
#include "common/utils.h"
#include "utils/convert_utils.h"
#include "utils/any.h"
#include "utils/utils.h"
#include "transform/convert.h"
#include "session/anf_runtime_algorithm.h"
#include "kernel/akg/akg_kernel_attrs_process.h"

namespace mindspore {
namespace kernel {
constexpr int ME_MAX_KERNEL_NAME_LENGTH = 200;
constexpr int32_t ARGS_SIZE = 1;
constexpr auto kCompileWithJsonFunc = "compilewithjson";
// json key
constexpr auto kInputDesc = "input_desc";
constexpr auto kShape = "shape";
constexpr auto kDataType = "data_type";
constexpr auto kOutputDesc = "output_desc";
constexpr auto kName = "name";
constexpr auto kTensorName = "tensor_name";
constexpr auto kValue = "value";
constexpr auto KInpputNames = "input_names";
constexpr auto KInput = "input";
constexpr auto KDtype = "dtype";
int AkgKernelBuild::op_cnt_ = 0;
std::mutex AkgKernelBuild::op_cnt_mtx_;

std::string PyObjectToStr(PyObject *const PyObj) {
  char *pChar = nullptr;
  std::string str_res;
  if (PyObj == nullptr) {
    MS_LOG(ERROR) << "Input parameter is nullptr.";
    return str_res;
  }
  PyObject *strArgs = PyObject_Str(PyObj);
  if (strArgs != nullptr) {
    (void)PyArg_Parse(strArgs, "s", &pChar);
  }
  if (pChar == nullptr) {
    MS_LOG(ERROR) << "pChar is nullptr.";
    return str_res;
  }
  str_res = pChar;
  return str_res;
}

std::string AkgKernelBuild::GetProcessor(const AnfNodePtr &anf_node) {
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
      MS_LOG(ERROR) << "Unknown processor type.";
      break;
  }

  return device;
}

bool GetIOSize(const nlohmann::json &node_json, std::vector<size_t> *const input_size,
               std::vector<size_t> *const output_size) {
  if (input_size == nullptr || output_size == nullptr) {
    MS_LOG(ERROR) << "input size or output size is nullptr";
    return false;
  }
  input_size->clear();
  output_size->clear();

  for (size_t i = 0; i < node_json[kInputDesc].size(); i++) {
    for (size_t m = 0; m < node_json[kInputDesc][i].size(); m++) {
      std::string dtype = node_json[kInputDesc][i][m][kDataType];
      size_t nbyte = GetDtypeNbyte(dtype);
      size_t size_i = std::accumulate(node_json[kInputDesc][i][m][kShape].begin(),
                                      node_json[kInputDesc][i][m][kShape].end(), nbyte, std::multiplies<size_t>());
      input_size->push_back(size_i);
    }
  }

  for (size_t i = 0; i < node_json[kOutputDesc].size(); i++) {
    std::string dtype = node_json[kOutputDesc][i][kDataType];
    size_t nbyte = GetDtypeNbyte(dtype);
    size_t size_i = std::accumulate(node_json[kOutputDesc][i][kShape].begin(), node_json[kOutputDesc][i][kShape].end(),
                                    nbyte, std::multiplies<size_t>());
    output_size->push_back(size_i);
  }

  return true;
}

int AkgKernelBuild::GetOpCntInc() {
  op_cnt_mtx_.lock();
  int cnt = op_cnt_++;
  op_cnt_mtx_.unlock();
  return cnt;
}

bool AkgKernelBuild::CreateInputDescJson(const AnfNodePtr &anf_node, nlohmann::json *const inputs_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(inputs_json);

  // for dynamic input number, dyn_input_sizes has the info of dynamic input num for each input.
  std::string op_name = AnfAlgo::GetCNodeName(anf_node);
  auto op_info = mindspore::kernel::OpLib::FindOp(op_name, OpImplyType::kAKG);
  if (op_info == nullptr) {
    MS_LOG(ERROR) << "Apply kernel [" << op_name << "] op_info is nullptr";
    return false;
  }

  std::vector<std::shared_ptr<OpIOInfo>> inputs_ptr = op_info->inputs_ptr();
  if (inputs_ptr.empty()) {
    MS_LOG(INFO) << "Apply kernel [" << op_name << "] regist info has no input info";
    return true;
  }
  auto op_info_input_num = inputs_ptr.size();

  // for dynamic input number, dyn_input_sizes has the info of dynamic input num for each input.
  std::vector<int> dyn_input_sizes;
  auto primitive = AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);

  if (primitive->GetAttr(kAttrDynInputSizes) != nullptr) {
    dyn_input_sizes = GetValue<const std::vector<int>>(primitive->GetAttr(kAttrDynInputSizes));
  }

  size_t real_input_index = 0;
  std::vector<nlohmann::json> input_list;
  for (size_t i = 0; i < op_info_input_num; i++) {
    size_t input_tensor_num;
    std::shared_ptr<OpIOInfo> input_ptr = inputs_ptr[i];
    std::string op_input_name;
    if (input_ptr == nullptr) {
      MS_LOG(ERROR) << "Apply kernel [" << op_name << "] regist input[" << i << "] is nullptr";
      return false;
    }

    op_input_name = input_ptr->name();
    if (dyn_input_sizes.empty()) {
      input_tensor_num = 1;
    } else {
      input_tensor_num = IntToSize(dyn_input_sizes[i]);
    }

    input_list.clear();
    for (size_t input_i = 0; input_i < input_tensor_num; input_i++) {
      // dtype : float16
      auto type_id = AnfAlgo::GetInputDeviceDataType(anf_node, real_input_index);
      TypePtr type_ptr = TypeIdToType(type_id);
      MS_EXCEPTION_IF_NULL(type_ptr);
      std::string dtype = type_ptr->ToString();
      dtype = Dtype2String(dtype);
      if (dtype.empty()) {
        MS_LOG(ERROR) << "Op [" << op_name << "] input [" << input_i << "] data type is null. ";
        return false;
      }
      nlohmann::json input_desc_json;
      input_desc_json[kDataType] = dtype;
      input_desc_json[kName] = op_input_name;
      input_desc_json[kTensorName] =
        op_input_name + "_" + std::to_string(real_input_index) + "_" + std::to_string(input_i);
      input_desc_json[kShape] = AnfAlgo::GetInputDeviceShape(anf_node, real_input_index);
      input_list.emplace_back(input_desc_json);
    }
    inputs_json->emplace_back(input_list);
    real_input_index++;
  }
  return true;
}

bool AkgKernelBuild::CreateOutputDescJson(const AnfNodePtr &anf_node, nlohmann::json *const outputs_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(outputs_json);
  size_t output_tensor_num = AnfAlgo::GetOutputTensorNum(anf_node);
  std::string op_name = AnfAlgo::GetCNodeName(anf_node);

  auto op_info_ptr = mindspore::kernel::OpLib::FindOp(op_name, OpImplyType::kAKG);
  auto outputs = op_info_ptr->outputs_ptr();
  for (size_t i = 0; i < output_tensor_num; i++) {
    nlohmann::json output_json;
    auto type_id = AnfAlgo::GetOutputDeviceDataType(anf_node, i);
    TypePtr type_ptr = TypeIdToType(type_id);
    MS_EXCEPTION_IF_NULL(type_ptr);
    std::string dtype = type_ptr->ToString();
    dtype = Dtype2String(dtype);
    if (dtype.empty()) {
      MS_LOG(ERROR) << "Op [" << op_name << "] output [" << i << "] data type is null. ";
      return false;
    }

    std::string output_name = outputs[i]->name();
    output_json[kDataType] = dtype;
    output_json[kName] = output_name;
    output_json[kTensorName] = output_name + "_" + std::to_string(i);
    output_json[kShape] = AnfAlgo::GetOutputDeviceShape(anf_node, i);
    outputs_json->push_back(output_json);
  }
  return true;
}

void GetJson(const AnfNodePtr &anf_node, const vector<int> &dyn_input_sizes, const shared_ptr<OpAttr> &op_attr,
             nlohmann::json *const attr_json, const ValuePtr &attr_value) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(op_attr);
  MS_EXCEPTION_IF_NULL(attr_json);
  std::string type = op_attr->type();
  if (type == "int") {
    (*attr_json)[kValue] = GetValue<int>(attr_value);
  } else if (type == "str") {
    (*attr_json)[kValue] = GetValue<std::string>(attr_value);
  } else if (type == "bool") {
    (*attr_json)[kValue] = GetValue<bool>(attr_value);
  } else if (type == "float") {
    (*attr_json)[kValue] = GetValue<float>(attr_value);
  } else if (type == "listInt") {
    (*attr_json)[kValue] = GetValue<std::vector<int>>(attr_value);
  } else if (type == "listStr") {
    std::vector<std::string> data_format;
    if (op_attr->name() == kArgDataformat) {
      size_t tensor_args_num = !dyn_input_sizes.empty() ? dyn_input_sizes.size() : AnfAlgo::GetInputTensorNum(anf_node);
      for (size_t format_i = 0; format_i < tensor_args_num; format_i++) {
        auto input_format = AnfAlgo::GetInputFormat(anf_node, format_i);
        data_format.push_back(input_format);
      }
    } else {
      data_format = GetValue<std::vector<std::string>>(attr_value);
    }
    (*attr_json)[kValue] = data_format;
  } else {
    MS_LOG(WARNING) << "attr type:" << type;
  }
}

bool AkgKernelBuild::CreateAttrDescJson(const AnfNodePtr &anf_node, const std::string &op_name,
                                        const std::shared_ptr<OpInfo> &op_info, nlohmann::json *const attrs_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(attrs_json);
  MS_EXCEPTION_IF_NULL(op_info);
  std::vector<std::shared_ptr<OpAttr>> attrs = op_info->attrs_ptr();
  if (attrs.empty()) {
    MS_LOG(INFO) << "Apply kernel [" << op_name << "] op info attrs is empty";
    return true;
  }
  std::vector<std::shared_ptr<OpIOInfo>> inputs = op_info->inputs_ptr();

  std::vector<int> dyn_input_sizes;
  auto primitive = AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  if (primitive->GetAttr(kAttrDynInputSizes) != nullptr) {
    dyn_input_sizes = GetValue<const std::vector<int>>(primitive->GetAttr(kAttrDynInputSizes));
  }

  if (inputs.empty()) {
    MS_LOG(ERROR) << "Apply kernel [" << op_name << "] op info inputs is empty";
    return false;
  }

  // create input name list for atch "x_shape" in att with "x" in primitive.
  std::map<size_t, std::string> op_info_shape_name;
  for (size_t op_info_input_i = 0; op_info_input_i < inputs.size(); op_info_input_i++) {
    std::string input_name = inputs[op_info_input_i]->name();
    std::string x_shape_name = input_name + "_shape";
    (void)op_info_shape_name.insert(make_pair(op_info_input_i, x_shape_name));
  }

  for (const auto &op_attr : attrs) {
    nlohmann::json attr_json;
    ValuePtr attr_value = primitive->GetAttr(op_attr->name());
    if (attr_value == nullptr && op_attr->name() != kArgDataformat) {
      if (op_attr->param_type() == "required") {
        // match "x_shape" in att with "x" in primitive.
        std::string attr_name = op_attr->name();
        auto find_item = std::find_if(
          op_info_shape_name.begin(), op_info_shape_name.end(),
          [attr_name](const std::map<size_t, std::string>::value_type item) { return item.second == attr_name; });
        if (find_item != op_info_shape_name.end()) {
          if (!dyn_input_sizes.empty()) {
            if (find_item->first >= dyn_input_sizes.size() - 1) {
              MS_LOG(EXCEPTION) << "dyn_input_sizes list index:" << find_item->first
                                << " is out of range:" << dyn_input_sizes.size() - 1 << ".";
              return false;
            }
            size_t tensor_idx = IntToSize(std::accumulate(&dyn_input_sizes[0], &dyn_input_sizes[find_item->first], 0));
            for (int input_i = 0; input_i < dyn_input_sizes[find_item->first]; input_i++) {
              attr_json[kValue] = AnfAlgo::GetPrevNodeOutputInferShape(anf_node, tensor_idx);
              attr_json[kName] = op_attr->name();
              attrs_json->push_back(attr_json);
              tensor_idx++;
            }
          } else {
            attr_json[kValue] = AnfAlgo::GetPrevNodeOutputInferShape(anf_node, find_item->first);
            attr_json[kName] = op_attr->name();
            attrs_json->push_back(attr_json);
          }
        } else {
          MS_LOG(ERROR) << "op [" << op_name << "] should have attr :" << op_attr->name();
          return false;
        }
      }
      continue;
    }

    GetJson(anf_node, dyn_input_sizes, op_attr, &attr_json, attr_value);

    attr_json[kName] = op_attr->name();
    attrs_json->push_back(attr_json);
  }
  return true;
}

bool AkgKernelBuild::GenerateSingleKernelJson(const AnfNodePtr &anf_node, const std::string &op_name,
                                              nlohmann::json *const node_json) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(node_json);
  int op_cnt = GetOpCntInc();
  auto op_info_ptr = mindspore::kernel::OpLib::FindOp(op_name, OpImplyType::kAKG);
  MS_EXCEPTION_IF_NULL(op_info_ptr);

  // get basic params from currentNodeOpDesc
  (*node_json)["platform"] = "AKG";
  (*node_json)[kName] = op_name;
  (*node_json)["fusion_type"] = AnfAlgo::GetFusionType(anf_node);
  (*node_json)["impl_path"] = op_info_ptr->impl_path();
  (*node_json)["process"] = AkgKernelBuild::GetProcessor(anf_node);

  auto primitive = AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  ValuePtr input_names_v = primitive->GetAttr(KInpputNames);
  if (input_names_v == nullptr) {
    MS_LOG(ERROR) << "ApplyKernel has no input_names, op[" << op_name << "].";
    return false;
  }
  std::vector<std::string> prim_input_names = GetValue<const std::vector<std::string>>(input_names_v);
  std::string inputs_name;
  for (const auto &prim_input_name : prim_input_names) {
    (void)inputs_name.append("_input_").append(prim_input_name).append("_");
  }

  // input desc
  nlohmann::json inputs_json;
  if (!CreateInputDescJson(anf_node, &inputs_json)) {
    MS_LOG(ERROR) << "Create input desc json failed, op[" << op_name << "].";
    return false;
  }
  (*node_json)[kInputDesc] = inputs_json;
  MS_LOG(INFO) << "Akg create input desc json success.";
  std::string inputs_shape = "inputs_shape_";
  for (auto &i : inputs_json) {
    for (auto &m : i) {
      std::string data_type = m[kDataType];
      (void)inputs_shape.append("_").append(data_type).append("_");
      for (auto &j : m[kShape]) {
        size_t n = j;
        (void)inputs_shape.append(std::to_string(n)).append("_");
      }
    }
  }

  // output desc
  nlohmann::json outputs_json;
  if (!CreateOutputDescJson(anf_node, &outputs_json)) {
    MS_LOG(ERROR) << "Create output desc json failed, op[" << op_name << "].";
    return false;
  }

  (*node_json)[kOutputDesc] = outputs_json;
  MS_LOG(INFO) << "Akg create output desc json success.";
  std::string outputs_shape = "outputs_shape_";
  for (auto &i : outputs_json) {
    std::string data_type = i[kDataType];
    (void)outputs_shape.append("_").append(data_type).append("_");
    for (auto &j : i[kShape]) {
      size_t m = j;
      (void)outputs_shape.append(std::to_string(m)).append("_");
    }
  }

  // attribute desc
  nlohmann::json attrs_json;
  if (!CreateAttrDescJson(anf_node, op_name, op_info_ptr, &attrs_json)) {
    MS_LOG(ERROR) << "Create attr desc json failed, op[" << op_name << "].";
    return false;
  }
  (*node_json)["attr"] = attrs_json;
  std::string json_str = node_json->dump();
  size_t hash_id = std::hash<std::string>()(json_str);
  json_name_ = op_name + "_";
  (void)json_name_.append(std::to_string(hash_id));
  MS_LOG(INFO) << "full scope name is : " << anf_node->fullname_with_scope() << ", json info name is : " << json_name_;
  json_info_ = json_str;
  (*node_json)["id"] = op_cnt;
  (*node_json)["op"] = json_name_;
  MS_LOG(INFO) << "Akg create node desc json success.";
  return true;
}

KernelPackPtr AkgKernelBuild::OpBuild(const std::string &node_json, const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto processor = AkgKernelBuild::GetProcessor(anf_node);
  auto cached_kernel_pack = SearchCache(json_name_, processor);
  if (cached_kernel_pack != nullptr) {
    MS_LOG(INFO) << "Use cached kernel, json_name_[" << json_name_ << "], fullname_with_scope["
                 << anf_node->fullname_with_scope() << "].";
    return cached_kernel_pack;
  }

  PyObject *pModule = nullptr;
  PyObject *pFunc = nullptr;
  PyObject *pArg = nullptr;
  PyObject *pRes = nullptr;

  pModule = PyImport_ImportModule(kAkgModule);
  if (pModule == nullptr) {
    MS_LOG(ERROR) << "Failed to import [" << kAkgModule << "].";
    return nullptr;
  }

  pFunc = PyObject_GetAttrString(pModule, kCompileWithJsonFunc);
  pArg = PyTuple_New(ARGS_SIZE);
  (void)PyTuple_SetItem(pArg, 0, Py_BuildValue("s", node_json.c_str()));

  (void)alarm(AUTODIFF_COMPILE_OVERTIME);
  pRes = PyEval_CallObject(pFunc, pArg);
  (void)alarm(0);
  if (pRes == nullptr) {
    MS_LOG(ERROR) << "No ret got, failed to call function [" << kCompileWithJsonFunc << "], args:\n("
                  << PyObjectToStr(pArg) << ").";
    return nullptr;
  }
  if (PyObject_IsTrue(pRes) != 1) {
    MS_LOG(ERROR) << "Illegal ret, failed to call function [" << kCompileWithJsonFunc << "], args:\n("
                  << PyObjectToStr(pArg) << ").";
    return nullptr;
  }

  auto new_kernel_pack = InsertCache(json_name_, processor);
  kernel::SaveJsonInfo(json_name_, json_info_);
  if (new_kernel_pack == nullptr) {
    MS_LOG(ERROR) << "Insert to cache failed, json_name_[" << json_name_ << "], fullname_with_scope["
                  << anf_node->fullname_with_scope() << "].";
    return nullptr;
  }
  return new_kernel_pack;
}

KernelPackPtr AkgKernelBuild::BuildByJson(const AnfNodePtr &anf_node, std::vector<size_t> *const input_size,
                                          std::vector<size_t> *const output_size) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::string op_name = AnfAlgo::GetCNodeName(anf_node);
  auto it = kAkgKernelAttrsProcessMap.find(op_name);
  if (it != kAkgKernelAttrsProcessMap.end()) {
    it->second(anf_node);
  }
  MS_LOG(INFO) << "Akg start compile, op[" << op_name << "], device[" << AkgKernelBuild::GetProcessor(anf_node) << "]";
  nlohmann::json node_json;
  if (!GenerateSingleKernelJson(anf_node, op_name, &node_json)) {
    MS_LOG(ERROR) << "Op[" << op_name << "] create single kernel json failed.";
  }

  std::string json_str = node_json.dump();
  auto kernel_pack = OpBuild(json_str, anf_node);
  if (kernel_pack == nullptr) {
    MS_LOG(ERROR) << "Akg build failed op[" << op_name << "], json:" << json_str;
    return nullptr;
  }

  if (!GetIOSize(node_json, input_size, output_size)) {
    MS_LOG(ERROR) << "Cal mem size failed.";
    return nullptr;
  }
  MS_LOG(INFO) << "Akg compile success, op[" << op_name << "], device[" << AkgKernelBuild::GetProcessor(anf_node)
               << "]";
  return kernel_pack;
}
}  // namespace kernel
}  // namespace mindspore
