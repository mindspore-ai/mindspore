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

#include "kernel/tbe/tbe_kernel_select.h"

#include <unordered_map>
#include <memory>
#include <map>

#include "session/anf_runtime_algorithm.h"
#include "kernel/oplib/oplib.h"
#include "kernel/tbe/tbe_kernel_build.h"
#include "nlohmann/json.hpp"
#include "common/utils.h"
#include "utils/context/ms_context.h"
#include "kernel/tbe/tbe_python_funcs.h"
#include "pre_activate/common/helper.h"
#include "kernel/tbe/tbe_convert_utils.h"

namespace mindspore {
namespace kernel {
constexpr auto kName = "name";
constexpr auto kDtype = "dtype";
constexpr auto kFormat = "format";
constexpr auto kPrefixInput = "input";
constexpr auto kPrefixOutput = "output";
const std::map<std::string, std::string> DYNAMIC_FORMAT_MAP = {{"NCHW", "DefaultFormat"},
                                                               {"NHWC", "DefaultFormat"},
                                                               {"ND", "DefaultFormat"},
                                                               {"FRACTAL_Z", "FracZ"},
                                                               {"NDHWC", "DefaultFormat"}};
static const std::vector<std::string> CHECK_SUPPORTED_OPTYPE{
  "MatMul", "BatchMatMul", "TopK", "InTopK", "Pack", "GatherNd", "UnsortedSegmentMinD", "UnsortedSegmentProdD", "Cast"};

bool CheckSupported(const AnfNodePtr &anf_node, const KernelBuildInfoPtr &select_kernel_build_info) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(select_kernel_build_info);

  std::string op_name = AnfAlgo::GetCNodeName(anf_node);
  auto iter = std::find(CHECK_SUPPORTED_OPTYPE.begin(), CHECK_SUPPORTED_OPTYPE.end(), op_name);
  if (iter == CHECK_SUPPORTED_OPTYPE.end()) {
    MS_LOG(DEBUG) << "Op " << op_name << "this op does not need to check op supported.";
    return true;
  }

  // replace kernel_info with current kernel info
  auto ori_select_kernel_info = AnfAlgo::GetSelectKernelBuildInfo(anf_node);
  AnfAlgo::SetSelectKernelBuildInfo(select_kernel_build_info, anf_node.get());

  nlohmann::json kernel_json;
  TbeKernelJsonCreator creator(CHECK_SUPPORTED);
  bool ret = creator.GenTbeSingleKernelJson(anf_node, &kernel_json);
  if (!ret) {
    MS_LOG(DEBUG) << "GenTbeSingleKernelJson failed";
    AnfAlgo::SetSelectKernelBuildInfo(ori_select_kernel_info, anf_node.get());
    return false;
  }

  ret = TbePythonFuncs::CheckSupported(kernel_json);
  AnfAlgo::SetSelectKernelBuildInfo(ori_select_kernel_info, anf_node.get());
  return ret;
}

bool CheckJsonItemValidity(const nlohmann::json &json_obj, const std::string &key_name,
                           const std::vector<std::string> &keys) {
  if (!json_obj[key_name].is_object()) {
    MS_LOG(DEBUG) << key_name << "is not an object!";
    return false;
  }
  for (auto key : keys) {
    if (json_obj[key_name].find(key) == json_obj[key_name].end()) {
      MS_LOG(DEBUG) << "Key" << key << "of " << key_name << " is not found!";
      return false;
    }
  }
  return true;
}

std::vector<std::string> SplitStr(const std::string &string, const std::string &sep) {
  std::vector<std::string> result;
  size_t start = 0;
  size_t index = string.find(sep, start);
  std::string substr;
  while (index != std::string::npos) {
    if (string.size() > start) {
      substr = string.substr(start, index - start);
    }
    (void)substr.erase(0, substr.find_first_not_of(' '));
    (void)substr.erase(substr.find_last_not_of(" ") + 1);
    auto iter = DYNAMIC_FORMAT_MAP.find(substr);
    if (iter != DYNAMIC_FORMAT_MAP.end()) {
      substr = iter->second;
    }
    result.push_back(substr);
    start = index + sep.size();
    index = string.find(sep, start);
  }

  if (string.size() > start) {
    substr = string.substr(start);
  }
  (void)substr.erase(0, substr.find_first_not_of(" "));
  (void)substr.erase(substr.find_last_not_of(" ") + 1);
  auto iter = DYNAMIC_FORMAT_MAP.find(substr);
  if (iter != DYNAMIC_FORMAT_MAP.end()) {
    substr = iter->second;
  }
  result.push_back(substr);
  return result;
}

void ConvertFormatDtype(const std::string &format, const std::string &dtype, const std::shared_ptr<OpIOInfo> io_info) {
  MS_EXCEPTION_IF_NULL(io_info);
  std::vector<std::string> format_vec = SplitStr(format, ",");
  std::vector<std::string> dtype_vec = SplitStr(dtype, ",");
  io_info->set_formats(format_vec);
  io_info->set_dtypes(dtype_vec);
}

bool ParseDynamicFormatJson(const std::string &jsonStr, std::vector<std::shared_ptr<OpIOInfo>> *const inputs,
                            std::vector<std::shared_ptr<OpIOInfo>> *const outputs) {
  nlohmann::json json_obj = nlohmann::json::parse(jsonStr);
  if (!json_obj.is_object()) {
    MS_LOG(DEBUG) << "JsonStr is not an object, the jsonStr is:" << jsonStr;
    return false;
  }
  std::vector<std::string> keys = {kName, kDtype, kFormat};
  for (const auto &item : json_obj.items()) {
    std::string key_name;
    key_name = item.key();
    if (key_name.empty()) {
      MS_LOG(DEBUG) << "Key name is empty!";
      return false;
    }
    if (!CheckJsonItemValidity(json_obj, key_name, keys)) {
      return false;
    }
    if (key_name.compare(0, strlen(kPrefixInput), kPrefixInput) == 0) {
      std::shared_ptr<OpIOInfo> input = std::make_shared<OpIOInfo>();
      MS_EXCEPTION_IF_NULL(input);
      input->set_name(json_obj[key_name].at(kName));
      ConvertFormatDtype(json_obj[key_name].at(kFormat), json_obj[key_name].at(kDtype), input);
      inputs->emplace_back(input);
    } else if (key_name.compare(0, strlen(kPrefixOutput), kPrefixOutput) == 0) {
      std::shared_ptr<OpIOInfo> output = std::make_shared<OpIOInfo>();
      MS_EXCEPTION_IF_NULL(output);
      output->set_name(json_obj[key_name].at(kName));
      ConvertFormatDtype(json_obj[key_name].at(kFormat), json_obj[key_name].at(kDtype), output);
      outputs->emplace_back(output);
    } else {
      MS_LOG(DEBUG) << "Key name:" << key_name << " is undefined!";
      return false;
    }
  }
  return true;
}

std::string OpSelectFormat(const std::shared_ptr<AnfNode> &anf_node) {
  nlohmann::json kernel_json;
  std::string res_json_str;
  TbeKernelJsonCreator creator(OP_SELECT_FORMAT);
  bool ret = creator.GenTbeSingleKernelJson(anf_node, &kernel_json);
  if (!ret) {
    MS_LOG(DEBUG) << "GenTbeSingleKernelJson failed";
    return res_json_str;
  }
  res_json_str = TbePythonFuncs::OpSelectFormat(kernel_json);
  MS_LOG(INFO) << "Dynamic select foramt response result:" << res_json_str;
  return res_json_str;
}

void SetTidyInputsInfo(const std::shared_ptr<AnfNode> &anf_node,
                       const std::shared_ptr<KernelBuildInfo::KernelBuildInfoBuilder> &builder,
                       const std::vector<std::shared_ptr<OpIOInfo>> &inputs) {
  std::vector<TypeId> inputs_type;
  std::vector<std::string> inputs_format;
  std::vector<int> dyn_input_sizes;
  size_t dyn_input_idx = 0;
  size_t kernel_info_index = 0;
  size_t real_input_num = AnfAlgo::GetInputTensorNum(anf_node);
  auto primitive = AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  if (primitive->GetAttr("dyn_input_sizes") != nullptr) {
    dyn_input_sizes = GetValue<std::vector<int>>(primitive->GetAttr("dyn_input_sizes"));
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    MS_EXCEPTION_IF_NULL(inputs[i]);
    std::string param_type = inputs[i]->param_type();
    if (i >= real_input_num) {
      MS_LOG(INFO) << "Input index:" << i << "is out of real_input_num:" << real_input_num;
      continue;
    }
    auto type_id = AnfAlgo::GetPrevNodeOutputInferDataType(anf_node, i);
    auto format = kOpFormat_DEFAULT;
    if (param_type == "dynamic") {
      if (!dyn_input_sizes.empty()) {
        for (int t = 0; t < dyn_input_sizes[dyn_input_idx]; t++) {
          kernel_info_index++;
          inputs_type.emplace_back(type_id);
          inputs_format.emplace_back(format);
        }
        dyn_input_idx++;
      }
    } else if (param_type == "required") {
      kernel_info_index++;
      inputs_type.emplace_back(type_id);
      inputs_format.emplace_back(format);
    } else {
      if (kernel_info_index < real_input_num) {
        MS_LOG(INFO) << "Input type is optional, input index is :" << kernel_info_index;
        kernel_info_index++;
        inputs_type.emplace_back(type_id);
        inputs_format.emplace_back(format);
      }
    }
  }
  builder->SetInputsDeviceType(inputs_type);
  builder->SetInputsFormat(inputs_format);
}

void SetTidyOutputsInfo(const std::shared_ptr<AnfNode> &anf_node,
                        const std::shared_ptr<KernelBuildInfo::KernelBuildInfoBuilder> &builder,
                        const std::vector<std::shared_ptr<OpIOInfo>> &outputs) {
  std::vector<TypeId> outputs_type;
  std::vector<std::string> outputs_format;
  auto real_output_num = AnfAlgo::GetOutputTensorNum(anf_node);
  size_t output_idx = 0;
  for (const auto output : outputs) {
    MS_EXCEPTION_IF_NULL(output);
    if (output_idx >= real_output_num) {
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
      auto type_id = AnfAlgo::GetOutputInferDataType(anf_node, output_idx);
      outputs_type.emplace_back(type_id);
      outputs_format.emplace_back(kOpFormat_DEFAULT);
      output_idx++;
    }
  }
  builder->SetOutputsDeviceType(outputs_type);
  builder->SetOutputsFormat(outputs_format);
}

void GenTidyKernelBuildInfo(const std::shared_ptr<AnfNode> &anf_node,
                            const std::vector<std::shared_ptr<OpIOInfo>> &inputs,
                            const std::vector<std::shared_ptr<OpIOInfo>> &outputs) {
  auto builder_tmp = std::make_shared<KernelBuildInfo::KernelBuildInfoBuilder>();
  builder_tmp->SetKernelType(TBE_KERNEL);
  SetTidyInputsInfo(anf_node, builder_tmp, inputs);
  SetTidyOutputsInfo(anf_node, builder_tmp, outputs);
  AnfAlgo::SetSelectKernelBuildInfo(builder_tmp->Build(), anf_node.get());
}

void ReplaceByDynamicFormatDtype(const CNodePtr &kernel_node, const std::shared_ptr<const OpInfo> &op_info_ptr,
                                 const std::shared_ptr<OpInfo> op_info_new_ptr) {
  std::vector<std::shared_ptr<OpIOInfo>> inputs_static = op_info_ptr->inputs_ptr();
  std::vector<std::shared_ptr<OpIOInfo>> outputs_static = op_info_ptr->outputs_ptr();
  std::vector<std::shared_ptr<OpIOInfo>> inputs_dyn;
  std::vector<std::shared_ptr<OpIOInfo>> outputs_dyn;
  if ((op_info_ptr->imply_type() == kTBE) && (!mindspore::opt::IsNopNode(kernel_node->cast<AnfNodePtr>()))) {
    // 1. create tidy kernelBuildInfo in order to generate json for calling op_select_format
    auto anf_node = kernel_node->cast<std::shared_ptr<AnfNode>>();
    auto kernel_build_info_ptr = AnfAlgo::GetSelectKernelBuildInfo(anf_node);
    GenTidyKernelBuildInfo(kernel_node, inputs_static, outputs_static);

    // 2.get dynamic format from op_impl
    std::string res_json_str;
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    if (context_ptr->execution_mode() != kPynativeMode) {
      res_json_str = OpSelectFormat(kernel_node);
    }
    if (!res_json_str.empty()) {
      (void)ParseDynamicFormatJson(res_json_str, &inputs_dyn, &outputs_dyn);
    }
    if (inputs_static.size() != inputs_dyn.size()) {
      inputs_dyn.clear();
    }
    if (outputs_static.size() != outputs_dyn.size()) {
      outputs_dyn.clear();
    }

    // 3. resume kernel node's SelectKernelBuildInfo
    // As it has been replaced by GenTidyKernelBuildInfo in order to call python func
    AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_ptr, anf_node.get());
  }
  // 4.replace by dynamic format and dtype
  if (inputs_dyn.empty() && outputs_dyn.empty()) {
    MS_LOG(INFO) << "Dynamic select format response is empty, use static register info.";
    op_info_new_ptr->set_inputs_ptr(inputs_static);
    op_info_new_ptr->set_outputs_ptr(outputs_static);
  } else {
    MS_LOG(INFO) << "Dynamic select format response successful, use dynamic format.";
    for (size_t i = 0; i < inputs_static.size(); i++) {
      inputs_dyn[i]->set_param_type(inputs_static[i]->param_type());
    }
    for (size_t j = 0; j < outputs_static.size(); j++) {
      outputs_dyn[j]->set_param_type(outputs_static[j]->param_type());
    }
    op_info_new_ptr->set_inputs_ptr(inputs_dyn);
    op_info_new_ptr->set_outputs_ptr(outputs_dyn);
  }

  // 5.copy other opinfo to new op_info_new
  op_info_new_ptr->set_op_name(op_info_ptr->op_name());
  op_info_new_ptr->set_imply_type(op_info_ptr->imply_type());
  op_info_new_ptr->set_fusion_type(op_info_ptr->fusion_type());
}

bool SetKernelBuilderInputInfo(const std::vector<std::shared_ptr<OpIOInfo>> &inputs, size_t real_input_num,
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
      MS_LOG(ERROR) << "Set input kernel builder info, dtyps size != formats size.";
      return false;
    }

    if (param_type == "dynamic") {
      if (dyn_input_sizes.empty()) {
        MS_LOG(ERROR) << "Set input kernel builder info, dyn_input_sizes's size is 0 when param_type is dynamic";
        return false;
      }

      for (int t = 0; t < dyn_input_sizes[dyn_input_idx]; t++) {
        kernel_info_index++;
        auto type_id = tbe::DtypeToTypeId(dtypes[builder_idex]);
        inputs_device_type.push_back(type_id);
        inputs_format.push_back(formats[builder_idex]);
      }
      dyn_input_idx++;
    } else if (param_type == "required") {
      kernel_info_index++;
      auto type_id = tbe::DtypeToTypeId(dtypes[builder_idex]);
      inputs_device_type.push_back(type_id);
      inputs_format.push_back(formats[builder_idex]);
    } else {
      if (kernel_info_index < real_input_num) {
        MS_LOG(INFO) << "Set input kernel builder info, input type is optional, input index is " << kernel_info_index;
        kernel_info_index++;
        auto type_id = tbe::DtypeToTypeId(dtypes[builder_idex]);
        inputs_device_type.push_back(type_id);
        inputs_format.push_back(formats[builder_idex]);
      }
    }
  }

  builder->SetInputsDeviceType(inputs_device_type);
  builder->SetInputsFormat(inputs_format);
  return true;
}

bool SetKernelBuilderOutputInfo(const std::vector<std::shared_ptr<OpIOInfo>> &outputs, size_t builder_idex,
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
      MS_LOG(WARNING) << "real_output_num: " << real_output_num << ", output_idx: " << output_idx << "is out of limit!";
      continue;
    }
    size_t output_num = 0;
    if (output->param_type() == "dynamic") {
      if (outputs.size() > 1) {
        MS_LOG(EXCEPTION) << "Dynamic output is unsupported multi output!";
      }
      output_num = real_output_num;
    } else if (output->param_type() == "required") {
      output_num = 1;
    } else {
      if (output_idx < real_output_num) {
        MS_LOG(INFO) << "Set output kernel builder info, output type is optional, output index is " << output_idx;
        output_num = 1;
      }
    }

    for (size_t i = 0; i < output_num; i++) {
      std::vector<std::string> dtypes = output->dtypes();
      std::vector<std::string> formats = output->formats();
      if (dtypes.size() != kernel_info_cnt || formats.size() != kernel_info_cnt) {
        MS_LOG(ERROR) << "Set output kernel builder info, dtyps size != formats size.";
        return false;
      }
      auto type_id = tbe::DtypeToTypeId(dtypes[builder_idex]);
      outputs_device_type.push_back(type_id);
      outputs_format.push_back(formats[builder_idex]);
      output_idx++;
    }
  }

  builder->SetOutputsFormat(outputs_format);
  builder->SetOutputsDeviceType(outputs_device_type);
  return true;
}

void SetKernelBuildCommonInfo(const std::shared_ptr<KernelBuildInfo::KernelBuildInfoBuilder> &builder,
                              Processor processor, const std::shared_ptr<const OpInfo> &op_info_ptr) {
  MS_EXCEPTION_IF_NULL(builder);
  MS_EXCEPTION_IF_NULL(op_info_ptr);

  builder->SetProcessor(processor);
  std::string fusion_type = op_info_ptr->fusion_type();
  if (tbe::GetFusionType(fusion_type) != UNKNOWN_FUSION_TYPE) {
    builder->SetFusionType(tbe::GetFusionType(fusion_type));
  }
  builder->SetKernelType(TBE_KERNEL);
}

bool ParseMetadata(const CNodePtr &kernel_node, const std::shared_ptr<const OpInfo> &op_info_ptr,
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
      SetKernelBuildCommonInfo(builder, Processor::AICORE, op_info_ptr);

      if (!SetKernelBuilderInputInfo(inputs, real_input_num, j, dyn_input_sizes, builder)) {
        MS_LOG(ERROR) << "Parse kernel metadata, set inputs kernel builder info failed.";
        return false;
      }

      if (outputs.size() > 0) {
        if (!SetKernelBuilderOutputInfo(outputs, j, real_output_num, builder)) {
          MS_LOG(ERROR) << "Parse kernel metadata, set outputs kernel builder info failed.";
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
      SetKernelBuildCommonInfo(builder, Processor::AICORE, op_info_ptr);

      if (!SetKernelBuilderOutputInfo(outputs, j, real_output_num, builder)) {
        MS_LOG(ERROR) << "Parse kernel metadata, set outputs kernel builder info failed.";
        return false;
      }

      kernel_info_list->push_back(builder->Build());
    }
  }
  return true;
}

void TbeMetadataInfo(const CNodePtr &kernel_node, std::vector<std::shared_ptr<KernelBuildInfo>> *kernel_info_list) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> parse_info_list;
  std::string op_name = AnfAlgo::GetCNodeName(kernel_node);
  auto op_info_ptr = mindspore::kernel::OpLib::FindOp(op_name, OpImplyType::kTBE);
  if (op_info_ptr == nullptr) {
    return;
  }
  // dynamic get op format and dtype and replace opinfo
  auto op_info_new_ptr = std::make_shared<OpInfo>();
  ReplaceByDynamicFormatDtype(kernel_node, op_info_ptr, op_info_new_ptr);

  if (!ParseMetadata(kernel_node, op_info_new_ptr, &parse_info_list)) {
    MS_LOG(INFO) << "Tbe parsed metadata of op[" << op_name << "] failed.";
    return;
  }

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  for (auto parse_info : parse_info_list) {
    if (context_ptr->execution_mode() == kPynativeMode) {
      kernel_info_list->push_back(parse_info);
    } else {
      if (CheckSupported(kernel_node, parse_info)) {
        kernel_info_list->push_back(parse_info);
      } else {
        MS_LOG(INFO) << "CheckSupported Failed for TBE op" << op_name << " kernel info.";
      }
    }
  }
  if (kernel_info_list->empty()) {
    MS_LOG(DEBUG) << "Tbe dose not has metadata of op[" << op_name << "].";
  }
}
}  // namespace kernel
}  // namespace mindspore
