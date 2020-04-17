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

#include "kernel/oplib/oplib.h"
#include <pybind11/pybind11.h>
#include <unordered_map>
#include <memory>
#include "utils/log_adapter.h"
#include "utils/overload.h"
#include "utils/context/ms_context.h"

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
constexpr auto kDynamicFormat = "dynamic_format";
constexpr auto kDtypeFormat = "dtype_format";
constexpr auto kAttr = "attr";
constexpr auto kIputs = "inputs";
constexpr auto kOutputs = "outputs";
constexpr auto kAiCPU = "AiCPU";
constexpr auto kTbe = "TBE";
constexpr auto kAkg = "akg";
constexpr auto kAutodiff = "AutoDiff";
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
std::vector<std::shared_ptr<OpInfo>> OpLib::op_info_;

std::string ImplTypeToStr(OpImplyType impl_type) {
  switch (impl_type) {
    case kTBE:
      return kTbe;
    case kAKG:
      return kAkg;
    case kAICPU:
      return kAiCPU;
    default:
      return "unknow";
  }
}
bool OpLib::RegOp(const std::string& json_string, const std::string& impl_path) {
  bool ret = false;
  try {
    auto op_json = nlohmann::json::parse(json_string);
    std::string imply_type_string = op_json.at(kImplyType);
    std::string op_name = op_json.at(kOpName);
    if (imply_type_string == kTbe) {
      OpImplyType imply_type = kTBE;
      ret = DecodeOpInfo(op_json, imply_type, impl_path);
    } else if (imply_type_string == kAutodiff) {
      OpImplyType imply_type = kAKG;
      ret = DecodeOpInfo(op_json, imply_type, impl_path);
    } else if (imply_type_string == kAiCPU) {
      OpImplyType imply_type = kAICPU;
      ret = DecodeOpInfo(op_json, imply_type, impl_path);
    } else {
      MS_LOG(DEBUG) << "Not support imply_type";
    }
    if (!ret) {
      MS_LOG(DEBUG) << "RegOp failed: opname:" << op_name << "imply_type" << imply_type_string;
    }
  } catch (const std::exception& e) {
    MS_LOG(DEBUG) << "get op_json elements failed:" << e.what();
  }
  return ret;
}

void OpLib::DecodeTBESpecificInfo(const nlohmann::json& obj, const std::shared_ptr<OpInfo>& op_info) {
  op_info->set_async_flag(obj.at(kAsyncFlag));
  op_info->set_binfile_name(obj.at(kBinfileName));
  op_info->set_compute_cost(obj.at(kComputeCost));
  op_info->set_kernel_name(obj.at(kKernelName));
  op_info->set_partial_flag(obj.at(kPartialFlag));
  if (obj.find(kOpPattern) != obj.end()) {
    op_info->set_op_pattern(obj.at(kOpPattern));
  }
  if (obj.find(kDynamicFormat) != obj.end()) {
    op_info->set_dynamic_format(obj.at(kDynamicFormat));
  }
}

bool OpLib::DecodeOpInfo(const nlohmann::json& obj, const mindspore::kernel::OpImplyType imply_type,
                         const std::string& impl_path) {
  std::shared_ptr<OpInfo> op_info = std::make_shared<OpInfo>();
  MS_EXCEPTION_IF_NULL(op_info);
  op_info->set_op_name(obj.at(kOpName));
  op_info->set_impl_path(impl_path);
  op_info->set_imply_type(imply_type);
  op_info->set_fusion_type(obj.at(kFusionType));
  if (imply_type == kTBE) {
    DecodeTBESpecificInfo(obj, op_info);
  }
  auto attrs = obj.at(kAttr);
  for (const auto& attr : attrs) {
    if (!DecodeAttr(attr, imply_type, op_info)) {
      MS_LOG(DEBUG) << "DecodeAttr Failed";
      return false;
    }
  }
  nlohmann::json dtype_format;
  if (obj.find(kDtypeFormat) != obj.end()) {
    dtype_format = obj.at(kDtypeFormat);
  }
  auto inputs = obj.at(kIputs);
  for (const auto& input : inputs) {
    if (!DecodeInputOutput(input, imply_type, kInput, op_info, dtype_format)) {
      MS_LOG(DEBUG) << "DecodeInputOutput Failed";
      return false;
    }
  }
  auto outputs = obj.at(kOutputs);
  for (const auto& output : outputs) {
    if (!DecodeInputOutput(output, imply_type, kOutput, op_info, dtype_format)) {
      MS_LOG(DEBUG) << "DecodeInputOutput Failed";
      return false;
    }
  }
  if (!GetRefInfo(op_info)) {
    MS_LOG(DEBUG) << "GetRefInfo Failed";
    return false;
  }
  if (!CheckRepetition(op_info)) {
    MS_LOG(DEBUG) << "CheckRepetition Failed";
    return false;
  }
  op_info_.push_back(op_info);
  return true;
}

bool OpLib::DecodeAttr(const nlohmann::json& obj, const OpImplyType imply_type,
                       const std::shared_ptr<OpInfo>& op_info) {
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
  } catch (const std::exception& e) {
    MS_LOG(DEBUG) << "DecodeAttr failed:" << e.what();
    ret = false;
  }
  return ret;
}

bool OpLib::DecodeDtypeFormat(const nlohmann::json& dtype_format, const std::shared_ptr<OpIOInfo>& op_io,
                              size_t index) {
  bool ret = true;
  try {
    std::vector<std::string> dtype;
    std::vector<std::string> format;
    for (const auto& it : dtype_format) {
      dtype.emplace_back(it[index][0]);
      format.emplace_back(it[index][1]);
    }
    op_io->set_dtypes(dtype);
    op_io->set_formats(format);
  } catch (const std::exception& e) {
    MS_LOG(ERROR) << "DecodeDtypeFormat falied" << e.what();
    ret = false;
  }
  return ret;
}

bool OpLib::DecodeInputOutput(const nlohmann::json& obj, const OpImplyType imply_type, const OpIOType io_type,
                              const std::shared_ptr<OpInfo>& op_info, const nlohmann::json& dtype_format) {
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
      MS_LOG(DEBUG) << "op" << op_io->name() << "dtype size:" << op_io->dtypes()
                    << "is not equal to format size:" << op_io->formats();
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
  } catch (const std::exception& e) {
    MS_LOG(DEBUG) << "DecodeInputOutput failed" << e.what();
    ret = false;
  }
  return ret;
}

std::shared_ptr<OpInfo> OpLib::FindOp(const std::string& op_name, OpImplyType imply_type) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool is_gpu = (context->device_target() == kGPUDevice);
  if ((is_gpu && (imply_type == kTBE || imply_type == kAICPU)) ||
      (!is_gpu && (imply_type != kTBE && imply_type != kAICPU))) {
    MS_LOG(ERROR) << "FindOp failed: opname:" << op_name << ", imply_type:" << ImplTypeToStr(imply_type)
                  << ", current op num:" << op_info_.size();
    return nullptr;
  }
  for (const auto& op_info : op_info_) {
    MS_EXCEPTION_IF_NULL(op_info);
    if (op_info->op_name() == op_name && op_info->imply_type() == imply_type) {
      return op_info;
    }
  }
  MS_LOG(DEBUG) << "FindOp failed: opname:" << op_name << ", imply_type:" << ImplTypeToStr(imply_type)
                << ", current op num:" << op_info_.size();
  return nullptr;
}

bool OpLib::GetRefInfo(const std::shared_ptr<OpInfo>& op_info) {
  MS_EXCEPTION_IF_NULL(op_info);
  const auto& output_infos = op_info->outputs_ptr();
  const auto& input_infos = op_info->inputs_ptr();
  for (size_t out_index = 0; out_index < output_infos.size(); out_index++) {
    const auto& out_name = output_infos[out_index]->name();
    for (size_t in_index = 0; in_index < input_infos.size(); in_index++) {
      const auto& in_name = input_infos[in_index]->name();
      if (out_name == in_name) {
        if (op_info->has_ref_index(out_index)) {
          MS_LOG(DEBUG) << "The out_index" << out_index << "is already in ref_info";
          return false;
        }
        op_info->add_ref_pair(out_index, in_index);
        MS_LOG(INFO) << "add ref info, op name is " << op_info->op_name() << ", outindex is " << out_index
                     << ", in_index is " << in_index;
      }
    }
  }
  return true;
}

bool OpLib::CheckRepetition(const std::shared_ptr<OpInfo>& op_info) {
  MS_EXCEPTION_IF_NULL(op_info);
  for (const auto& exist_op_info : op_info_) {
    MS_EXCEPTION_IF_NULL(exist_op_info);
    if (exist_op_info->op_name() == op_info->op_name() && exist_op_info->imply_type() == op_info->imply_type() &&
        exist_op_info->impl_path() != op_info->impl_path()) {
      MS_LOG(DEBUG) << "Has already exist, drop the latter one, op name:" << op_info->op_name()
                    << "op type:" << ImplTypeToStr(op_info->imply_type());
      return false;
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
