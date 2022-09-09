/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <fstream>
#include <map>
#include "tools/converter/quantizer/debug_info_manager.h"
#include "src/litert/tensor_category.h"
#include "src/common/log_adapter.h"
#include "include/errorcode.h"
#include "tools/converter/preprocess/image_preprocess.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "src/common/file_utils.h"
#include "src/litert/weight_decoder.h"
#include "tools/common/string_util.h"

namespace mindspore::lite {
namespace {
constexpr int kNumUsPerMs = 1000;
}  // namespace
std::vector<LiteQuantParam> DebugInfoManager::ConvertTensorsQuantParam(const mindspore::schema::Tensor *src_tensor) {
  MS_ASSERT(src_tensor != nullptr);
  auto quant_params = src_tensor->quantParams();
  std::vector<LiteQuantParam> lite_quant_params;
  if (quant_params != nullptr) {
    for (size_t j = 0; j < quant_params->size(); j++) {
      auto quant_param = quant_params->Get(j);
      LiteQuantParam quant_arg{};
      if (quant_param == nullptr) {
        quant_arg.inited = false;
      } else {
        quant_arg.inited = true;
        quant_arg.bitNum = quant_param->numBits();
        quant_arg.scale = quant_param->scale();
        quant_arg.zeroPoint = quant_param->zeroPoint();
        quant_arg.var_corr = quant_param->varCorr();
        quant_arg.mean_corr = quant_param->meanCorr();
        quant_arg.roundType = quant_param->roundType();
        quant_arg.multiplier = quant_param->multiplier();
        quant_arg.dstDtype = quant_param->dstDtype();
        quant_arg.min = quant_param->min();
        quant_arg.max = quant_param->max();
      }
      lite_quant_params.push_back(quant_arg);
    }
  }
  return lite_quant_params;
}

void DebugInfoManager::AddQuantParamExtend(const mindspore::lite::LiteGraph::Node *node,
                                           const mindspore::schema::Tensor *tensor) {
  CHECK_NULL_RETURN_VOID(node);
  CHECK_NULL_RETURN_VOID(tensor);
  auto q_param = ConvertTensorsQuantParam(tensor);
  if (!q_param.empty()) {
    QuantParamExtend quant_param_extend;
    quant_param_extend.quant_params = q_param;
    quant_param_extend.node_name = node->name_;
    quant_param_extend.tensor_name = tensor->name()->str();
    quant_param_extend.node_type = schema::EnumNamePrimitiveType(static_cast<PrimitiveType>(node->node_type_));
    std::vector<int> dims;
    int element_num = 1;
    MS_CHECK_PTR_IF_NULL(tensor->dims());
    for (size_t j = 0; j < tensor->dims()->size(); j++) {
      auto dim = tensor->dims()->data()[j];
      dims.push_back(dim);
      element_num *= dim;
    }
    quant_param_extend.dims = dims;
    quant_param_extend.element_num = element_num;
    quant_params_.push_back(quant_param_extend);
  }
}

std::string DebugInfoManager::ParseInOutTensorToString(InOutFlag in_out_flag) {
  switch (in_out_flag) {
    case INPUT:
      return "Input";
    case OUTPUT:
      return "Output";
  }
  return "ERROR";
}

std::string DebugInfoManager::ParseDataTypeFlagToString(DataTypeFlag data_type_flag) const {
  switch (data_type_flag) {
    case ORIGIN:
      return "Origin";
    case DEQUANT:
      return "Dequant";
  }
  return "ERROR";
}

std::string DebugInfoManager::ParseTensorTypeFlagToString(TensorTypeFlag data_type_flag) {
  switch (data_type_flag) {
    case ACTIVATION:
      return "Activation";
    case WEIGHT:
      return "Weight";
  }
  return "ERROR";
}

void DebugInfoManager::FreeBuffer() {
  for (auto iter = origin_info_.begin(); iter != origin_info_.end(); ++iter) {
    auto tensor = iter->second.tensor_data;
    if (tensor.data != nullptr) {
      free(tensor.data);
      tensor.data = nullptr;
      tensor.size = 0;
      tensor.elements_num = 0;
    }
  }
  origin_info_.clear();
  for (const auto &info : compared_info_) {
    auto tensor = info.tensor_data;
    if (tensor.data != nullptr) {
      free(tensor.data);
      tensor.data = nullptr;
      tensor.size = 0;
      tensor.elements_num = 0;
    }
  }
  compared_info_.clear();
}

void DebugInfoManager::PrintInfo(const QuantDebugInfo &info) {
  std::cout << info.primary_key.node_name << ",";
  std::cout << info.node_type << ",";
  std::cout << info.tensor_name << ",";
  std::cout << ParseInOutTensorToString(info.primary_key.in_out_flag) << ",";
  std::cout << ParseDataTypeFlagToString(info.data_type_flag) << ",";
  std::cout << ParseTensorTypeFlagToString(info.tensor_type_flag) << ",";
  std::cout << info.min << ",";
  std::cout << info.quartile1 << ",";
  std::cout << info.median << ",";
  std::cout << info.quartile3 << ",";
  std::cout << info.max << ",";
  std::cout << info.mean << ",";
  std::cout << info.var << ",";
  std::cout << info.sparsity << ",";
  std::cout << info.clip << ",";
  std::cout << info.cos_similarity << ",";
  std::cout << std::endl;
}

void DebugInfoManager::PrintAllDebugInfo() {
  std::cout << ",NodeName,NodeType,TensorName,InOutFlag,DataTypeFlag,TensorTypeFlag,Min,Q1,Median,Q3,Max,Mean,Var,"
               "Sparsity,Clip,"
               "CosineSimilarity,\n";
  size_t total = 0;
  for (size_t i = 0; i < compared_info_.size(); ++i) {
    auto compared_info = compared_info_[i];
    auto origin_info = origin_info_.find(compared_info.primary_key);
    if (origin_info != origin_info_.end()) {
      std::cout << total++ << ",";
      PrintInfo(origin_info->second);
    }
    std::cout << total++ << ",";
    PrintInfo(compared_info);
  }
}

void DebugInfoManager::SaveInfo(std::ofstream &out_file, const QuantDebugInfo &info) {
  out_file << info.primary_key.node_name << ",";
  out_file << info.node_type << ",";
  out_file << info.tensor_name << ",";
  out_file << ParseInOutTensorToString(info.primary_key.in_out_flag) << ",";
  out_file << ParseDataTypeFlagToString(info.data_type_flag) << ",";
  out_file << ParseTensorTypeFlagToString(info.tensor_type_flag) << ",";
  out_file << info.min << ",";
  out_file << info.quartile1 << ",";
  out_file << info.median << ",";
  out_file << info.quartile3 << ",";
  out_file << info.max << ",";
  out_file << info.mean << ",";
  out_file << info.var << ",";
  out_file << info.sparsity << ",";
  out_file << info.clip << ",";
  out_file << info.cos_similarity << ",";
  out_file << "\n";
}

int DebugInfoManager::SaveInfo(const std::string &file_path) {
  std::ofstream out_file;
  out_file.open(file_path, std::ios::out);
  if (!out_file.is_open()) {
    MS_LOG(ERROR) << "file open failed";
    return RET_ERROR;
  }
  out_file << ",NodeName,NodeType,TensorName,InOutFlag,DataTypeFlag,TensorTypeFlag,Min,Q1,Median,Q3,Max,Mean,Var,"
              "Sparsity,Clip,"
              "CosineSimilarity,\n";
  size_t total = 0;
  for (size_t i = 0; i < compared_info_.size(); ++i) {
    auto compared_info = compared_info_[i];
    auto origin_info = origin_info_.find(compared_info.primary_key);
    if (origin_info != origin_info_.end()) {
      out_file << total++ << ",";
      SaveInfo(out_file, origin_info->second);
    }
    out_file << total++ << ",";
    SaveInfo(out_file, compared_info);
  }
  out_file.close();
  std::cout << "Success save debug info to " + file_path << "\n";
  return RET_OK;
}

int DebugInfoManager::SetOriginStaticInfo(QuantDebugInfo *quant_debug_info, const mindspore::lite::Tensor &tensor,
                                          const quant::DebugMode &debug_mode) {
  CHECK_NULL_RETURN(quant_debug_info);
  if (debug_mode == quant::DETAIL) {
    TypeId data_type = tensor.data_type();
    if (data_type == kNumberTypeFloat32) {
      GetStatByTensor<float>(static_cast<float *>(tensor.data()), tensor.ElementsNum(), quant_debug_info);
    } else if (data_type == kNumberTypeInt32) {
      GetStatByTensor<int>(static_cast<int *>(tensor.data()), tensor.ElementsNum(), quant_debug_info);
    } else if (data_type == kNumberTypeInt8) {
      GetStatByTensor<int8_t>(static_cast<int8_t *>(tensor.data()), tensor.ElementsNum(), quant_debug_info);
    } else {
      MS_LOG(ERROR) << tensor.tensor_name() << " unsupported data type " << tensor.data_type();
      return RET_ERROR;
    }
  }
  quant_debug_info->clip = 0;

  CHECK_NULL_RETURN(tensor.data());
  quant_debug_info->tensor_data.data = malloc(tensor.Size());
  CHECK_MALLOC_RES(quant_debug_info->tensor_data.data, RET_NULL_PTR);
  auto ret = memcpy_s(quant_debug_info->tensor_data.data, tensor.Size(), tensor.data(), tensor.Size());
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy memory failed.";
    free(quant_debug_info->tensor_data.data);
    quant_debug_info->tensor_data.data = nullptr;
    return RET_ERROR;
  }
  quant_debug_info->tensor_data.data_type = tensor.data_type();
  quant_debug_info->tensor_data.size = tensor.Size();
  quant_debug_info->tensor_data.elements_num = static_cast<size_t>(tensor.ElementsNum());
  return RET_OK;
}

int DebugInfoManager::SetQuantStaticInfo(const std::vector<mindspore::lite::Tensor *> &inputs,
                                         const OpParameter *op_parameter, int tensor_index,
                                         QuantDebugInfo *quant_debug_info, const mindspore::lite::Tensor &tensor,
                                         const quant::DebugMode &debug_mode) {
  MS_CHECK_TRUE_MSG(quant_debug_info != nullptr, RET_ERROR, "quant_debug_info is nullptr.");
  auto preferred_dim =
    mindspore::lite::WeightDecoder::GetPreferredDim(inputs, op_parameter, tensor_index, tensor.shape(), Version());
  float *quant_data;
  if (tensor.data_type() == kNumberTypeInt8) {
    quant_data = mindspore::lite::WeightDecoder::DequantData<int8_t, float>(&tensor, preferred_dim);
  } else if (tensor.data_type() == kNumberTypeInt16) {
    quant_data = mindspore::lite::WeightDecoder::DequantData<int16_t, float>(&tensor, preferred_dim);
  } else if (tensor.data_type() == kNumberTypeInt32) {  // Bias
    quant_data = mindspore::lite::WeightDecoder::DequantData<int, float>(&tensor, preferred_dim);
  } else if (tensor.data_type() == kNumberTypeFloat32) {  // QuantDTypeCast(float32->int8)
    quant_data = static_cast<float *>(tensor.data());
  } else {
    MS_LOG(ERROR) << tensor.tensor_name() << " unsupported data type " << tensor.data_type();
    return RET_ERROR;
  }
  CHECK_NULL_RETURN(quant_data);
  if (debug_mode == quant::DETAIL) {
    GetStatByTensor<float>(static_cast<float *>(quant_data), tensor.ElementsNum(), quant_debug_info);
  }

  size_t buf_size = tensor.ElementsNum() * sizeof(float);
  quant_debug_info->tensor_data.data = malloc(buf_size);
  CHECK_MALLOC_RES(quant_debug_info->tensor_data.data, RET_NULL_PTR);
  auto ret = memcpy_s(quant_debug_info->tensor_data.data, buf_size, quant_data, buf_size);
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy memory failed.";
    free(quant_debug_info->tensor_data.data);
    quant_debug_info->tensor_data.data = nullptr;
    return false;
  }
  quant_debug_info->tensor_data.data_type = kNumberTypeFloat32;
  quant_debug_info->tensor_data.size = buf_size;
  quant_debug_info->tensor_data.elements_num = static_cast<size_t>(tensor.ElementsNum());
  return RET_OK;
}

int DebugInfoManager::AddOriginInfo(const mindspore::MSCallBackParam &call_back_param, bool is_input,
                                    size_t tensor_index, const mindspore::lite::Tensor *origin_tensor,
                                    const quant::DebugMode &debug_mode) {
  CHECK_NULL_RETURN(origin_tensor);
  if (call_back_param.node_type == schema::EnumNamePrimitiveType(schema::PrimitiveType_QuantDTypeCast)) {
    return RET_OK;
  }
  QuantDebugInfo origin_debug_info;
  origin_debug_info.primary_key.node_name = call_back_param.node_name;
  origin_debug_info.primary_key.in_out_flag = is_input ? INPUT : OUTPUT;
  origin_debug_info.primary_key.index = tensor_index;
  origin_debug_info.node_type = call_back_param.node_type;
  origin_debug_info.tensor_name = origin_tensor->tensor_name();
  auto is_const = origin_tensor->category() == CONST_TENSOR || origin_tensor->category() == CONST_SCALAR;
  origin_debug_info.tensor_type_flag = is_const ? WEIGHT : ACTIVATION;
  auto ret = SetOriginStaticInfo(&origin_debug_info, *origin_tensor, debug_mode);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << origin_tensor->tensor_name() << " get origin static info failed.";
    return RET_ERROR;
  }
  origin_debug_info.data_type_flag = ORIGIN;
  origin_debug_info.clip = 0;
  origin_debug_info.cos_similarity = 1;

  origin_debug_info.tensor_data.data_type = origin_tensor->data_type();
  origin_debug_info.tensor_data.size = origin_tensor->Size();
  origin_debug_info.tensor_data.elements_num = origin_tensor->ElementsNum();
  auto iter = origin_info_.find(origin_debug_info.primary_key);
  if (iter == origin_info_.end()) {
    origin_info_[origin_debug_info.primary_key] = origin_debug_info;
  } else {
    MS_LOG(ERROR) << iter->second.primary_key << " is exit.";
  }
  return RET_OK;
}

int DebugInfoManager::AddComparedInfo(const mindspore::MSCallBackParam &call_back_param,
                                      const std::vector<mindspore::lite::Tensor *> &inputs, OpParameter *op_parameter,
                                      bool is_input, size_t tensor_index,
                                      const mindspore::lite::Tensor *compared_tensor,
                                      const quant::DebugMode &debug_mode) {
  CHECK_NULL_RETURN(op_parameter);
  CHECK_NULL_RETURN(compared_tensor);
  QuantDebugInfo compared_debug_info;
  compared_debug_info.primary_key.index = tensor_index;
  compared_debug_info.primary_key.node_name = call_back_param.node_name;
  compared_debug_info.primary_key.in_out_flag = is_input ? INPUT : OUTPUT;
  compared_debug_info.node_type = call_back_param.node_type;
  compared_debug_info.tensor_name = compared_tensor->tensor_name();
  compared_debug_info.data_type_flag = DEQUANT;
  auto is_const = compared_tensor->category() == CONST_TENSOR || compared_tensor->category() == CONST_SCALAR;
  compared_debug_info.tensor_type_flag = is_const ? WEIGHT : ACTIVATION;
  if (!compared_tensor->quant_params().empty()) {
    auto ret =
      SetQuantStaticInfo(inputs, op_parameter, tensor_index, &compared_debug_info, *compared_tensor, debug_mode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << compared_tensor->tensor_name() << " get quant static info failed.";
      return RET_ERROR;
    }
  } else {
    auto ret = SetOriginStaticInfo(&compared_debug_info, *compared_tensor, debug_mode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << compared_tensor->tensor_name() << " get origin static info failed.";
      return RET_ERROR;
    }
  }
  compared_info_.push_back(compared_debug_info);
  return RET_OK;
}

std::map<std::string, mindspore::schema::Tensor *> DebugInfoManager::ParseInputTensors(
  const mindspore::lite::LiteModel &model) const {
  std::map<std::string, mindspore::schema::Tensor *> maps;
  for (auto &node : model.graph_.all_nodes_) {
    for (auto &index : node->input_indices_) {
      auto tensor_name = model.graph_.all_tensors_[index]->name()->str();
      maps[tensor_name] = model.graph_.all_tensors_[index];
    }
  }
  return maps;
}

std::map<std::string, mindspore::schema::Tensor *> DebugInfoManager::ParseOutputTensorFromModel(const Model &model) {
  std::map<std::string, mindspore::schema::Tensor *> maps;
  for (auto &node : model.graph_.all_nodes_) {
    for (auto &index : node->output_indices_) {
      auto tensor_name = model.graph_.all_tensors_[index]->name()->str();
      maps[tensor_name] = model.graph_.all_tensors_[index];
    }
  }
  return maps;
}

int DebugInfoManager::GetDataFromTensorMap(const mindspore::schema::Tensor &schema_tensor,
                                           mindspore::lite::Tensor *dst_tensor) {
  MS_CHECK_TRUE_MSG(dst_tensor != nullptr, RET_ERROR, "dst_tensor is nullptr.");
  auto src_tensor = new (std::nothrow) SchemaTensorWrapper();
  if (src_tensor == nullptr) {
    MS_LOG(ERROR) << "Create SchemaTensorWrapper return nullptr";
    return RET_ERROR;
  }
  auto init_ret = src_tensor->Init(schema_tensor, SCHEMA_CUR, "");
  if (!init_ret) {
    MS_LOG(ERROR) << "src_tensor init failed.";
    delete src_tensor;
    return RET_ERROR;
  }
  auto ret = WeightDecoder::DecompressTensor(*src_tensor, dst_tensor);
  if (ret == RET_NO_CHANGE) {
    if (src_tensor->length() < dst_tensor->Size()) {
      MS_LOG(ERROR) << "Tensor data shape invalid";
      return RET_ERROR;
    }
    auto data_pair = src_tensor->ReleaseData();
    dst_tensor->set_data(data_pair.second);
    // this buffer must be freed by framework and set own data to false, this memory will not be released.
    dst_tensor->set_own_data(false);
  } else if (ret != RET_OK) {
    MS_LOG(ERROR) << "Decompress tensor data failed: " << ret;
    return ret;
  }
  delete src_tensor;
  return RET_OK;
}

int DebugInfoManager::GetConstTensor(const std::map<std::string, mindspore::schema::Tensor *> &input_tensor_map,
                                     const mindspore::lite::Tensor *tensor, mindspore::lite::Tensor *new_tensor) {
  CHECK_NULL_RETURN(tensor);
  CHECK_NULL_RETURN(new_tensor);
  auto iter = input_tensor_map.find(tensor->tensor_name());
  if (iter == input_tensor_map.end()) {
    MS_LOG(ERROR) << tensor->tensor_name() << " find failed.";
    return RET_ERROR;
  }
  new_tensor->set_data_type(static_cast<TypeId>(iter->second->dataType()));
  new_tensor->set_shape(tensor->shape());
  new_tensor->set_quant_params(ConvertTensorsQuantParam(iter->second));
  new_tensor->set_tensor_name(tensor->tensor_name());
  new_tensor->set_category(tensor->category());
  new_tensor->set_format(tensor->format());
  auto ret = GetDataFromTensorMap(*iter->second, new_tensor);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << tensor->tensor_name() << " get data from tensor map failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

MSKernelCallBack DebugInfoManager::GetOriginBeforeCallBack(
  const std::map<std::string, mindspore::schema::Tensor *> &input_tensor_map, const quant::DebugMode &debug_mode) {
  auto before_callback = [&](const std::vector<mindspore::MSTensor> &inputs,
                             const std::vector<mindspore::MSTensor> &outputs, const MSCallBackParam &call_param) {
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto tensor = inputs.at(i);
      if (debug_mode == quant::FAST) {
        continue;
      }
      auto lite_tensor = quant::MSTensorToLiteTensor(inputs.at(i));
      MS_LOG(INFO) << "Get input " << tensor.Name() << " statistics info.";
      if (tensor.IsConst()) {
        mindspore::lite::Tensor new_tensor;
        auto ret = GetConstTensor(input_tensor_map, lite_tensor, &new_tensor);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << tensor.Name() << " get const tensor failed.";
          return false;
        }
        ret = AddOriginInfo(call_param, true, i, &new_tensor, debug_mode);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << tensor.Name() << " add origin info failed.";
          return false;
        }
      } else {
        auto ret = AddOriginInfo(call_param, true, i, static_cast<mindspore::lite::Tensor *>(lite_tensor), debug_mode);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << tensor.Name() << " add origin info failed.";
          return false;
        }
      }
    }
    return true;
  };
  return before_callback;
}

MSKernelCallBack DebugInfoManager::GetQuantBeforeCallBack(
  const std::map<std::string, mindspore::schema::Tensor *> &input_tensor_map,
  const std::map<std::string, OpParameter *> &op_parameters, const quant::DebugMode &debug_mode) {
  auto before_callback = [&](const std::vector<mindspore::MSTensor> &inputs,
                             const std::vector<mindspore::MSTensor> &outputs, const MSCallBackParam &call_param) {
    auto lite_inputs = quant::MSTensorToLiteTensors(inputs);
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto tensor = inputs.at(i);
      auto lite_tensor = quant::MSTensorToLiteTensor(tensor);
      MS_CHECK_TRUE_RET(lite_tensor != nullptr, false);
      if (debug_mode == quant::FAST && (origin_outputs_.find(tensor.Name()) == origin_outputs_.end())) {
        continue;
      }
      MS_LOG(DEBUG) << "Get input " << tensor.Name() << " statistics info.";
      if (op_parameters.find(call_param.node_name) == op_parameters.end()) {
        MS_LOG(ERROR) << tensor.Name() << " op_parameters find node name " << call_param.node_name << " failed.";
        return false;
      }
      auto is_const = static_cast<mindspore::lite::Tensor *>(lite_tensor)->category() == CONST_TENSOR ||
                      static_cast<mindspore::lite::Tensor *>(lite_tensor)->category() == CONST_SCALAR;
      if (is_const) {
        mindspore::lite::Tensor new_tensor;
        auto ret = GetConstTensor(input_tensor_map, lite_tensor, &new_tensor);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << tensor.Name() << " get const tensor failed.";
          return false;
        }
        ret = AddComparedInfo(call_param, lite_inputs, op_parameters.at(call_param.node_name), true, i, &new_tensor,
                              debug_mode);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << tensor.Name() << " add compared info failed.";
          return false;
        }
      } else {
        auto ret = AddComparedInfo(call_param, lite_inputs, op_parameters.at(call_param.node_name), true, i,
                                   lite_tensor, debug_mode);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << tensor.Name() << " add compared info failed.";
          return false;
        }
      }
    }
    return true;
  };
  return before_callback;
}

MSKernelCallBack DebugInfoManager::GetBeforeCallBack(
  const std::map<std::string, mindspore::schema::Tensor *> &input_tensor_map,
  const std::map<std::string, OpParameter *> &op_parameters, bool is_origin, const quant::DebugMode &debug_mode) {
  if (is_origin) {
    return GetOriginBeforeCallBack(input_tensor_map, debug_mode);
  } else {
    return GetQuantBeforeCallBack(input_tensor_map, op_parameters, debug_mode);
  }
}

MSKernelCallBack DebugInfoManager::GetAfterCallBack(const std::map<std::string, OpParameter *> &op_parameters,
                                                    bool is_origin, const quant::DebugMode &debug_mode) {
  MSKernelCallBack after_callback;
  if (is_origin) {
    after_callback = [&](const std::vector<mindspore::MSTensor> &inputs,
                         const std::vector<mindspore::MSTensor> &outputs, const MSCallBackParam &call_param) {
      // all outputs are same dtype.
      for (size_t i = 0; i < outputs.size(); ++i) {
        auto tensor = outputs.at(i);
        auto lite_tensor = quant::MSTensorToLiteTensor(tensor);
        // subgraph isolation by appending "_duplicate" to output tensor
        std::string delimiter = "_duplicate";
        std::vector<std::string> tensor_names = SplitStringToVector(tensor.Name(), delimiter);
        if (debug_mode == quant::FAST && origin_outputs_.find(tensor_names[0]) == origin_outputs_.end()) {
          continue;
        }
        MS_LOG(INFO) << "Get output " << tensor.Name() << " statistics info.";
        auto ret = AddOriginInfo(call_param, false, i, static_cast<mindspore::lite::Tensor *>(lite_tensor), debug_mode);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << tensor.Name() << " add origin info failed.";
          return false;
        }
      }
      return true;
    };
  } else {
    after_callback = [&](const std::vector<mindspore::MSTensor> &inputs,
                         const std::vector<mindspore::MSTensor> &outputs, const MSCallBackParam &call_param) {
      // all outputs are same dtype.
      for (size_t i = 0; i < outputs.size(); ++i) {
        auto tensor = outputs.at(i);
        // subgraph isolation by appending "_duplicate" to output tensor
        std::string delimiter = "_duplicate";
        std::vector<std::string> tensor_names = SplitStringToVector(tensor.Name(), delimiter);
        if (debug_mode == quant::FAST && (origin_outputs_.find(tensor_names[0]) == origin_outputs_.end())) {
          continue;
        }
        MS_LOG(INFO) << " Get output " << tensor.Name() << " statistics info.";
        auto lite_tensor = quant::MSTensorToLiteTensor(tensor);
        auto lite_inputs = quant::MSTensorToLiteTensors(inputs);
        if (op_parameters.find(call_param.node_name) == op_parameters.end()) {
          MS_LOG(ERROR) << tensor.Name() << " op_parameters find node name " << call_param.node_name << " failed.";
          return false;
        }
        auto ret = AddComparedInfo(call_param, lite_inputs, op_parameters.at(call_param.node_name), false, i,
                                   lite_tensor, debug_mode);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << tensor.Name() << " add compared info failed.";
          return false;
        }
      }
      return true;
    };
  }

  return after_callback;
}

void DebugInfoManager::PrintQuantParam() {
  if (quant_params_.empty()) {
    return;
  }
  std::cout << "NodeName,NodeType,TensorName,ElementsNum,Dims,Scale,ZeroPoint,Bits,CorrectionVar,CorrectionMean,";
  std::cout << "\n";
  for (const auto &quant_param : quant_params_) {
    for (const auto &param : quant_param.quant_params) {
      std::cout << quant_param.node_name << ",";
      std::cout << quant_param.node_type << ",";
      std::cout << quant_param.tensor_name << ",";
      std::cout << quant_param.element_num << ",";
      for (auto &dim : quant_param.dims) {
        std::cout << dim << " ";
      }
      std::cout << ",";
      std::cout << param.scale << ",";
      std::cout << param.zeroPoint << ",";
      std::cout << param.bitNum << ",";
      std::cout << param.var_corr << ",";
      std::cout << param.mean_corr << ",";
      std::cout << "\n";
    }
  }
}

int DebugInfoManager::SaveQuantParam(const std::string &file_path) {
  if (quant_params_.empty()) {
    return RET_OK;
  }
  std::ofstream out_file;
  out_file.open(file_path, std::ios::out);
  if (!out_file.is_open()) {
    MS_LOG(ERROR) << "file open failed";
    return RET_ERROR;
  }
  out_file << "NodeName,NodeType,TensorName,ElementsNum,Dims,Scale,ZeroPoint,Bits,CorrectionVar,CorrectionMean,";
  out_file << "\n";
  for (const auto &quant_param : quant_params_) {
    for (const auto &param : quant_param.quant_params) {
      out_file << quant_param.node_name << ",";
      out_file << quant_param.node_type << ",";
      out_file << quant_param.tensor_name << ",";
      out_file << quant_param.element_num << ",";
      for (auto &dim : quant_param.dims) {
        out_file << dim << " ";
      }
      out_file << ",";
      out_file << param.scale << ",";
      out_file << param.zeroPoint << ",";
      out_file << param.bitNum << ",";
      out_file << param.var_corr << ",";
      out_file << param.mean_corr << ",";
      out_file << "\n";
    }
  }
  out_file.close();
  std::cout << "Success save quant param to " + file_path << "\n";
  return RET_OK;
}

int DebugInfoManager::GetClipAndCos(const quant::DebugMode &debug_mode) {
  for (auto &info : compared_info_) {
    auto iter = origin_info_.find(info.primary_key);
    if (iter == origin_info_.end()) {
      continue;
    }
    if (iter->second.tensor_data.data_type != info.tensor_data.data_type ||
        iter->second.tensor_data.size != info.tensor_data.size ||
        iter->second.tensor_data.elements_num != info.tensor_data.elements_num) {
      MS_LOG(ERROR) << info.primary_key << " "
                    << " data is not match origin";
      FreeBuffer();
      return RET_ERROR;
    }
    info.cos_similarity = mindspore::lite::GetCosSimilarity(iter->second.tensor_data.data, info.tensor_data.data,
                                                            info.tensor_data.elements_num, info.tensor_data.data_type);
    if (debug_mode == quant::DETAIL) {
      info.clip = mindspore::lite::GetClipRate(iter->second.tensor_data.data, info.tensor_data.data,
                                               info.tensor_data.elements_num, info.tensor_data.data_type);
    }
  }
  return RET_OK;
}

void DebugInfoManager::GetOutputInfo() {
  std::vector<QuantDebugInfo> output_info;
  for (auto iter = compared_info_.begin(); iter != compared_info_.end(); ++iter) {
    // subgraph isolation by appending "_duplicate" to output tensor
    std::string delimiter = "_duplicate";
    std::vector<std::string> tensor_names = SplitStringToVector(iter->tensor_name, delimiter);
    if (origin_outputs_.find(tensor_names[0]) != origin_outputs_.end() && iter->primary_key.in_out_flag == OUTPUT) {
      MS_LOG(INFO) << "output tensor name: " << iter->tensor_name << " data_type_flag: " << iter->data_type_flag;
      output_info.push_back(*iter);
    }
  }
  output_infos_.push_back(output_info);
}

int DebugInfoManager::SaveOutputInfo(const std::string &file_path) {
  if (output_infos_.empty()) {
    return RET_OK;
  }
  std::ofstream out_file;
  out_file.open(file_path, std::ios::out);
  if (!out_file.is_open()) {
    MS_LOG(ERROR) << "file open failed";
    return RET_ERROR;
  }
  out_file << "Round,TensorName,CosineSimilarity,";
  out_file << "\n";
  for (size_t round = 0; round < output_infos_.size(); round++) {
    for (const auto &param : output_infos_[round]) {
      out_file << round << ",";
      out_file << param.tensor_name << ",";
      out_file << param.cos_similarity << ",";
      out_file << "\n";
    }
  }
  out_file.close();
  std::cout << "Success save quant param to " + file_path << "\n";
  return RET_OK;
}

int DebugInfoManager::StatisticsDataPerRound(
  const std::shared_ptr<mindspore::Model> &origin, const std::shared_ptr<mindspore::Model> &quant,
  const std::map<std::string, OpParameter *> &op_parameters, const std::shared_ptr<ConverterPara> &param,
  const std::map<std::string, mindspore::schema::Tensor *> &origin_input_tensor_map,
  const std::map<std::string, mindspore::schema::Tensor *> &quant_input_tensor_map, const size_t &round) {
  int ret;
  auto data_preprocess = param->dataPreProcessParam;
  for (auto tensor : origin->GetInputs()) {
    if (data_preprocess.calibrate_size > 0) {
      ret = preprocess::PreProcess(data_preprocess, tensor.Name(), round, &tensor);
    } else {
      ret = GenerateRandomData(&tensor);
    }
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "round" << round << ":" << tensor.Name() << " pre-process failed.";
      return ret;
    }
  }
  std::cout << "Statistics the original data distribution. Round " << round << std::endl;
  auto origin_before_callBack =
    GetBeforeCallBack(origin_input_tensor_map, op_parameters, true, param->commonQuantParam.debug_mode);
  auto origin_after_callBack = GetAfterCallBack(op_parameters, true, param->commonQuantParam.debug_mode);
  auto origin_outputs = origin->GetOutputs();
  auto status = origin->Predict(origin->GetInputs(), &origin_outputs, origin_before_callBack, origin_after_callBack);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "round:" << round << " origin model run graph failed.";
    return RET_ERROR;
  }

  std::cout << "Statistics the quant data distribution. Round " << round << std::endl;
  auto quant_before_callBack =
    GetBeforeCallBack(quant_input_tensor_map, op_parameters, false, param->commonQuantParam.debug_mode);
  auto quant_after_callBack = GetAfterCallBack(op_parameters, false, param->commonQuantParam.debug_mode);
  for (auto &tensor : quant->GetInputs()) {
    auto tensor_data = tensor.MutableData();
    CHECK_NULL_RETURN(tensor_data);
    ret = memcpy_s(tensor_data, tensor.DataSize(), origin->GetInputByTensorName(tensor.Name()).Data().get(),
                   origin->GetInputByTensorName(tensor.Name()).DataSize());
    if (ret != EOK) {
      MS_LOG(ERROR) << tensor.Name() << " memcpy failed.";
      return RET_ERROR;
    }
  }
  auto quant_outputs = quant->GetOutputs();
  status = quant->Predict(quant->GetInputs(), &quant_outputs, quant_before_callBack, quant_after_callBack);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "round:" << round << " quant model run graph failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

void DebugInfoManager::CollectQuantParam(const mindspore::lite::LiteModel &quant_lite_model) {
  for (auto &node : quant_lite_model.graph_.all_nodes_) {
    for (auto &index : node->input_indices_) {
      auto tensor = quant_lite_model.graph_.all_tensors_[index];
      AddQuantParamExtend(node, tensor);
    }
    for (auto &index : node->output_indices_) {
      auto tensor = quant_lite_model.graph_.all_tensors_[index];
      AddQuantParamExtend(node, tensor);
    }
  }
}

std::string DebugInfoManager::CreateFilePath(const std::string &dir_path, const std::string &file_name) const {
  auto real_path = RealPath(dir_path.c_str());
  std::string file_path = real_path + FILE_SEPARATOR + file_name;
  return file_path;
}

int DebugInfoManager::CompareOriginWithQuant(const std::shared_ptr<mindspore::Model> &origin,
                                             const std::shared_ptr<mindspore::Model> &quant,
                                             const std::map<std::string, OpParameter *> &op_parameters,
                                             const std::shared_ptr<ConverterPara> &param,
                                             const mindspore::lite::LiteModel &origin_lite_model,
                                             const mindspore::lite::LiteModel &quant_lite_model) {
  auto begin = GetTimeUs();
  CollectQuantParam(quant_lite_model);
  std::string file_name = "quant_param.csv";
  auto quant_param_save_path = CreateFilePath(param->commonQuantParam.debug_info_save_path, file_name);
  auto ret = SaveQuantParam(quant_param_save_path);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Failed to save quant param to " + quant_param_save_path;
    return ret;
  }

  auto origin_input_tensor_map = ParseInputTensors(origin_lite_model);
  auto quant_input_tensor_map = ParseInputTensors(quant_lite_model);
  for (auto &tensor : origin->GetOutputs()) {
    origin_outputs_[tensor.Name()] = tensor;
  }
  auto data_preprocess = param->dataPreProcessParam;
  // When the calibration data set does not exist, use 1 round of random numbers for comparison
  size_t rounds = static_cast<size_t>(data_preprocess.calibrate_size > 0 ? data_preprocess.calibrate_size : 1);
  for (size_t round = 0; round < rounds; round++) {
    ret = StatisticsDataPerRound(origin, quant, op_parameters, param, origin_input_tensor_map, quant_input_tensor_map,
                                 round);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Statistics Data failed for round: " << round;
      FreeBuffer();
      return RET_ERROR;
    }
    ret = GetClipAndCos(param->commonQuantParam.debug_mode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Get clip and cos failed.";
      FreeBuffer();
      return ret;
    }
    GetOutputInfo();
    if (param->commonQuantParam.debug_mode == quant::DETAIL) {
      file_name = "round_" + std::to_string(round) + ".csv";
      auto file_path = CreateFilePath(param->commonQuantParam.debug_info_save_path, file_name);
      ret = SaveInfo(file_path);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Failed to save debug info to " + file_path;
        FreeBuffer();
        return ret;
      }
    }
    FreeBuffer();
  }

  file_name = "output_summary.csv";
  auto output_param_save_path = CreateFilePath(param->commonQuantParam.debug_info_save_path, file_name);
  ret = SaveOutputInfo(output_param_save_path);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Failed to save output param to " + output_param_save_path;
    return ret;
  }
  auto end = GetTimeUs();
  MS_LOG(INFO) << "Total time spent " << ((end - begin) / kNumUsPerMs) << " ms.\n";
  return RET_OK;
}
}  // namespace mindspore::lite
