/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <fstream>
#include <map>
#include "tools/converter/quantizer/debug_info_manager.h"
#include "src/weight_decoder.h"
#include "src/common/log_adapter.h"
#include "src/lite_session.h"
#include "include/errorcode.h"
#include "tools/converter/preprocess/image_preprocess.h"
#include "tools/common/tensor_util.h"

namespace mindspore::lite {
namespace {
constexpr int kNumUsPerMs = 1000;
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

std::string DebugInfoManager::ParseDataTypeFlagToString(DataTypeFlag data_type_flag) {
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

int DebugInfoManager::SetOriginStaticInfo(QuantDebugInfo *quant_debug_info, const mindspore::lite::Tensor &tensor) {
  if (tensor.data_type() == kNumberTypeFloat32) {
    GetStatByTensor<float>(static_cast<float *>(tensor.data()), tensor.ElementsNum(), quant_debug_info);
  } else if (tensor.data_type() == kNumberTypeInt32) {
    GetStatByTensor<int>(static_cast<int *>(tensor.data()), tensor.ElementsNum(), quant_debug_info);
  } else if (tensor.data_type() == kNumberTypeInt8) {
    GetStatByTensor<int8_t>(static_cast<int8_t *>(tensor.data()), tensor.ElementsNum(), quant_debug_info);
  } else {
    MS_LOG(ERROR) << tensor.tensor_name() << " unsupported data type " << tensor.data_type();
    return RET_ERROR;
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
  quant_debug_info->tensor_data.elements_num = tensor.ElementsNum();
  return RET_OK;
}

int DebugInfoManager::SetQuantStaticInfo(const std::vector<mindspore::tensor::MSTensor *> &inputs,
                                         OpParameter *op_parameter, int tensor_index, QuantDebugInfo *quant_debug_info,
                                         const mindspore::lite::Tensor &tensor) {
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
  GetStatByTensor<float>(static_cast<float *>(quant_data), tensor.ElementsNum(), quant_debug_info);

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
  quant_debug_info->tensor_data.elements_num = tensor.ElementsNum();
  return RET_OK;
}

int DebugInfoManager::AddOriginInfo(const mindspore::CallBackParam &call_back_param, OpParameter *op_parameter,
                                    bool is_input, int tensor_index, mindspore::lite::Tensor *origin_tensor) {
  CHECK_NULL_RETURN(op_parameter);
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
  auto ret = SetOriginStaticInfo(&origin_debug_info, *origin_tensor);
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

int DebugInfoManager::AddComparedInfo(const mindspore::CallBackParam &call_back_param,
                                      const std::vector<mindspore::tensor::MSTensor *> &inputs,
                                      OpParameter *op_parameter, bool is_input, int tensor_index,
                                      mindspore::lite::Tensor *compared_tensor) {
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
    auto ret = SetQuantStaticInfo(inputs, op_parameter, tensor_index, &compared_debug_info, *compared_tensor);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << compared_tensor->tensor_name() << " get quant static info failed.";
      return RET_ERROR;
    }
  } else {
    auto ret = SetOriginStaticInfo(&compared_debug_info, *compared_tensor);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << compared_tensor->tensor_name() << " get origin static info failed.";
      return RET_ERROR;
    }
  }
  compared_info_.push_back(compared_debug_info);
  return RET_OK;
}

std::map<std::string, mindspore::schema::Tensor *> DebugInfoManager::ParseInputTensorFromModel(const Model &model) {
  std::map<std::string, mindspore::schema::Tensor *> maps;
  for (auto node : model.all_nodes_) {
    for (auto index : node->input_indices_) {
      auto tensor_name = model.all_tensors_[index]->name()->str();
      maps[tensor_name] = model.all_tensors_[index];
    }
  }
  return maps;
}

std::map<std::string, mindspore::schema::Tensor *> DebugInfoManager::ParseOutputTensorFromModel(const Model &model) {
  std::map<std::string, mindspore::schema::Tensor *> maps;
  for (auto node : model.all_nodes_) {
    for (auto index : node->output_indices_) {
      auto tensor_name = model.all_tensors_[index]->name()->str();
      maps[tensor_name] = model.all_tensors_[index];
    }
  }
  return maps;
}

int DebugInfoManager::GetDataFromTensorMap(const mindspore::schema::Tensor &schema_tensor,
                                           mindspore::lite::Tensor *dst_tensor) {
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
                                     mindspore::tensor::MSTensor *tensor, mindspore::lite::Tensor *new_tensor) {
  auto iter = input_tensor_map.find(tensor->tensor_name());
  if (iter == input_tensor_map.end()) {
    MS_LOG(ERROR) << tensor->tensor_name() << " find failed.";
    return RET_ERROR;
  }
  new_tensor->set_data_type(static_cast<TypeId>(iter->second->dataType()));
  new_tensor->set_shape(tensor->shape());
  new_tensor->set_quant_params(tensor->quant_params());
  new_tensor->set_tensor_name(tensor->tensor_name());
  new_tensor->set_category(static_cast<mindspore::lite::Tensor *>(tensor)->category());
  new_tensor->set_format(tensor->format());
  auto ret = GetDataFromTensorMap(*iter->second, new_tensor);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << tensor->tensor_name() << " get data from tensor map failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

KernelCallBack DebugInfoManager::GetOriginBeforeCallBack(
  const std::map<std::string, mindspore::schema::Tensor *> &input_tensor_map,
  const std::map<std::string, OpParameter *> &op_parameters) {
  auto before_callback = [&](const std::vector<mindspore::tensor::MSTensor *> &inputs,
                             const std::vector<mindspore::tensor::MSTensor *> &outputs,
                             const CallBackParam &call_param) {
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto tensor = inputs.at(i);
      MS_LOG(INFO) << " Get " << tensor->tensor_name() << " statistics info.";
      auto is_const = static_cast<mindspore::lite::Tensor *>(tensor)->category() == CONST_TENSOR ||
                      static_cast<mindspore::lite::Tensor *>(tensor)->category() == CONST_SCALAR;
      if (is_const) {
        mindspore::lite::Tensor new_tensor;
        auto ret = GetConstTensor(input_tensor_map, tensor, &new_tensor);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << tensor->tensor_name() << " get const tensor failed.";
          return false;
        }
        ret = AddOriginInfo(call_param, op_parameters.at(call_param.node_name), true, i, &new_tensor);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << tensor->tensor_name() << " add origin info failed.";
          return false;
        }
      } else {
        auto ret = AddOriginInfo(call_param, op_parameters.at(call_param.node_name), true, i,
                                 static_cast<mindspore::lite::Tensor *>(tensor));
        if (ret != RET_OK) {
          MS_LOG(ERROR) << tensor->tensor_name() << " add origin info failed.";
          return false;
        }
      }
    }
    return true;
  };
  return before_callback;
}

KernelCallBack DebugInfoManager::GetQuantBeforeCallBack(
  const std::map<std::string, mindspore::schema::Tensor *> &input_tensor_map,
  const std::map<std::string, OpParameter *> &op_parameters) {
  auto before_callback = [&](const std::vector<mindspore::tensor::MSTensor *> &inputs,
                             const std::vector<mindspore::tensor::MSTensor *> &outputs,
                             const CallBackParam &call_param) {
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto tensor = inputs.at(i);
      MS_LOG(INFO) << " Get " << tensor->tensor_name() << " statistics info.";
      if (save_flag_ && !tensor->quant_params().empty()) {
        QuantParamExtend quant_param;
        quant_param.node_name = call_param.node_name;
        quant_param.node_type = call_param.node_type;
        quant_param.quant_params = tensor->quant_params();
        quant_param.tensor_name = tensor->tensor_name();
        quant_param.element_num = tensor->ElementsNum();
        quant_param.dims = tensor->shape();
        quant_params_.push_back(quant_param);
      }
      auto is_const = static_cast<mindspore::lite::Tensor *>(tensor)->category() == CONST_TENSOR ||
                      static_cast<mindspore::lite::Tensor *>(tensor)->category() == CONST_SCALAR;
      if (is_const) {
        mindspore::lite::Tensor new_tensor;
        auto ret = GetConstTensor(input_tensor_map, tensor, &new_tensor);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << tensor->tensor_name() << " get const tensor failed.";
          return false;
        }
        ret = AddComparedInfo(call_param, inputs, op_parameters.at(call_param.node_name), true, i, &new_tensor);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << tensor->tensor_name() << " add compared info failed.";
          return false;
        }
      } else {
        auto ret = AddComparedInfo(call_param, inputs, op_parameters.at(call_param.node_name), true, i,
                                   static_cast<mindspore::lite::Tensor *>(tensor));
        if (ret != RET_OK) {
          MS_LOG(ERROR) << tensor->tensor_name() << " add compared info failed.";
          return false;
        }
      }
    }
    return true;
  };
  return before_callback;
}

KernelCallBack DebugInfoManager::GetBeforeCallBack(
  const std::map<std::string, mindspore::schema::Tensor *> &input_tensor_map,
  const std::map<std::string, OpParameter *> &op_parameters, bool is_origin) {
  if (is_origin) {
    return GetOriginBeforeCallBack(input_tensor_map, op_parameters);
  } else {
    return GetQuantBeforeCallBack(input_tensor_map, op_parameters);
  }
}

KernelCallBack DebugInfoManager::GetAfterCallBack(const std::map<std::string, OpParameter *> &op_parameters,
                                                  bool is_origin) {
  KernelCallBack after_callback;
  if (is_origin) {
    after_callback = [&](const std::vector<mindspore::tensor::MSTensor *> &inputs,
                         const std::vector<mindspore::tensor::MSTensor *> &outputs, const CallBackParam &call_param) {
      // all outputs are same dtype.
      for (size_t i = 0; i < outputs.size(); ++i) {
        auto tensor = outputs.at(i);
        if (save_flag_ && !tensor->quant_params().empty()) {
          QuantParamExtend quant_param;
          quant_param.node_name = call_param.node_name;
          quant_param.node_type = call_param.node_type;
          quant_param.quant_params = tensor->quant_params();
          quant_param.tensor_name = tensor->tensor_name();
          quant_param.element_num = tensor->ElementsNum();
          quant_param.dims = tensor->shape();
          quant_params_.push_back(quant_param);
        }
        AddOriginInfo(call_param, op_parameters.at(call_param.node_name), false, i,
                      static_cast<mindspore::lite::Tensor *>(tensor));
      }
      return true;
    };
  } else {
    after_callback = [&](const std::vector<mindspore::tensor::MSTensor *> &inputs,
                         const std::vector<mindspore::tensor::MSTensor *> &outputs, const CallBackParam &call_param) {
      // all outputs are same dtype.
      for (size_t i = 0; i < outputs.size(); ++i) {
        auto tensor = outputs.at(i);
        AddComparedInfo(call_param, inputs, op_parameters.at(call_param.node_name), false, i,
                        static_cast<mindspore::lite::Tensor *>(tensor));
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
      for (auto dim : quant_param.dims) {
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
      for (auto dim : quant_param.dims) {
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

int DebugInfoManager::GetClipAndCos() {
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
    info.clip = mindspore::lite::GetClipRate(iter->second.tensor_data.data, info.tensor_data.data,
                                             info.tensor_data.elements_num, info.tensor_data.data_type);
  }
  return RET_OK;
}

int DebugInfoManager::CompareOriginWithQuant(const quant::SessionModel &origin, const quant::SessionModel &quant,
                                             const std::map<std::string, OpParameter *> &op_parameters,
                                             const std::string &debug_info_save_path,
                                             const preprocess::DataPreProcessParam &data_preprocess) {
  auto begin = GetTimeUs();
  auto origin_input_tensor_map = ParseInputTensorFromModel(*origin.model);
  auto quant_input_tensor_map = ParseInputTensorFromModel(*quant.model);
  int ret;
  // When the calibration data set does not exist, use 1 round of random numbers for comparison
  int rounds = data_preprocess.calibrate_size > 0 ? data_preprocess.calibrate_size : 1;

  for (int round = 0; round < rounds; round++) {
    for (auto tensor : origin.session->GetInputs()) {
      if (data_preprocess.calibrate_size > 0) {
        ret = preprocess::PreProcess(data_preprocess, tensor->tensor_name(), round, tensor);
      } else {
        ret = GenerateRandomData(tensor);
      }
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "round" << round << ":" << tensor->tensor_name() << " pre-process failed.";
        return ret;
      }
    }
    std::cout << "Statistics the original data distribution. Round " << round << std::endl;
    auto origin_before_callBack = GetBeforeCallBack(origin_input_tensor_map, op_parameters, true);
    auto origin_after_callBack = GetAfterCallBack(op_parameters, true);
    origin.session->BindThread(true);
    ret = origin.session->RunGraph(origin_before_callBack, origin_after_callBack);
    origin.session->BindThread(false);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "round:" << round << " origin session run graph failed.";
      FreeBuffer();
      return ret;
    }
    std::cout << "Statistics the quant data distribution. Round " << round << std::endl;
    auto quant_before_callBack = GetBeforeCallBack(quant_input_tensor_map, op_parameters, false);
    auto quant_after_callBack = GetAfterCallBack(op_parameters, false);
    for (auto tensor : quant.session->GetInputs()) {
      auto tensor_data = tensor->MutableData();
      CHECK_NULL_RETURN(tensor_data);
      ret = memcpy_s(tensor_data, tensor->Size(), origin.session->GetInputsByTensorName(tensor->tensor_name())->data(),
                     origin.session->GetInputsByTensorName(tensor->tensor_name())->Size());
      if (ret != EOK) {
        MS_LOG(ERROR) << tensor->tensor_name() << " memcpy failed.";
        return RET_ERROR;
      }
    }
    quant.session->BindThread(true);
    ret = quant.session->RunGraph(quant_before_callBack, quant_after_callBack);
    quant.session->BindThread(false);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "round:" << round << " quant session run graph failed.";
      FreeBuffer();
      return ret;
    }
    ret = GetClipAndCos();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Get clip and cos failed.";
      return ret;
    }
    auto info_save_path = debug_info_save_path + FILE_SEPARATOR + "round" + "_" + std::to_string(round) + ".csv";
    ret = SaveInfo(info_save_path);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Failed to save debug info to " + info_save_path;
      FreeBuffer();
      return ret;
    }
    FreeBuffer();
    save_flag_ = false;
  }
  auto quant_param_save_path = debug_info_save_path + FILE_SEPARATOR + "quant_param" + ".csv";
  ret = SaveQuantParam(quant_param_save_path);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Failed to quant param to " + quant_param_save_path;
    return ret;
  }
  auto end = GetTimeUs();
  MS_LOG(INFO) << "Total time spent " << ((end - begin) / kNumUsPerMs) << " ms.\n";
  return RET_OK;
}
}  // namespace mindspore::lite
