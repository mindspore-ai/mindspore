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
#include "src/common/log_adapter.h"
#include "include/errorcode.h"
#include "tools/converter/preprocess/image_preprocess.h"

namespace mindspore::lite {
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

void DebugInfoManager::PrintInfo() {
  std::cout << ",NodeName,NodeType,TensorName,InOutFlag,DataTypeFlag,TensorTypeFlag,Min,Q1,Median,Q3,Max,Mean,Var,"
               "sparsity,Clip,"
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
              "sparsity,Clip,"
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
  return RET_OK;
}

int DebugInfoManager::SetOriginStaticInfo(QuantDebugInfo *quant_debug_info, const mindspore::lite::Tensor &tensor) {
  if (tensor.data_type() == kNumberTypeFloat32) {
    std::vector<float> tensor_data_vector(static_cast<float *>(tensor.data()),
                                          static_cast<float *>(tensor.data()) + tensor.ElementsNum());
    GetStatByTensor(tensor_data_vector, quant_debug_info);
  } else if (tensor.data_type() == kNumberTypeInt32) {
    std::vector<int> tensor_data_vector(static_cast<int *>(tensor.data()),
                                        static_cast<int *>(tensor.data()) + tensor.ElementsNum());
    GetStatByTensor(tensor_data_vector, quant_debug_info);
  } else {
    MS_LOG(ERROR) << tensor.tensor_name() << " unsupported data type " << tensor.data_type();
    return RET_ERROR;
  }
  quant_debug_info->clip = 0;

  quant_debug_info->tensor_data.data = malloc(tensor.Size());
  CHECK_MALLOC_RES(quant_debug_info->tensor_data.data, RET_NULL_PTR);
  auto ret = memcpy_s(quant_debug_info->tensor_data.data, tensor.Size(), tensor.data(), tensor.Size());
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy memory failed.";
    return false;
  }
  quant_debug_info->tensor_data.data_type = tensor.data_type();
  quant_debug_info->tensor_data.size = tensor.Size();
  quant_debug_info->tensor_data.elements_num = tensor.ElementsNum();
  return RET_OK;
}

int DebugInfoManager::SetDequantStaticInfo(OpParameter *op_parameter, int tensor_index,
                                           QuantDebugInfo *quant_debug_info, const mindspore::lite::Tensor &tensor) {
  auto preferred_dim = mindspore::lite::WeightDecoder::GetPreferredDim(op_parameter, tensor_index, tensor.shape());
  float *dequant_data;
  if (tensor.data_type() == kNumberTypeInt8) {
    dequant_data = mindspore::lite::WeightDecoder::DequantData<int8_t, float>(&tensor, preferred_dim);
  } else if (tensor.data_type() == kNumberTypeInt32) {  // Bias
    dequant_data = mindspore::lite::WeightDecoder::DequantData<int, float>(&tensor, preferred_dim);
  } else if (tensor.data_type() == kNumberTypeFloat32) {  // QuantDTypeCast(float32->int8)
    dequant_data = static_cast<float *>(tensor.data());
  } else {
    MS_LOG(ERROR) << tensor.tensor_name() << " unsupported data type " << tensor.data_type();
    return RET_ERROR;
  }
  std::vector<float> dequant_tensor_data(dequant_data, dequant_data + tensor.ElementsNum());
  GetStatByTensor(dequant_tensor_data, quant_debug_info);

  size_t buf_size = dequant_tensor_data.size() * sizeof(float);
  quant_debug_info->tensor_data.data = malloc(buf_size);
  CHECK_MALLOC_RES(quant_debug_info->tensor_data.data, RET_NULL_PTR);
  auto ret = memcpy_s(quant_debug_info->tensor_data.data, buf_size, dequant_tensor_data.data(), buf_size);
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy memory failed.";
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
  origin_debug_info.tensor_type_flag = origin_tensor->IsConst() ? WEIGHT : ACTIVATION;
  auto ret = SetOriginStaticInfo(&origin_debug_info, *origin_tensor);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << origin_tensor->tensor_name() << " get origin static info failed.";
    return RET_ERROR;
  }
  origin_debug_info.data_type_flag = ORIGIN;
  origin_debug_info.clip = 0;
  origin_debug_info.cos_similarity = 1;

  origin_debug_info.tensor_data.data = malloc(origin_tensor->Size());
  CHECK_MALLOC_RES(origin_debug_info.tensor_data.data, RET_NULL_PTR);
  ret =
    memcpy_s(origin_debug_info.tensor_data.data, origin_tensor->Size(), origin_tensor->data(), origin_tensor->Size());
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy memory failed.";
    return RET_ERROR;
  }
  origin_debug_info.tensor_data.data_type = origin_tensor->data_type();
  origin_debug_info.tensor_data.size = origin_tensor->Size();
  origin_debug_info.tensor_data.elements_num = origin_tensor->ElementsNum();
  origin_info_[origin_debug_info.primary_key] = origin_debug_info;
  return RET_OK;
}

int DebugInfoManager::AddComparedInfo(const mindspore::CallBackParam &call_back_param, OpParameter *op_parameter,
                                      bool is_input, int tensor_index, mindspore::lite::Tensor *compared_tensor) {
  CHECK_NULL_RETURN(op_parameter);
  CHECK_NULL_RETURN(compared_tensor);
  QuantDebugInfo compared_debug_info;
  compared_debug_info.primary_key.index = tensor_index;
  compared_debug_info.primary_key.node_name = call_back_param.node_name;
  compared_debug_info.primary_key.in_out_flag = is_input ? INPUT : OUTPUT;
  compared_debug_info.node_type = call_back_param.node_type;
  compared_debug_info.tensor_name = compared_tensor->tensor_name();
  compared_debug_info.data_type_flag = DEQUANT;
  compared_debug_info.tensor_type_flag = compared_tensor->IsConst() ? WEIGHT : ACTIVATION;
  if (!compared_tensor->quant_params().empty()) {
    auto ret = SetDequantStaticInfo(op_parameter, tensor_index, &compared_debug_info, *compared_tensor);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << compared_tensor->tensor_name() << " get dequant static info failed.";
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

int DebugInfoManager::GetDataFromTensorMap(const std::map<std::string, mindspore::schema::Tensor *> &maps,
                                           const std::string &tensor_name, mindspore::lite::Tensor *dst_tensor) {
  auto iter = maps.find(tensor_name);
  if (iter == maps.end()) {
    MS_LOG(ERROR) << tensor_name << " find failed.";
    return RET_ERROR;
  }
  auto src_tensor = new (std::nothrow) SchemaTensorWrapper();
  if (src_tensor == nullptr) {
    MS_LOG(ERROR) << "Create SchemaTensorWrapper return nullptr";
    return RET_ERROR;
  }
  src_tensor->Init(*iter->second, SCHEMA_CUR, "");
  auto ret = WeightDecoder::DecompressTensor(*src_tensor, dst_tensor);
  if (ret == RET_NO_CHANGE) {
    if (src_tensor->length() < dst_tensor->Size()) {
      MS_LOG(ERROR) << "Tensor data shape invalid";
      return RET_ERROR;
    }
    auto data_pair = src_tensor->ReleaseData();
    dst_tensor->set_data(data_pair.second);
    dst_tensor->set_own_data(data_pair.first);
  } else if (ret != RET_OK) {
    MS_LOG(ERROR) << "Decompress tensor data failed: " << ret;
    return ret;
  }
  delete src_tensor;
  return RET_OK;
}

KernelCallBack DebugInfoManager::GetBeforeCallBack(
  const std::map<std::string, mindspore::schema::Tensor *> &input_tensor_map,
  const std::map<std::string, OpParameter *> &op_parameters, bool is_origin) {
  KernelCallBack before_callback;
  if (is_origin) {
    before_callback = [&](const std::vector<mindspore::tensor::MSTensor *> &inputs,
                          const std::vector<mindspore::tensor::MSTensor *> &outputs, const CallBackParam &call_param) {
      for (size_t i = 0; i < inputs.size(); ++i) {
        auto tensor = inputs.at(i);
        if (tensor->IsConst()) {
          auto ret = GetDataFromTensorMap(input_tensor_map, tensor->tensor_name(),
                                          static_cast<mindspore::lite::Tensor *>(tensor));
          if (ret != RET_OK) {
            MS_LOG(ERROR) << tensor->tensor_name() << " get data from tensor map failed.";
            return false;
          }
        }
        AddOriginInfo(call_param, op_parameters.at(call_param.node_name), true, i,
                      static_cast<mindspore::lite::Tensor *>(tensor));
      }
      return true;
    };
  } else {
    before_callback = [&](const std::vector<mindspore::tensor::MSTensor *> &inputs,
                          const std::vector<mindspore::tensor::MSTensor *> &outputs, const CallBackParam &call_param) {
      for (size_t i = 0; i < inputs.size(); ++i) {
        auto tensor = inputs.at(i);
        if (tensor->IsConst()) {
          auto ret = GetDataFromTensorMap(input_tensor_map, tensor->tensor_name(),
                                          static_cast<mindspore::lite::Tensor *>(tensor));
          if (ret != RET_OK) {
            MS_LOG(ERROR) << tensor->tensor_name() << " get data from tensor map failed.";
            return false;
          }
        }
        AddComparedInfo(call_param, op_parameters.at(call_param.node_name), true, i,
                        static_cast<mindspore::lite::Tensor *>(tensor));
      }
      return true;
    };
  }
  return before_callback;
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
        AddComparedInfo(call_param, op_parameters.at(call_param.node_name), false, i,
                        static_cast<mindspore::lite::Tensor *>(tensor));
      }
      return true;
    };
  }

  return after_callback;
}

int DebugInfoManager::CompareOriginWithDequant(const quant::SessionModel &origin, const quant::SessionModel &dequant,
                                               const preprocess::DataPreProcessParam &data_preprocess,
                                               const std::string &debug_info_save_path,
                                               const std::map<std::string, OpParameter *> &op_parameters) {
  auto origin_input_tensor_map = ParseInputTensorFromModel(*origin.model);
  auto dequant_input_tensor_map = ParseInputTensorFromModel(*dequant.model);

  for (int round = 0; round < data_preprocess.calibrate_size; round++) {
    for (auto tensor : origin.session->GetInputs()) {
      auto ret = preprocess::PreProcess(data_preprocess, tensor->tensor_name(), round, tensor);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "round" << round << ":" << tensor->tensor_name() << " pre-process failed.";
        return ret;
      }
    }
    MS_LOG(INFO) << "round:" << round << " origin session run graph.";
    auto origin_before_callBack = GetBeforeCallBack(origin_input_tensor_map, op_parameters, true);
    auto origin_after_callBack = GetAfterCallBack(op_parameters, true);
    auto ret = origin.session->RunGraph(origin_before_callBack, origin_after_callBack);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "round:" << round << " origin session run graph failed.";
      FreeBuffer();
      return ret;
    }
    MS_LOG(INFO) << "round:" << round << " dequant session run graph.";
    auto dequant_before_callBack = GetBeforeCallBack(dequant_input_tensor_map, op_parameters, false);
    auto dequant_after_callBack = GetAfterCallBack(op_parameters, false);
    ret = dequant.session->RunGraph(dequant_before_callBack, dequant_after_callBack);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "round:" << round << " dequant session run graph failed.";
      FreeBuffer();
      return ret;
    }
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
      info.cos_similarity =
        mindposre::lite::GetCosSimilarity(iter->second.tensor_data.data, info.tensor_data.data,
                                          info.tensor_data.elements_num, info.tensor_data.data_type);
      info.clip = mindposre::lite::GetClipRate(iter->second.tensor_data.data, info.tensor_data.data,
                                               info.tensor_data.elements_num, info.tensor_data.data_type);
    }
    PrintInfo();
    SaveInfo(debug_info_save_path);
    FreeBuffer();
  }
  return RET_OK;
}
}  // namespace mindspore::lite
