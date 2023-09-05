/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "tools/converter/quantizer/gptq_quantizer.h"
#include "tools/common/node_util.h"
#include "tools/converter/converter_metagraph.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "include/errorcode.h"
#include "tools/converter/quantizer/eigen_util.h"
#include "tools/converter/quantizer/gptq.h"
#include "tools/converter/preprocess/image_preprocess.h"
#include "ops/mat_mul.h"
#include "src/litert/weight_decoder.h"

namespace mindspore::lite::quant {
GptqQuantizer::~GptqQuantizer() {
  if (meta_graph_ != nullptr) {
    delete meta_graph_;
    meta_graph_ = nullptr;
  }
  if (model_ != nullptr) {
    model_->buf = nullptr;
  }
  delete (model_);
  model_ = nullptr;
}
int GptqQuantizer::FilterWeightNode(const FuncGraphPtr &func_graph,
                                    const std::set<PrimitivePtr> support_weight_quant_types,
                                    std::map<std::string, std::unique_ptr<WeightInfo>> *weights) {
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    // filter matmul op
    if (!CheckNodeInSet(cnode, support_weight_quant_types)) {
      continue;
    }
    MS_LOG(INFO) << cnode->fullname_with_scope() << " start gptq quantization.";
    for (size_t i = 1; i < cnode->size(); i++) {
      auto input_node = cnode->input(i);
      MS_CHECK_TRUE_RET(input_node != nullptr, RET_NULL_PTR);
      if (!input_node->isa<mindspore::Parameter>() || !input_node->cast<ParameterPtr>()->has_default()) {
        continue;
      }
      auto weight_tensor_name = input_node->fullname_with_scope();
      auto weight_info = std::make_unique<WeightInfo>();
      CHECK_NULL_RETURN(weight_info);
      weight_info->input_index = i - kPrimOffset;
      weights->insert(std::pair<std::string, std::unique_ptr<WeightInfo>>(weight_tensor_name, std::move(weight_info)));
    }
  }
  return RET_OK;
}

// extract weight params
int GptqQuantizer::ExtractWeightParams(schema::MetaGraphT *meta_graph,
                                       std::map<std::string, std::unique_ptr<WeightInfo>> *weights) {
  for (auto &tensor : meta_graph->allTensors) {
    if (weights->find(tensor->name) == weights->end()) {
      continue;
    }
    MS_LOG(INFO) << "Extract weight params, tensor name: " << tensor->name;
    weights->at(tensor->name)->weight_data = reinterpret_cast<float *>(tensor->data.data());
    tensor->data = {};
    tensor->nodeType = NodeType_CNode;
  }
  return RET_OK;
}

int GptqQuantizer::CompileModel(std::shared_ptr<DynamicSession> dynamic_session, const schema::MetaGraphT &meta_graph,
                                const std::set<std::string> &weight_names) {
  CHECK_NULL_RETURN(dynamic_session);
  size_t length = 0;
  flatbuffers::FlatBufferBuilder builder(1024);
  auto packed_buffer = MetaGraphSerializer::GetMetaGraphPackedBuff(&builder, meta_graph, &length);
  auto model = ImportFromBuffer(reinterpret_cast<const char *>(packed_buffer), length, true);
  CHECK_NULL_RETURN(model);
  auto status = dynamic_session->CompileGraph(model);
  model->buf = nullptr;
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Compile graph failed.";
    delete model;
    return status;
  }
  dynamic_session->SetWeightsName(weight_names);
  setModel(model);
  return RET_OK;
}

int GptqQuantizer::GenerateInputData(lite::Tensor *tensor,
                                     const lite::preprocess::DataPreProcessParam &preprocess_param) {
  CHECK_NULL_RETURN(tensor);
  return preprocess::PreProcessBatch(preprocess_param, tensor->tensor_name(), tensor);
}

bool GptqQuantizer::CheckTensorDtype(const lite::Tensor &input_tensor, const lite::Tensor &weight_tensor) {
  if (input_tensor.data_type() != kNumberTypeFloat32 && input_tensor.data_type() != kNumberTypeFloat) {
    MS_LOG(ERROR) << input_tensor.tensor_name() << ", input tensor data_type not supported.";
    return false;
  }
  if (weight_tensor.data_type() != kNumberTypeFloat32 && weight_tensor.data_type() != kNumberTypeFloat) {
    MS_LOG(ERROR) << input_tensor.tensor_name() << ", weight tensor data_type not supported.";
    return false;
  }
  return true;
}

int GptqQuantizer::GetMatMulDeep(const std::vector<int> &weight_dims, const MatMulParameter *op_param,
                                 int input_index) {
  int last_first_index = static_cast<int>(weight_dims.size()) - 1;
  int last_second_index = static_cast<int>(weight_dims.size()) - 2;
  // input a
  if (input_index == kInputIndex) {
    if (op_param->a_transpose_) {
      return weight_dims[last_second_index];
    } else {
      return weight_dims[last_first_index];
    }
  } else if (input_index == kWeightIndex) {
    if (op_param->b_transpose_) {
      return weight_dims[last_first_index];
    } else {
      return weight_dims[last_second_index];
    }
  }
  return weight_dims[last_second_index];
}

int GptqQuantizer::DequantWeight(WeightInfo *weight_info, const lite::Tensor *weight_tensor, int prefer_dim) {
  auto dims = weight_tensor->shape();
  for (int i = 0; i < weight_tensor->ElementsNum(); i++) {
    auto bucket_index = GetBucketIndex(dims, prefer_dim, i);
    MS_CHECK_GT(static_cast<int>(weight_info->quant_params.size()), bucket_index, RET_ERROR);
    auto quant_param = weight_info->quant_params.at(bucket_index);
    auto quant_data = weight_info->quant_data[i];
    weight_info->weight_data[i] = quant_param.scale * (quant_data - quant_param.zeroPoint);
  }
  return RET_OK;
}

template <typename T>
int GptqQuantizer::AddBatch(const lite::Tensor &tensor, float *hessian_data, int deep, int batch_num, bool transpose) {
  CHECK_NULL_RETURN(hessian_data);
  auto tensor_data = reinterpret_cast<T *>(tensor.data());
  CHECK_NULL_RETURN(tensor_data);
  int nsamples = 0;
  MS_CHECK_TRUE_RET(!tensor.shape().empty(), RET_ERROR);
  int tmp = tensor.shape()[0];
  size_t data_size = deep * deep;
  for (size_t i = 0; i < data_size; i++) {
    *(hessian_data + i) = 0.0;
  }
  for (int batch_index = 0; batch_index < batch_num; batch_index++) {
    float *hessian_tmp = reinterpret_cast<float *>(malloc(data_size * sizeof(float)));
    MS_CHECK_TRUE_MSG(hessian_tmp != nullptr, RET_NULL_PTR, "Malloc hessian data failed.");

    auto batch_data = tensor_data + batch_index * tensor.ElementsNum();
    if (quant::CalculateHessianMatrix<T>(batch_data, hessian_tmp, tensor.shape()[0], tensor.shape()[1], transpose) !=
        RET_OK) {
      MS_LOG(ERROR) << "Calculate Hessian matrix failed, tensor name: " << tensor.tensor_name()
                    << " batch index: " << batch_index;
      free(hessian_tmp);
      return RET_ERROR;
    }

    for (int32_t i = 0; i < deep; i++) {
      for (int32_t j = 0; j < deep; j++) {
        // self.H *= self.nsamples / (self.nsamples + tmp)
        *(hessian_data + i * deep + j) *= nsamples / (nsamples + tmp);
        // self.nsamples += tmp
        // self.H += (2 * inp.matmul(inp.t())) / self.nsamples
        *(hessian_data + i * deep + j) += *(hessian_tmp + i * deep + j) / (nsamples + tmp);
      }
    }
    nsamples += tmp;
    free(hessian_tmp);
  }
  return RET_OK;
}

int GptqQuantizer::UpdateWeightNode(const FuncGraphPtr &func_graph,
                                    const std::set<PrimitivePtr> support_weight_quant_types,
                                    const std::map<std::string, std::unique_ptr<WeightInfo>> &weights) {
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    if (!CheckNodeInSet(cnode, support_weight_quant_types)) {
      continue;
    }
    for (size_t i = 1; i < cnode->size(); i++) {
      auto input_node = cnode->input(i);
      CHECK_NULL_RETURN(input_node);
      auto weight_tensor_name = input_node->fullname_with_scope();
      if (input_node->isa<mindspore::Parameter>() && weights.find(weight_tensor_name) != weights.end()) {
        auto parameter = input_node->cast<ParameterPtr>();
        CHECK_NULL_RETURN(parameter);
        MS_CHECK_TRUE_MSG(parameter->has_default(), RET_ERROR, "Parameter has no default_param.");
        auto weight_tensor = parameter->default_param()->cast<tensor::TensorPtr>();
        MS_CHECK_TRUE_MSG(weight_tensor != nullptr, RET_ERROR, "default_param can not cast to tensor::Tensor.");
        weight_tensor->set_data_type(kNumberTypeInt8);
        size_t new_size = weights.at(weight_tensor_name)->elements_num * sizeof(int8_t);
        if (new_size != static_cast<size_t>(weight_tensor->data().nbytes())) {
          MS_LOG(ERROR) << "Data size of tensor info is error, new_size: " << new_size
                        << ", weight nbytes: " << static_cast<size_t>(weight_tensor->data().nbytes());
          return RET_ERROR;
        }
        if (memcpy_s(weight_tensor->data_c(), weight_tensor->data().nbytes(),
                     weights.at(weight_tensor_name)->quant_data, new_size) != EOK) {
          MS_LOG(ERROR) << "memcpy data failed.";
          return RET_ERROR;
        }
        auto ret = UpdateDataType(input_node, kNumberTypeInt8);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << input_node->fullname_with_scope() << " set new dtype failed.";
          return ret;
        }
        auto quantization_ptr =
          quant::ConvertQuantParamTToQuantizationParam(weights.at(weight_tensor_name)->quant_params);
        CHECK_NULL_RETURN(quantization_ptr);
        weight_tensor->set_quant_param(std::vector<std::shared_ptr<mindspore::QuantizationParam>>{quantization_ptr});

        auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
        CHECK_NULL_RETURN(primitive);
        primitive->AddAttr(quant::kQuantType, MakeValue(static_cast<int>(quant::QUANT_WEIGHT)));
      }
    }
  }
  return RET_OK;
}

int GptqQuantizer::RunKernel() {
  auto beforeCallBack = [&](kernel::KernelExec *kernel, const MSCallBackParam &kernel_info, int sample_num) -> bool {
    if (kernel->type() != schema::PrimitiveType_MatMulFusion) {
      MS_LOG(DEBUG) << kernel->name() << " not need gptq quantizer.";
      return true;
    }
    auto in_tensors = kernel->in_tensors();
    MS_CHECK_TRUE_MSG(in_tensors.size() >= 2, false, "Input tensors size less than 2.");
    auto op_param = reinterpret_cast<MatMulParameter *>(kernel->op_parameter());
    lite::Tensor *input_tensor = nullptr;
    lite::Tensor *weight_tensor = nullptr;
    if (weights_.find(in_tensors.at(kWeightIndex)->tensor_name()) != weights_.end()) {
      input_tensor = in_tensors.at(kInputIndex);
      weight_tensor = in_tensors.at(kWeightIndex);
    } else if (weights_.find(in_tensors.at(kInputIndex)->tensor_name()) != weights_.end()) {
      input_tensor = in_tensors.at(kWeightIndex);
      weight_tensor = in_tensors.at(kInputIndex);
    } else {
      MS_LOG(ERROR) << "weights not exist, kernel name: " << kernel->name();
      return false;
    }
    if (!CheckTensorDtype(*input_tensor, *weight_tensor)) {
      MS_LOG(ERROR) << "CheckTensorDtype failed.";
      return false;
    }
    auto tensor_name = weight_tensor->tensor_name();
    MS_LOG(INFO) << "Calculate Hessian matrix for tensor: " << input_tensor->tensor_name();
    auto &weight_info = weights_[tensor_name];
    int64_t deep = GetMatMulDeep(weight_tensor->shape(), op_param, weight_info->input_index);
    size_t hessian_size = static_cast<size_t>(deep * deep);
    float *hessian_data = reinterpret_cast<float *>(malloc(hessian_size * sizeof(float)));
    MS_CHECK_TRUE_MSG(hessian_data != nullptr, false, "Malloc hessian data failed.");
    bool transpose = (weight_info->input_index == kWeightIndex);
    if (AddBatch<float>(*input_tensor, hessian_data, deep, batch_num_, transpose) != RET_OK) {
      MS_LOG(ERROR) << "AddBatch failed, tensor name: " << input_tensor->tensor_name();
      return false;
    }
    if (weight_info->quant_data != nullptr) {
      MS_LOG(INFO) << "weight already be quantized, tensor name: " << tensor_name;
      return true;
    }
    size_t elements_num = static_cast<size_t>(weight_tensor->ElementsNum());
    weight_info->elements_num = elements_num;
    weight_info->quant_data = reinterpret_cast<int8_t *>(malloc(sizeof(int8_t) * elements_num));
    MS_CHECK_TRUE_MSG(weight_info->quant_data != nullptr, RET_ERROR, "malloc quant data memory failed.");
    int bit_num = param_->commonQuantParam.bit_num;
    auto quantizer =
      std::make_unique<quant::Gptq>(weight_tensor, weight_info.get(), hessian_data, deep, bit_num, transpose, op_param);
    if (quantizer->DoQuantize() != RET_OK) {
      MS_LOG(ERROR) << "FasterQuantizer failed.";
      return false;
    }
    // update weights when current block quantization done.
    // self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
    auto prefer_dim =
      WeightDecoder::GetMatMulPreferredDim(kernel->op_parameter(), weight_info->input_index, weight_tensor->shape());
    if (DequantWeight(weight_info.get(), weight_tensor, prefer_dim) != RET_OK) {
      MS_LOG(ERROR) << "DequantWeight failed.";
      return false;
    }
    weight_tensor->set_data(weight_info->weight_data, false);
    free(hessian_data);
    return true;
  };
  // set calibration data into tensor
  dynamic_session_->SetSampleNum(batch_num_);
  for (auto input_tensor : dynamic_session_->GetInputs()) {
    auto unit_size = input_tensor->Size();
    auto data = input_tensor->allocator()->Malloc(unit_size * batch_num_);
    input_tensor->set_data(data);
    if (GenerateInputData(input_tensor, param_->dataPreProcessParam) != RET_OK) {
      MS_LOG(ERROR) << "Generate input data failed, tensor name: " << input_tensor->tensor_name();
      return RET_ERROR;
    }
    MS_LOG(WARNING) << "input_tensor set data, tensor name: " << input_tensor->tensor_name()
                    << " tensor size: " << input_tensor->Size();
  }
  size_t layer_num = dynamic_session_->GetKernelNum();
  for (size_t index = 0; index < layer_num; index++) {
    auto ret = dynamic_session_->RunKernel(index, beforeCallBack, nullptr);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Run kernel failed.";
      return ret;
    }
  }
  return RET_OK;
}

int GptqQuantizer::DoQuantize() {
  // step1 extract weight data from func_graph, optimize weights memory
  if (FilterWeightNode(func_graph_, support_primitive_types_, &weights_) != RET_OK) {
    MS_LOG(ERROR) << "Filter weight node failed.";
    return RET_ERROR;
  }
  meta_graph_ = ConverterToMetaGraph::Build(param_, func_graph_);
  if (meta_graph_ == nullptr) {
    MS_LOG(ERROR) << "Convert to meta graph failed.";
    return RET_ERROR;
  }
  if (ExtractWeightParams(meta_graph_, &weights_) != RET_OK) {
    MS_LOG(ERROR) << "Extract weight params failed.";
    return RET_ERROR;
  }
  // step2 compile model
  std::set<std::string> weight_names = extract_keys<std::string, std::unique_ptr<WeightInfo>>(weights_);
  if (CompileModel(dynamic_session_, *meta_graph_, weight_names) != RET_OK) {
    MS_LOG(ERROR) << "Compile model failed.";
    return RET_ERROR;
  }
  // step3 gptq quantize layer by layer
  if (RunKernel() != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  // step4 modify func_graph, weight info
  if (UpdateWeightNode(func_graph_, support_primitive_types_, weights_) != RET_OK) {
    MS_LOG(ERROR) << "Update weight node failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
