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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_GPTQ_QUANTIZER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_GPTQ_QUANTIZER_H_

#include <map>
#include <set>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <utility>
#include "schema/inner/model_generated.h"
#include "base/base.h"
#include "tools/converter/session/dynamic_session.h"
#include "tools/converter/cxx_api/converter_para.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "nnacl/matmul_parameter.h"

namespace mindspore {
namespace lite::quant {

struct WeightInfo {
  float *weight_data = nullptr;
  int8_t *quant_data = nullptr;
  size_t elements_num{0};
  int input_index{0};
  std::vector<schema::QuantParamT> quant_params;
};

class GptqQuantizer {
 public:
  GptqQuantizer() {}

  GptqQuantizer(const FuncGraphPtr &func_graph, const std::shared_ptr<ConverterPara> &param,
                std::set<PrimitivePtr> support_primitive_types)
      : func_graph_(func_graph), param_(param), support_primitive_types_(support_primitive_types) {
    auto context = std::make_shared<InnerContext>();
    context->thread_num_ = 4;
    dynamic_session_ = std::make_shared<DynamicSession>(context);
    batch_num_ = param_->dataPreProcessParam.calibrate_size;
  }

  ~GptqQuantizer();

  int DoQuantize();

 private:
  int FilterWeightNode(const FuncGraphPtr &func_graph, const std::set<PrimitivePtr> support_primitive_types,
                       std::map<std::string, std::unique_ptr<WeightInfo>> *weights);

  int ExtractWeightParams(schema::MetaGraphT *meta_graph, std::map<std::string, std::unique_ptr<WeightInfo>> *weights);

  int CompileModel(std::shared_ptr<DynamicSession> dynamic_session, const schema::MetaGraphT &meta_graph,
                   const std::set<std::string> &weight_names);

  int GenerateInputData(lite::Tensor *tensor, const lite::preprocess::DataPreProcessParam &preprocess_param);

  bool CheckTensorDtype(const lite::Tensor &input_tensor, const lite::Tensor &weight_tensor);

  int GetMatMulDeep(const std::vector<int> &weight_dims, const MatMulParameter *op_param, int input_index);

  int DequantWeight(WeightInfo *weight_info, const lite::Tensor *weight_tensor, int prefer_dim);

  int RunKernel();

  int UpdateWeightNode(const FuncGraphPtr &func_graph, const std::set<PrimitivePtr> support_weight_quant_types,
                       const std::map<std::string, std::unique_ptr<WeightInfo>> &weights);
  void setModel(Model *model) { this->model_ = model; }

  template <typename T>
  int AddBatch(const lite::Tensor &tensor, float *hessian_data, int deep, int batch_num, bool transpose);

  template <typename TK, typename TV>
  std::set<TK> extract_keys(std::map<TK, TV> const &input_map) {
    std::set<TK> retval;
    for (auto const &element : input_map) {
      retval.insert(element.first);
    }
    return retval;
  }

  FuncGraphPtr func_graph_ = nullptr;
  std::shared_ptr<ConverterPara> param_ = nullptr;
  std::set<PrimitivePtr> support_primitive_types_;
  std::shared_ptr<DynamicSession> dynamic_session_ = nullptr;
  schema::MetaGraphT *meta_graph_ = nullptr;
  int batch_num_{0};
  std::map<std::string, std::unique_ptr<WeightInfo>> weights_;
  Model *model_ = nullptr;
};
}  // namespace lite::quant
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_GPTQ_QUANTIZER_H_
