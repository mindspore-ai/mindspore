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

#define USE_DEPRECATED_API

#include "tools/converter/quantizer/quantization_optimizer.h"
#include <memory>
#include <string>
#include <deque>
#include <map>
#include <set>
#include "tools/lite_exporter/fetch_content.h"
#include "base/base.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/converter/quantizer/weight_quantizer.h"
#include "tools/converter/quantizer/full_quant_quantizer.h"
#include "tools/converter/quantizer/debug_info_manager.h"
#include "tools/converter/quantizer/parameter_tunner.h"
#include "tools/converter/quantizer/dynamic_quantizer.h"
#include "tools/lite_exporter/anf_exporter.h"
#include "tools/converter/quantizer/cle_strategy.h"
#include "tools/optimizer/common/pass_manager_extends.h"
#include "tools/optimizer/fusion/quant_dtype_cast_fusion.h"
#include "backend/common/optimizer/graph_optimizer.h"
#include "tools/optimizer/graph/infershape_pass.h"

namespace mindspore::lite::quant {
int DoFullQuant(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param) {
  auto quantizer = std::make_unique<FullQuantQuantizer>(param);
  if (quantizer == nullptr) {
    MS_LOG(ERROR) << "New FullQuantQuantizer failed";
    return RET_ERROR;
  }
  auto status = quantizer->DoQuantize(old_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "DoQuantization failed " << status;
    return RET_ERROR;
  }
  return RET_OK;
}

int DoWeightQuant(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param) {
  double init_scale = param->mixedBitWeightQuantParam.init_scale;
  if (param->commonQuantParam.bit_num == 0 && param->mixedBitWeightQuantParam.auto_tune) {
    ParameterOptimizer optimizer;
    auto status = optimizer.GridSearchForScale(old_graph, param, &init_scale);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Grid search with scale failed.";
      return status;
    }
    auto quantizer = std::make_unique<WeightQuantizer>(param, init_scale);
    if (quantizer == nullptr) {
      MS_LOG(ERROR) << "New WeightQuantizer failed";
      return RET_ERROR;
    }
    status = static_cast<WeightQuantizer *>(quantizer.get())->DoQuantize(old_graph);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "DoQuantization failed " << status;
      return RET_ERROR;
    }
  } else {
    auto quantizer = std::make_unique<WeightQuantizer>(param);
    if (quantizer == nullptr) {
      MS_LOG(ERROR) << "New WeightQuantizer failed";
      return RET_ERROR;
    }
    auto status = quantizer->DoQuantize(old_graph);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "DoQuantization failed " << status;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int DoDynamicQuant(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param) {
  auto quantizer = std::make_unique<DynamicQuantizer>(param);
  if (quantizer == nullptr) {
    MS_LOG(ERROR) << "New DynamicQuantizer failed";
    return RET_ERROR;
  }
  auto status = quantizer->DoQuantize(old_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "DoQuantization failed " << status;
    return RET_ERROR;
  }
  return RET_OK;
}

lite::LiteModel *ParseLiteModel(const FuncGraphPtr &func_graph, const std::shared_ptr<ConverterPara> &param) {
  auto meta_graph = Export(func_graph, true, true);
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "Export to meta_graph failed";
    return static_cast<lite::LiteModel *>(nullptr);
  }

  // transform
  GraphDefTransform fb_transform;
  fb_transform.SetGraphDef(meta_graph);
  auto status = fb_transform.Transform(param);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "FBTransform model failed";
    delete meta_graph;
    return static_cast<LiteModel *>(nullptr);
  }
  meta_graph->version = Version();

  flatbuffers::FlatBufferBuilder builder(kMaxNum1024);
  auto offset = schema::MetaGraph::Pack(builder, meta_graph);
  builder.Finish(offset);
  schema::FinishMetaGraphBuffer(builder, offset);
  size_t size = builder.GetSize();
  auto content = reinterpret_cast<const char *>(builder.GetBufferPointer());
  if (content == nullptr) {
    MS_LOG(ERROR) << "GetBufferPointer nullptr";
    return static_cast<LiteModel *>(nullptr);
  }
  return static_cast<LiteModel *>(LiteModel::Import(content, size));
}

int DoQuantDebug(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param,
                 const std::shared_ptr<mindspore::Model> &origin_model,
                 const mindspore::lite::LiteModel *origin_lite_model) {
  auto quant_model = std::make_shared<mindspore::Model>();
  CHECK_NULL_RETURN(quant_model);
  size_t size = 0;
  auto status = BuildModelByFuncGraph(quant_model, old_graph, param, &size);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Build model failed";
    return RET_ERROR;
  }
  std::map<std::string, OpParameter *> op_parameters;
  auto ret = FetchOpParameterFromFuncGraph(old_graph, &op_parameters);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Fetch op parameter from funcgraph failed";
    return ret;
  }
  DebugInfoManager manager;

  auto quant_lite_model = ParseLiteModel(old_graph, param);
  if (quant_lite_model == nullptr) {
    MS_LOG(ERROR) << "Parse quant lite model failed";
    return RET_ERROR;
  }
  if (origin_lite_model == nullptr) {
    MS_LOG(ERROR) << "Origin lite model nullptr.";
    return RET_ERROR;
  }
  ret = manager.CompareOriginWithQuant(origin_model, quant_model, op_parameters, param, *origin_lite_model,
                                       *quant_lite_model);
  auto free_buffer = [&] {
    for (auto parameter : op_parameters) {
      if (parameter.second != nullptr) {
        free(parameter.second);
        parameter.second = nullptr;
      }
    }
    op_parameters.clear();
  };
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Compare origin with quant failed.";
    free_buffer();
    return ret;
  }
  free_buffer();
  return RET_OK;
}

int ConvertFp16ToFp32(const FuncGraphPtr &old_graph) {
  auto cnodes = old_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    for (size_t i = kPrimOffset; i < cnode->size(); ++i) {
      auto input = cnode->input(i);
      if (!input->isa<Parameter>() || !input->cast<ParameterPtr>()->has_default()) {
        continue;
      }
      ParameterPtr param_node;
      tensor::TensorPtr tensor_info;
      GetLiteParameter(input, &param_node, &tensor_info);
      CHECK_NULL_RETURN(tensor_info);
      CHECK_NULL_RETURN(param_node);
      if (tensor_info->data_type() == kNumberTypeFloat16) {
        MS_LOG(INFO) << "convert " << input->fullname_with_scope() << " from fp16 to fp32.";
        auto data = static_cast<float16 *>(tensor_info->data_c());
        std::vector<float> fp32_data(tensor_info->DataSize());
        for (size_t j = 0; j < tensor_info->DataSize(); j++) {
          fp32_data[j] = mindspore::Float16::ToFloat32(data[j]);
        }
        mindspore::tensor::TensorPtr tensor_ptr = std::make_shared<mindspore::tensor::Tensor>(
          kNumberTypeFloat32, tensor_info->shape_c(), fp32_data.data(), fp32_data.size() * sizeof(float));
        param_node->set_default_param(tensor_ptr);
        param_node->set_abstract(tensor_ptr->ToAbstract());
      }
    }
  }
  return RET_OK;
}

int DoSingleGraphQuantize(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param) {
  CHECK_NULL_RETURN(param);
  if (param->commonQuantParam.quant_type == schema::QuantType_QUANT_NONE) {
    return RET_OK;
  }
  int status;

  status = ConvertFp16ToFp32(old_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Convert fp16 To fp32 failed.";
    return status;
  }

  bool per_layer = param->commonQuantParam.quant_type == schema::QuantType_QUANT_ALL &&
                   !param->fullQuantParam.per_channel && param->fullQuantParam.target_device != DSP;
  if (per_layer) {
    CLEStrategy cle_strategy(old_graph);
    status = cle_strategy.Run();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "do pre process failed!";
      return status;
    }
  }

  std::shared_ptr<mindspore::Model> origin;
  lite::LiteModel *origin_lite_model = nullptr;
  if (param->commonQuantParam.is_debug) {  // Bak fp32 model for debug
    auto quant_type = param->commonQuantParam.quant_type;
    param->commonQuantParam.quant_type = schema::QuantType_QUANT_NONE;
    origin = std::make_shared<mindspore::Model>();
    CHECK_NULL_RETURN(origin);
    size_t size = 0;
    auto ret = BuildModelByFuncGraph(origin, old_graph, param, &size);
    param->commonQuantParam.quant_type = quant_type;
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Build model failed";
      return RET_ERROR;
    }
    origin_lite_model = ParseLiteModel(old_graph, param);
    if (origin_lite_model == nullptr) {
      MS_LOG(ERROR) << "Parse lite model failed.";
      return RET_ERROR;
    }
  }
  if (param->commonQuantParam.quant_type == schema::QuantType_QUANT_ALL) {  // Full Quantization
    status = DoFullQuant(old_graph, param);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Do full quant failed.";
      return status;
    }
  } else if (param->commonQuantParam.quant_type == schema::QuantType_QUANT_WEIGHT) {  // Weight Quantization
    status = DoWeightQuant(old_graph, param);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Do weight quant failed.";
      return status;
    }
  } else if (param->commonQuantParam.quant_type == schema::QuantType_QUANT_DYNAMIC) {  // Dynamic Quantization
    status = DoDynamicQuant(old_graph, param);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Do dynamic quant failed.";
      return status;
    }
  }

  {
    auto optimizer = std::make_shared<opt::GraphOptimizer>();
    CHECK_NULL_RETURN(optimizer);
    auto fusion_pm = std::make_shared<opt::LitePassManager>("fusion pass manager after quant", false);
    CHECK_NULL_RETURN(fusion_pm);
    fusion_pm->AddPass(std::make_shared<opt::QuantDtypeCastFusion>());
    fusion_pm->AddPass(std::make_shared<opt::InferShapePass>(param->fmk_type, param->train_model));
    optimizer->AddPassManager(fusion_pm);
    if (optimizer->Optimize(old_graph) == nullptr) {
      MS_LOG(ERROR) << "run cast node fusion failed.";
      return RET_ERROR;
    }
  }

  if (param->commonQuantParam.is_debug) {
    status = DoQuantDebug(old_graph, param, origin, origin_lite_model);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Do quant debug failed.";
      return status;
    }
  }
  return RET_OK;
}

int QuantizationOptimizer::Run(const mindspore::FuncGraphPtr &func_graph) {
  std::set<FuncGraphPtr> all_func_graphs{};
  quant::GetFuncGraphs(func_graph, &all_func_graphs);
  // Support for multi-subgraph models
  for (auto &item : all_func_graphs) {
    auto status = DoSingleGraphQuantize(item, param_);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Do Quantize failed.";
      return status;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
