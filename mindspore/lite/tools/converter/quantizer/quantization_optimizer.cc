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
#include "tools/optimizer/graph/redundant_op_remove_pass.h"
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
#include "include/backend/optimizer/graph_optimizer.h"
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

std::shared_ptr<lite::Model> ParseLiteModel(const FuncGraphPtr &func_graph,
                                            const std::shared_ptr<ConverterPara> &param) {
  FuncGraphPtr func_graph_clone;
  if (CloneFuncGraph(func_graph, param, &func_graph_clone) != RET_OK) {
    MS_LOG(ERROR) << "Clone func_graph failed";
    return nullptr;
  }
  auto meta_graph = Export(func_graph_clone, true, true);
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "Export to meta_graph failed";
    return nullptr;
  }

  // transform
  GraphDefTransform fb_transform;
  fb_transform.SetGraphDef(meta_graph);
  auto status = fb_transform.Transform(param);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "FBTransform model failed";
    delete meta_graph;
    return nullptr;
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
    delete meta_graph;
    return nullptr;
  }
  delete meta_graph;
  return std::shared_ptr<lite::Model>(LiteModel::Import(content, size));
}

int DoQuantDebug(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param,
                 const std::shared_ptr<mindspore::Model> &origin_model,
                 const std::shared_ptr<lite::Model> &origin_lite_model) {
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

  ret = manager.CompareOriginWithQuant(origin_model, quant_model, op_parameters, param, origin_lite_model,
                                       quant_lite_model);
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

int ConvertValueNodeToParameter(const FuncGraphPtr &func_graph) {
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    for (size_t i = kPrimOffset; i < cnode->size(); ++i) {
      auto input = cnode->input(i);
      if (!input->isa<ValueNode>()) {
        continue;
      }
      auto tensor_info = input->cast<ValueNodePtr>()->value()->cast<tensor::TensorPtr>();
      if (tensor_info == nullptr) {
        MS_LOG(INFO) << cnode->fullname_with_scope() << " input index: " << i << " cast tensor nullptr.";
        continue;
      }
      auto parameter = func_graph->add_parameter();
      auto status = InitParameterFromTensorInfo(parameter, tensor_info);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "Init parameter From tensor failed, tenor: " << tensor_info->name();
        return status;
      }
      parameter->set_name(input->fullname_with_scope());
      auto manage = Manage(func_graph);
      manage->Replace(input, parameter);
    }
  }
  return RET_OK;
}

int PrepareQuantize(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param) {
  if (!param->train_model) {
    auto status = ConvertValueNodeToParameter(old_graph);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Convert value node To parameter failed.";
      return status;
    }
  }

  auto convert_pm = std::make_shared<opt::LitePassManager>("anf graph convert pass manager", true);
  convert_pm->AddPass(std::make_shared<opt::RemoveRedundantOpPass>(param->train_model));
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  optimizer->AddPassManager(convert_pm);
  if (optimizer->Optimize(old_graph) == nullptr) {
    MS_LOG(ERROR) << "run graph pass failed";
    return RET_ERROR;
  }

  bool per_layer = param->commonQuantParam.quant_type == quant::QUANT_ALL && !param->fullQuantParam.per_channel &&
                   param->fullQuantParam.target_device != DSP;
  if (per_layer) {
    CLEStrategy cle_strategy(old_graph);
    auto status = cle_strategy.Run();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "do pre process failed!";
      return status;
    }
  }
  return RET_OK;
}

int DoSingleGraphQuantize(const FuncGraphPtr &func_graph, const std::shared_ptr<ConverterPara> &param) {
  CHECK_NULL_RETURN(param);
  int status = PrepareQuantize(func_graph, param);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "PrepareQuantize failed.";
    return status;
  }

  std::shared_ptr<mindspore::Model> origin;
  std::shared_ptr<lite::Model> origin_lite_model;
  if (param->commonQuantParam.is_debug) {  // Bak fp32 model for debug
    auto quant_type = param->commonQuantParam.quant_type;
    param->commonQuantParam.quant_type = quant::QUANT_NONE;
    origin = std::make_shared<mindspore::Model>();
    CHECK_NULL_RETURN(origin);
    size_t size = 0;
    auto ret = BuildModelByFuncGraph(origin, func_graph, param, &size);
    param->commonQuantParam.quant_type = quant_type;
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Build model failed";
      return RET_ERROR;
    }
    origin_lite_model = ParseLiteModel(func_graph, param);
    if (origin_lite_model == nullptr) {
      MS_LOG(ERROR) << "Parse lite model failed.";
      return RET_ERROR;
    }
  }
  if (param->commonQuantParam.quant_type == quant::QUANT_ALL) {  // Full Quantization
    status = ConvertFp16ToFp32(func_graph);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Converter fp16 to fp32 failed.";
      return status;
    }
    status = DoFullQuant(func_graph, param);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Do full quant failed.";
      return status;
    }
  } else if (param->commonQuantParam.quant_type == quant::QUANT_WEIGHT) {  // Weight Quantization
    status = DoWeightQuant(func_graph, param);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Do weight quant failed.";
      return status;
    }
  } else if (param->commonQuantParam.quant_type == quant::QUANT_DYNAMIC) {  // Dynamic Quantization
    status = DoDynamicQuant(func_graph, param);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Do dynamic quant failed.";
      return status;
    }
  }

  if (param->fullQuantParam.target_device != ASCEND) {
    auto optimizer = std::make_shared<opt::GraphOptimizer>();
    CHECK_NULL_RETURN(optimizer);
    auto fusion_pm = std::make_shared<opt::LitePassManager>("fusion pass manager after quant", false);
    CHECK_NULL_RETURN(fusion_pm);
    fusion_pm->AddPass(std::make_shared<opt::QuantDtypeCastFusion>());
    fusion_pm->AddPass(std::make_shared<opt::InferShapePass>(param->fmk_type, param->train_model));
    optimizer->AddPassManager(fusion_pm);
    if (optimizer->Optimize(func_graph) == nullptr) {
      MS_LOG(ERROR) << "run cast node fusion failed.";
      return RET_ERROR;
    }
  }

  if (param->commonQuantParam.is_debug) {
    status = DoQuantDebug(func_graph, param, origin, origin_lite_model);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Do quant debug failed.";
      return status;
    }
  }
  return RET_OK;
}

int QuantizationOptimizer::Run(const mindspore::FuncGraphPtr &func_graph) {
  if (param_->commonQuantParam.quant_type == quant::QUANT_NONE || param_->fullQuantParam.target_device == ASCEND) {
    return RET_OK;
  }
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
