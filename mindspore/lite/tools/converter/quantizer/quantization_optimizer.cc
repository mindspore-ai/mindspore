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

namespace mindspore::lite::quant {
void GetFuncGraphs(const FuncGraphPtr &func_graph, std::set<FuncGraphPtr> *all_func_graphs) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(all_func_graphs != nullptr);
  all_func_graphs->insert(func_graph);
  auto nodes = func_graph->GetOrderedCnodes();
  std::deque<CNodePtr> to_process{};
  to_process.insert(to_process.end(), nodes.begin(), nodes.end());
  while (!to_process.empty()) {
    auto &cur_cnode = to_process.front();
    for (auto &input : cur_cnode->inputs()) {
      if (!IsValueNode<FuncGraph>(input)) {
        continue;
      }
      auto new_fg = GetValueNode<FuncGraphPtr>(input);
      if (all_func_graphs->find(new_fg) != all_func_graphs->end()) {
        continue;
      }
      all_func_graphs->insert(new_fg);
      auto new_nodes = new_fg->GetOrderedCnodes();
      to_process.insert(to_process.end(), new_nodes.begin(), new_nodes.end());
    }
    to_process.pop_front();
  }
}

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
    auto quantizer = std::make_unique<WeightQuantizer>(param);
    if (quantizer == nullptr) {
      MS_LOG(ERROR) << "New WeightQuantizer failed";
      return RET_ERROR;
    }
    status = static_cast<WeightQuantizer *>(quantizer.get())->DoQuantize(old_graph, init_scale);
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
  int size = builder.GetSize();
  auto content = builder.GetBufferPointer();
  if (content == nullptr) {
    MS_LOG(ERROR) << "GetBufferPointer nullptr";
    return static_cast<LiteModel *>(nullptr);
  }
  return static_cast<LiteModel *>(LiteModel::Import((const char *)content, size));
}

int DoQuantDebug(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param,
                 const std::shared_ptr<mindspore::Model> &origin_model, mindspore::lite::LiteModel *origin_lite_model) {
  auto quant_model = std::make_shared<mindspore::Model>();
  CHECK_NULL_RETURN(quant_model);
  auto ret = BuildModelByFuncGraph(quant_model, old_graph, param);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Build model failed";
    return RET_ERROR;
  }
  std::map<std::string, OpParameter *> op_parameters;
  FetchOpParameterFromFuncGraph(old_graph, &op_parameters);
  DebugInfoManager manager;

  auto quant_lite_model = ParseLiteModel(old_graph, param);
  if (quant_lite_model == nullptr) {
    MS_LOG(ERROR) << "Parse lite model failed";
    return RET_ERROR;
  }
  auto status = manager.CompareOriginWithQuant(origin_model, quant_model, op_parameters, param, *origin_lite_model,
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
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Compare origin with quant failed.";
    free_buffer();
    return status;
  }
  free_buffer();
  return RET_OK;
}

int DoSingleGraphQuantize(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param) {
  CHECK_NULL_RETURN(param);
  if (param->commonQuantParam.quant_type == schema::QuantType_QUANT_NONE) {
    return RET_OK;
  }
  int status;

  bool per_layer =
    param->commonQuantParam.quant_type == schema::QuantType_QUANT_ALL && !param->fullQuantParam.per_channel;
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
    auto ret = BuildModelByFuncGraph(origin, old_graph, param);
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
  GetFuncGraphs(func_graph, &all_func_graphs);
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
