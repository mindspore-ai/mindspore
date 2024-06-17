/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/auto_parallel/stage_compute.h"

#include <algorithm>
#include <utility>
#include <string>
#include <vector>
#include <map>
#include <regex>

#include "mindspore/core/ops/array_ops.h"
#include "frontend/parallel/step_parallel.h"
#include "utils/parallel_node_check.h"
#include "ir/func_graph.h"
#include "include/common/utils/parallel_context.h"
#include "mindspore/core/utils/ms_utils.h"

namespace mindspore {
namespace parallel {

constexpr size_t PARSING_FAILED = SIZE_MAX;
constexpr size_t Kilo = 1024;
constexpr double MARGIN_FACTOR = 1.1;

// Thousand separators for memory numbers
string TSepBytes(size_t n) {
  string res = std::to_string(n / Kilo / Kilo);
  int thousand_digit_num = 3;
  for (int i = static_cast<int>(res.size()) - thousand_digit_num; i > 0; i -= thousand_digit_num) res.insert(i, ",");
  return res + "M";
}

std::vector<AnfNodePtr> GetNodes(const FuncGraphPtr &root) {
  AnfNodePtr ret_forward = root->get_return();
  return DeepScopedGraphSearch(ret_forward);
}

// Get Number of Layers ((each model has unique layers name to analyse))
size_t GetNumLayers(const FuncGraphPtr &root) {
  const std::string kHeadLayer = "Head";
  const std::string kNormLayer = "Norm";
  const std::string kEmbeddingLayer = "Embedding";
  const std::string kLinearLayer = "Linear";
  std::vector<FuncGraphPtr> pipeline_cells;
  size_t num_layers = 0;

  auto forward_nodes = GetNodes(root);
  for (auto node : forward_nodes) {
    if (!node->isa<CNode>()) continue;

    auto cnode = node->cast<CNodePtr>();
    if (!IsValueNode<FuncGraph>(cnode->input(0))) {
      continue;
    }

    auto graph = GetValueNode<FuncGraphPtr>(cnode->input(0));
    if (graph->stage() == -1 ||
        std::find(pipeline_cells.begin(), pipeline_cells.end(), graph) != pipeline_cells.end()) {
      continue;
    }

    pipeline_cells.push_back(graph);
    std::string name = graph->ToString();
    // Remove pre/post cells
    if (!(name.find(kHeadLayer) != std::string::npos || name.find(kNormLayer) != std::string::npos ||
          name.find(kEmbeddingLayer) != std::string::npos || name.find(kLinearLayer) != std::string::npos)) {
      MS_LOG(DEBUG) << name << " is counted as a layer";
      num_layers++;
    } else {
      MS_LOG(DEBUG) << name << " is NOT a normal layer";
    }
  }

  return (num_layers > 0) ? num_layers : PARSING_FAILED;
}

size_t GetNumDevices() { return g_device_manager->DeviceNum(); }

// Get parallel_optimizer
bool HasParallelOptimizer(const FuncGraphPtr &root) {
  return parallel::ParallelContext::GetInstance()->enable_parallel_optimizer();
}

// Check if recomputation was chosen. Currently only able to check select_recompute
bool HasRecompute(const FuncGraphPtr &root) {
  auto forward_nodes = GetNodes(root);
  for (auto &forward_node : forward_nodes) {
    if (!forward_node->isa<CNode>()) {
      continue;
    }
    auto cnode = forward_node->cast<CNodePtr>();
    if (IsValueNode<FuncGraph>(cnode->input(0))) {
      auto fg = GetValueNode<FuncGraphPtr>(cnode->input(0));
      if (fg->has_flag(FUNC_GRAPH_RECOMPUTE_GRAD_GRAPH) || fg->has_flag(FUNC_GRAPH_RECOMPUTE_K_GRAPH)) {
        MS_LOG(DEBUG) << "found recompute cell " << fg->ToString();
        return true;
      }
    }

    if (IsParallelCareNode(cnode) && cnode->has_user_data<OperatorInfo>()) {
      auto current_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
      if (current_prim != nullptr) {
        auto prim_recompute_attr = current_prim->GetAttr(kAttrRecompute);
        if (prim_recompute_attr != nullptr && prim_recompute_attr->isa<BoolImm>()) {
          auto recomputed = GetValue<bool>(prim_recompute_attr);
          if (recomputed) {
            MS_LOG(DEBUG) << "found recompute node " << current_prim->name();
            return true;
          }
        }
      }
    }
  }

  return false;
}

// Get DP and MP dimensions
std::tuple<size_t, size_t> GetDPAndMP(const std::shared_ptr<Graph> &graph, const size_t stage) {
  std::map<std::string, int> strategy_occurrence;

  size_t dp = 0;
  size_t mp = 0;
  const unsigned int kTargetLength = 2;
  size_t roll_back = FloatToSize(log2(stage));
  for (auto &node_ptr : graph->nodes) {
    if (node_ptr.apply.op_type == kRecMatMul) {
      size_t n_cut = node_ptr.apply.strs.size() - roll_back - 1;
      if (n_cut >= node_ptr.apply.strs.size()) {
        MS_LOG(WARNING) << "Strategy of  " << node_ptr.name << " not available";
        return {PARSING_FAILED, PARSING_FAILED};
      }
      StrategyRec strategy = node_ptr.apply.strs[n_cut];
      if (sizeof(strategy.inputTensor) / sizeof(TensorStr4D) >= kTargetLength) {
        MS_LOG(DEBUG) << "inputTensor[0] " << strategy.inputTensor[0].str_w << " " << strategy.inputTensor[0].str_h
                      << " " << strategy.inputTensor[0].str_c << " " << strategy.inputTensor[0].str_n;
        MS_LOG(DEBUG) << "inputTensor[1] " << strategy.inputTensor[1].str_w << " " << strategy.inputTensor[1].str_h
                      << " " << strategy.inputTensor[1].str_c << " " << strategy.inputTensor[1].str_n;
        int mp_strat = 1;
        int dp_strat = 1;
        if (strategy.inputTensor[1].str_h * strategy.inputTensor[1].str_w != 0) {
          mp_strat = static_cast<int>(1 / (strategy.inputTensor[1].str_h * strategy.inputTensor[1].str_w));
        }
        if (strategy.inputTensor[0].str_h != 0) {
          dp_strat = static_cast<int>(1 / strategy.inputTensor[0].str_h);
        }
        MS_LOG(DEBUG) << "dp_strat: " << dp_strat << ", mp_strat: " << mp_strat;
        std::string strategy_str = std::to_string(dp_strat) + "," + std::to_string(mp_strat);
        auto it = strategy_occurrence.find(strategy_str);
        if (it == strategy_occurrence.end()) {
          strategy_occurrence.insert(std::pair<std::string, int>(strategy_str, 1));
        } else {
          it->second++;
        }
      } else {
        MS_LOG(DEBUG) << "MatMul strategy found but null";
      }
    }
  }
  // Take the (DP,MP) that appears the most
  int occurrence = 0;
  for (auto it = strategy_occurrence.begin(); it != strategy_occurrence.end(); it++) {
    if (it->second > occurrence) {
      auto stra = it->first;
      auto pos = stra.find(",");
      dp = static_cast<size_t>(std::stoi(stra.substr(0, pos)));
      mp = static_cast<size_t>(std::stoi(stra.substr(pos + 1, stra.size() - pos)));
      occurrence = it->second;
    }
  }
  if (dp > 0 && mp > 0) {
    return {dp, mp};
  }
  return {PARSING_FAILED, PARSING_FAILED};
}

// Get Vocab Size and Hidden Size as a tuple
std::tuple<size_t, size_t> GetVocabAndHiddenSize(const FuncGraphPtr &root) {
  size_t hidden_size = 0;
  size_t vocab_size = 0;
  std::vector<AnfNodePtr> parameters = root->parameters();
  for (auto &p : parameters) {
    auto parameter_ptr = p->cast<ParameterPtr>();
    Shapes param_shapes = GetNodeShape(p);
    if (hidden_size == 0 && std::regex_match(parameter_ptr->name().c_str(), std::regex(".*0.attention.*.weight"))) {
      hidden_size = static_cast<size_t>(param_shapes[0][1]);
      MS_LOG(DEBUG) << "Parameter for hidden size: " << parameter_ptr->name().c_str() << "; with shape " << param_shapes
                    << "; h = " << hidden_size;
    } else if (vocab_size == 0 && (std::regex_match(parameter_ptr->name().c_str(), std::regex(".*word_embedding.*")) ||
                                   std::regex_match(parameter_ptr->name().c_str(), std::regex(".*tok_embeddings.*")))) {
      vocab_size = static_cast<size_t>(param_shapes[0][0]);
      MS_LOG(DEBUG) << "Parameter for vocab size: " << parameter_ptr->name().c_str() << "; with shape " << param_shapes
                    << "; v = " << vocab_size;
    } else {
      MS_LOG(DEBUG) << "Parameter " << parameter_ptr->name().c_str() << "; with shape " << param_shapes;
    }
    if (hidden_size > 0 && vocab_size > 0) break;
  }
  if (hidden_size > 0 && vocab_size > 0) {
    return {hidden_size, vocab_size};
  }
  return {PARSING_FAILED, PARSING_FAILED};
}

// Get Attention Heads and Sequence Length
std::tuple<size_t, size_t> GetSeqLengthAndAttentionHeads(const FuncGraphPtr &root) {
  size_t seq_length = 0;
  size_t attention_heads = 0;
  auto forward_nodes = GetNodes(root);
  const size_t kTargetShape = 4;
  for (auto &forward_node : forward_nodes) {
    if (forward_node->isa<CNode>()) {
      auto cnode = forward_node->cast<CNodePtr>();
      if (IsParallelCareNode(cnode) && cnode->has_user_data<OperatorInfo>()) {
        auto current_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
        if (seq_length == 0 && strcmp(current_prim->name().c_str(), GET_NEXT) == 0) {
          Shapes param_shapes = GetNodeShape(cnode);
          MS_LOG(DEBUG) << current_prim->name().c_str() << " with shape " << param_shapes;
          seq_length = static_cast<size_t>(param_shapes[0][1] - 1);
        }
        if (attention_heads == 0 && (strcmp(current_prim->name().c_str(), SOFTMAX) == 0 ||
                                     strcmp(current_prim->name().c_str(), FLASH_ATTENTION_SCORE) == 0)) {
          Shapes param_shapes = GetNodeShape(cnode);
          if (param_shapes[0].size() == kTargetShape) {
            MS_LOG(DEBUG) << current_prim->name().c_str() << " with shape " << param_shapes;
            attention_heads = static_cast<size_t>(param_shapes[0][1]);
          }
        }
        if (attention_heads > 0 && seq_length > 0) {
          break;
        }
      }
    }
  }
  if (seq_length > 0 && attention_heads > 0) {
    return {seq_length, attention_heads};
  }
  return {PARSING_FAILED, PARSING_FAILED};
}

// Get num micro
size_t GetNumMicro(const FuncGraphPtr &root) {
  auto manager = root->manager();
  AnfNodePtr virtual_dataset;
  for (auto &fg : manager->func_graphs()) {
    for (auto &node : fg->nodes()) {
      if (IsPrimitiveCNode(node, prim::kPrimVirtualDataset)) {
        virtual_dataset = node;
        break;
      }
    }
  }
  auto node_user_map = manager->node_users();
  auto node_users = node_user_map[virtual_dataset];
  for (auto &node_user : node_users) {
    if (IsPrimitiveCNode(node_user.first, prim::kPrimTupleGetItem)) {
      auto data_users = manager->node_users()[node_user.first];
      auto node_first = data_users.front().first;
      if (!IsPrimitiveCNode(node_first, prim::kPrimStridedSlice)) {
        data_users = node_user_map[node_first];
      }
      MS_LOG(DEBUG) << "micro batch size found: " << int64_t(data_users.size());
      return int64_t(data_users.size());
    }
  }
  MS_LOG(WARNING) << "micro batch size not found";
  return PARSING_FAILED;
}

// Get per batch
size_t GetPerBatch(const FuncGraphPtr &root, size_t seq_l) {
  size_t per_batch = 0;
  auto forward_nodes = GetNodes(root);
  for (auto &forward_node : forward_nodes) {
    if (forward_node->isa<CNode>()) {
      auto cnode = forward_node->cast<CNodePtr>();
      if (IsParallelCareNode(cnode) && cnode->has_user_data<OperatorInfo>()) {
        auto current_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
        if (per_batch == 0 && strcmp(current_prim->name().c_str(), MATMUL) == 0) {
          Shapes param_shapes = GetNodeShape(cnode);
          MS_LOG(DEBUG) << current_prim->name().c_str() << " with shape " << param_shapes;
          per_batch = static_cast<size_t>(param_shapes[0][0]) / seq_l;
          MS_LOG(DEBUG) << "batch size found: " << per_batch;
        }
        if (per_batch > 0) {
          break;
        }
      }
    }
  }
  return (per_batch > 0) ? per_batch : PARSING_FAILED;
}

std::tuple<size_t, size_t, size_t, size_t> GetFPFromParams(const FuncGraphPtr &root) {
  size_t fp_params = 0;
  size_t fp_optim = 0;
  size_t fp_grads = 0;
  size_t fp_norm = 0;

  std::vector<AnfNodePtr> parameters = root->parameters();
  for (auto &p : parameters) {
    if (p == nullptr) {
      continue;
    }
    auto parameter_ptr = p->cast<ParameterPtr>();
    mindspore::TypePtr element_type;
    auto data_type = p->Type();
    MS_EXCEPTION_IF_NULL(data_type);
    if (!data_type->isa<mindspore::TensorType>()) {
      continue;
    }
    element_type = data_type->cast<mindspore::TensorTypePtr>()->element();
    MS_EXCEPTION_IF_NULL(element_type);
    auto type_id = element_type->type_id();
    if (fp_grads == 0 && std::regex_match(parameter_ptr->name().c_str(), std::regex("accu_grads.*embedding.*"))) {
      fp_grads = GetTypeByte(TypeIdToType(type_id));
    } else if (fp_optim == 0 && std::regex_match(parameter_ptr->name().c_str(), std::regex("adam_m.*embedding.*"))) {
      fp_optim = GetTypeByte(TypeIdToType(type_id));
    } else if (fp_params == 0 &&
               (std::regex_match(parameter_ptr->name().c_str(), std::regex("^model.*embedding.*")) ||
                std::regex_match(parameter_ptr->name().c_str(), std::regex("^backbone.*embedding.*")))) {
      fp_params = GetTypeByte(TypeIdToType(type_id));
    } else if (fp_norm == 0 &&
               (std::regex_match(parameter_ptr->name().c_str(), std::regex("^model.*attention_norm.*")) ||
                std::regex_match(parameter_ptr->name().c_str(), std::regex("^backbone.*layernorm.*")))) {
      fp_norm = GetTypeByte(TypeIdToType(type_id));
    }
    if (fp_optim > 0 && fp_params > 0 && fp_grads > 0 && fp_norm > 0) {
      return {fp_optim, fp_params, fp_grads, fp_norm};
    }
  }

  return {fp_params, fp_optim, fp_grads, fp_norm};
}

std::tuple<size_t, size_t> GetFPFromNodes(const FuncGraphPtr &root) {
  size_t fp_dropout = 0;
  size_t fp_softmax = 0;

  auto forward_nodes = GetNodes(root);
  for (auto &forward_node : forward_nodes) {
    if (forward_node->isa<CNode>()) {
      auto cnode = forward_node->cast<CNodePtr>();
      if (IsParallelCareNode(cnode) && cnode->has_user_data<OperatorInfo>()) {
        auto current_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
        mindspore::TypePtr element_type;
        auto data_type = cnode->Type();
        MS_EXCEPTION_IF_NULL(data_type);
        if (!data_type->isa<mindspore::TensorType>()) {
          if (std::regex_match(current_prim->name().c_str(), std::regex("Dropout"))) {
            fp_dropout = 1;
          }
          continue;
        }
        element_type = data_type->cast<mindspore::TensorTypePtr>()->element();
        MS_EXCEPTION_IF_NULL(element_type);
        auto type_id = element_type->type_id();
        if (fp_softmax == 0 && (std::regex_match(current_prim->name().c_str(), std::regex(SOFTMAX)))) {
          fp_softmax = GetTypeByte(TypeIdToType(type_id));
        } else if (fp_dropout == 0 && std::regex_match(current_prim->name().c_str(), std::regex(DROPOUT))) {
          fp_dropout = GetTypeByte(TypeIdToType(type_id));
        }
        if (fp_softmax > 0 && fp_dropout > 0) {
          return {fp_dropout, fp_softmax};
        }
      }
    }
  }

  return {fp_dropout, fp_softmax};
}

// Get FP format for params, optimizer, gradient, norm, softmax, dropout
std::tuple<size_t, size_t, size_t, size_t, size_t, size_t> GetFP_formats(const FuncGraphPtr &root) {
  size_t fp_params = 0;
  size_t fp_optim = 0;
  size_t fp_grads = 0;
  size_t fp_norm = 0;
  size_t fp_dropout = 0;
  size_t fp_softmax = 0;

  std::tie(fp_params, fp_optim, fp_grads, fp_norm) = GetFPFromParams(root);
  std::tie(fp_dropout, fp_softmax) = GetFPFromNodes(root);

  if (fp_params == 0 || fp_optim == 0 || fp_grads == 0 || fp_norm == 0) {
    return {PARSING_FAILED, PARSING_FAILED, PARSING_FAILED, PARSING_FAILED, PARSING_FAILED, PARSING_FAILED};
  }

  return {fp_params, fp_optim, fp_grads, fp_norm, fp_dropout, fp_softmax};
}

size_t GetExpansionRatio(const FuncGraphPtr &root) {
  std::vector<AnfNodePtr> parameters = root->parameters();
  for (auto &p : parameters) {
    auto parameter_ptr = p->cast<ParameterPtr>();
    Shapes param_shapes = GetNodeShape(p);
    if (std::regex_match(parameter_ptr->name().c_str(), std::regex(".*0.feed_forward.*.weight")) ||
        std::regex_match(parameter_ptr->name().c_str(), std::regex(".*0.output.projection.weight"))) {
      return param_shapes[0][0];
    }
  }
  return PARSING_FAILED;
}

// Get FP format for multi head attention block, norms, feed forward block
std::tuple<size_t, size_t, size_t> GetNumTransformerComponents(const FuncGraphPtr &root) {
  size_t n_mha = 0;
  size_t n_ff = 0;
  size_t n_norm = 0;

  std::vector<AnfNodePtr> parameters = root->parameters();
  for (auto &p : parameters) {
    if (p == nullptr) {
      continue;
    }
    auto parameter_ptr = p->cast<ParameterPtr>();
    if (std::regex_match(parameter_ptr->name().c_str(), std::regex("^model.layers.0.attention.wo.weight")) ||
        std::regex_match(parameter_ptr->name().c_str(), std::regex("^backbone.blocks.0.attention.dense1.weight"))) {
      n_mha++;
    } else if (std::regex_match(parameter_ptr->name().c_str(), std::regex("^model.layers.0.feed_forward.w1.weight")) ||
               std::regex_match(parameter_ptr->name().c_str(),
                                std::regex("^backbone.blocks.0.output.mapping.weight"))) {
      n_ff++;
    } else if (std::regex_match(parameter_ptr->name().c_str(), std::regex("^model.layers.0.attention_norm.weight")) ||
               std::regex_match(parameter_ptr->name().c_str(), std::regex("^model.layers.0.ffn_norm.weight")) ||
               std::regex_match(parameter_ptr->name().c_str(), std::regex("^backbone.blocks.0.layernorm.*gamma"))) {
      n_norm++;
    }
  }
  if (n_mha > 0 && n_ff > 0 && n_norm > 0) {
    return {n_mha, n_ff, n_norm};
  }
  return {PARSING_FAILED, PARSING_FAILED, PARSING_FAILED};
}

// Count weights matrixes in MHA and FF block
std::tuple<size_t, size_t> GetNumWeightsTransformer(const FuncGraphPtr &root) {
  size_t n_weight_MHA = 0;
  size_t n_weight_FF = 0;

  std::vector<AnfNodePtr> parameters = root->parameters();
  for (auto &p : parameters) {
    if (p == nullptr) {
      continue;
    }
    auto parameter_ptr = p->cast<ParameterPtr>();
    if (std::regex_match(parameter_ptr->name().c_str(), std::regex("^model.layers.0.attention.w.*weight")) ||
        std::regex_match(parameter_ptr->name().c_str(), std::regex("^backbone.blocks.0.attention.*weight"))) {
      n_weight_MHA++;
    } else if (std::regex_match(parameter_ptr->name().c_str(), std::regex("^model.layers.0.feed_forward.*.weight")) ||
               std::regex_match(parameter_ptr->name().c_str(), std::regex("^backbone.blocks.0.output.*weight"))) {
      n_weight_FF++;
    }
  }
  if (n_weight_MHA > 0 && n_weight_FF > 0) {
    return {n_weight_MHA, n_weight_FF};
  }
  return {PARSING_FAILED, PARSING_FAILED};
}

StageComputing::StageComputing(const FuncGraphPtr &r, const std::shared_ptr<Graph> &g, size_t device_num,
                               size_t device_capacity, size_t hidden_size, size_t vocab_size, size_t seq_length,
                               size_t head_num, size_t layer_num, size_t expansion_ratio, size_t dp, size_t mp,
                               size_t pp, size_t per_batch, size_t micro, bool parallel_opt, bool recompute)
    : root_(r),
      graph_(g),
      num_devices_(device_num),
      device_capacity_(device_capacity),
      vocab_size_(vocab_size),
      seq_length_(seq_length),
      hidden_size_(hidden_size),
      attention_heads_(head_num),
      num_layers_(layer_num),
      expansion_ratio_(expansion_ratio),
      parallel_opt_(parallel_opt),
      recompute_(recompute),
      dp_dim_(dp),
      mp_dim_(mp),
      pp_dim_(pp),
      per_batch_(per_batch),
      num_micros_(micro) {}

void StageComputing::SaveConfig() {
  saved_config_ = std::make_tuple(dp_dim_, mp_dim_, pp_dim_, per_batch_, num_micros_);
}

void StageComputing::LoadConfig() { std::tie(dp_dim_, mp_dim_, pp_dim_, per_batch_, num_micros_) = saved_config_; }

// Generalization of num parameters for transformer-based, relying on parsing
size_t StageComputing::NumParametersParsing(size_t l) {
  size_t n_weight_MHA;
  size_t n_weight_FF;
  std::tie(n_weight_MHA, n_weight_FF) = GetNumWeightsTransformer(root_);
  size_t n_MHA;
  size_t n_FF;
  size_t n_norm;
  std::tie(n_MHA, n_FF, n_norm) = GetNumTransformerComponents(root_);

  if (n_weight_MHA == PARSING_FAILED || n_MHA == PARSING_FAILED) {
    return PARSING_FAILED;
  }
  const size_t P_MHA = n_weight_MHA * (hidden_size_ * hidden_size_ + hidden_size_);
  const size_t P_FF = n_weight_FF * (expansion_ratio_ * hidden_size_) + expansion_ratio_ + hidden_size_;
  const size_t P_norm = 2 * hidden_size_;
  const size_t P_linear = hidden_size_ * vocab_size_ + vocab_size_;
  const size_t P_embedding = hidden_size_ * vocab_size_;

  return l * (n_MHA * P_MHA + n_norm * P_norm + n_FF * P_FF) + P_linear + P_embedding;
}

// Generalization of static memory for transformer-based, relying on parsing
size_t StageComputing::GetStaticMemoryParsing(size_t d, size_t t, size_t p, size_t P) {
  size_t FP_params;
  size_t FP_optimizer;
  size_t FP_gradient;
  std::tie(FP_params, FP_optimizer, FP_gradient, std::ignore, std::ignore, std::ignore) = GetFP_formats(root_);
  if (FP_params == PARSING_FAILED) {
    return PARSING_FAILED;
  }
  size_t model_params_size = (FP_params * P) / ((p == 1) ? (d * t) : t);
  size_t accu_gradients_size = (FP_gradient * P) / ((p == 1 && parallel_opt_) ? (d * t) : t);
  size_t optim_states_size = (2 * FP_optimizer * P) / ((parallel_opt_) ? (d * t) : t);
  MS_LOG(DEBUG) << "model_params_size: " << TSepBytes(static_cast<size_t>(model_params_size));
  MS_LOG(DEBUG) << "accu_gradients_size: " << TSepBytes(static_cast<size_t>(accu_gradients_size));
  MS_LOG(DEBUG) << "optim_states_size: " << TSepBytes(static_cast<size_t>(optim_states_size));
  return (model_params_size + accu_gradients_size + optim_states_size);
}

// Generalization of dynamic memory for transformer-based, relying on parsing
// Assuming seq parallelism for dropout and norms
// Assuming full recomputation
size_t StageComputing::GetDynamicMemoryParsing(size_t l, size_t b, size_t m, size_t p, size_t t) {
  size_t n_weight_MHA;
  size_t n_weight_FF;
  std::tie(n_weight_MHA, n_weight_FF) = GetNumWeightsTransformer(root_);
  size_t FP_params;
  size_t FP_optimizer;
  size_t FP_gradient;
  size_t FP_norm;
  size_t FP_dropout;
  size_t FP_softmax;
  size_t n_MHA;
  size_t n_FF;
  size_t n_norm;
  std::tie(FP_params, FP_optimizer, FP_gradient, FP_norm, FP_dropout, FP_softmax) = GetFP_formats(root_);
  FP_softmax = 0;
  std::tie(n_MHA, n_FF, n_norm) = GetNumTransformerComponents(root_);
  MS_LOG(DEBUG) << "FP_params: " << FP_params << ", FP_optimizer: " << FP_optimizer << ", FP_gradient: " << FP_gradient;
  MS_LOG(DEBUG) << "FP_dropout: " << FP_dropout << ", FP_softmax: " << FP_softmax << ", FP_norm: " << FP_norm;
  MS_LOG(DEBUG) << "n_MHA: " << n_MHA << ", n_FF: " << n_FF << ", n_norm: " << n_norm;
  MS_LOG(DEBUG) << "n_weight_MHA: " << n_weight_MHA << ", n_weight_FF: " << n_weight_FF;
  float sbh = seq_length_ * b * hidden_size_;
  float A_norm = n_norm * (FP_norm * sbh) / t;
  float A_MHA =
    n_MHA * ((n_weight_MHA * FP_params * sbh + FP_softmax * seq_length_ * seq_length_ * b * attention_heads_ +
              FP_dropout * 2 * seq_length_ * seq_length_ * b * attention_heads_) /
               t +
             FP_dropout * sbh / t);
  float A_FF = n_FF * (static_cast<float>(n_weight_FF * FP_params * seq_length_ * b * expansion_ratio_) / t +
                       FP_dropout * sbh / t);
  float A_intermediate = A_norm + A_MHA + A_FF;
  float nodes = static_cast<float>(num_devices_) / 8;
  float A_input = ((p > 1 && nodes == 1) ? m : ceil(nodes / 4)) * sbh / t;
  float n_Checkpoints = num_layers_ / 1.5;
  float full_recompute_size = n_Checkpoints * A_input + (l / n_Checkpoints) * A_intermediate;

  float communications_size = static_cast<float>(8 * seq_length_ * b * hidden_size_ * (t - 1)) / t;
  MS_LOG(DEBUG) << "l: " << l;
  MS_LOG(DEBUG) << "n_Checkpoints: " << n_Checkpoints;
  MS_LOG(DEBUG) << "A_input: " << TSepBytes(static_cast<size_t>(A_input));
  MS_LOG(DEBUG) << "A_norm: " << TSepBytes(static_cast<size_t>(A_norm));
  MS_LOG(DEBUG) << "A_MHA: " << TSepBytes(static_cast<size_t>(A_MHA));
  MS_LOG(DEBUG) << "A_FF: " << TSepBytes(static_cast<size_t>(A_FF));
  MS_LOG(DEBUG) << "A_intermediate: " << TSepBytes(static_cast<size_t>(A_intermediate));
  MS_LOG(DEBUG) << "l*communication: " << TSepBytes(static_cast<size_t>(l * communications_size));
  MS_LOG(DEBUG) << "l/n_Checkpoints * A_intermediate: "
                << TSepBytes(static_cast<size_t>((l / n_Checkpoints) * A_intermediate));
  MS_LOG(DEBUG) << "n_Checkpoints * A_input: " << TSepBytes(static_cast<size_t>(n_Checkpoints * A_input));

  return static_cast<size_t>(full_recompute_size + l * communications_size);
}

// Manually compute global batch size
size_t StageComputing::GlobalBatchSize() { return per_batch_ * num_micros_; }

// Get layer per stage
size_t StageComputing::GetLayerPerStage() {
  return ceil(static_cast<float>(num_layers_) / static_cast<float>(pp_dim_));
}

size_t StageComputing::GetMemory() {
  size_t P3 = NumParametersParsing(GetLayerPerStage());
  if (P3 == PARSING_FAILED) {
    return PARSING_FAILED;
  }
  size_t sMem3 = GetStaticMemoryParsing(dp_dim_, mp_dim_, pp_dim_, P3);
  if (sMem3 == PARSING_FAILED) {
    return PARSING_FAILED;
  }
  size_t dMem3 = GetDynamicMemoryParsing(GetLayerPerStage(), per_batch_, num_micros_, pp_dim_, mp_dim_);
  PrintResults(sMem3, dMem3, P3);
  return (sMem3 + dMem3);
}

// MS_LOG
void StageComputing::PrintHyperparams() {
  MS_LOG(INFO) << "Hyperparameters : h : " << hidden_size_ << ", s : " << seq_length_ << ", v : " << vocab_size_
               << ", a : " << attention_heads_ << ", L : " << num_layers_ << ", pb:" << per_batch_
               << ", B :" << GlobalBatchSize() << ", er: " << expansion_ratio_ << ", opt: " << parallel_opt_
               << ", rcpt: " << recompute_;
}

void Suggestion(const std::string &suggestion) {
  MS_LOG(INFO) << std::endl
               << "=================== Auto Parallel Config Suggestion by SAPP ===================" << std::endl
               << suggestion << std::endl
               << "===============================================================================";
}

std::string ParamSuggest(float mem_coeff, size_t stage, size_t batch, size_t micro) {
  std::stringstream ss;

  ss << " mem_coeff: " << mem_coeff;
  ss << ", pipeline_stage: " << stage << std::endl;
  ss << " batch_size: " << batch;
  ss << ", micro_batch_num: " << micro;

  return ss.str();
}

void StageComputing::FittingSuggestion() {
  std::stringstream ss;

  ss << " SAPP algorithm suggests the following parallel configuration:" << std::endl;
  ss << ParamSuggest(CostModelContext::GetInstance()->rp_matmul_mem_coef(), pp_dim_, per_batch_, num_micros_);

  Suggestion(ss.str());
}

void StageComputing::OOMSuggestion() {
  std::stringstream ss;

  float default_coeff = 1024;
  ss << " The current configuration seem to not fit in memory." << std::endl;
  ss << " SAPP algorithm suggests to change configuration to:" << std::endl;
  ss << ParamSuggest(default_coeff, pp_dim_, 1, num_micros_ * per_batch_);

  Suggestion(ss.str());
}

void StageComputing::ParsingException() {
  MS_LOG(WARNING) << "Something went wrong during the graph parsing process.";
  MS_LOG(WARNING) << "SAPP algorithm uses original stage number";
}

void StageComputing::PrintResults(size_t StaticMEM, size_t DynamicMEM, size_t num_param) {
  MS_LOG(INFO) << "DP: " << dp_dim_ << ", MP: " << mp_dim_ << ", Stages: " << pp_dim_ << ", n_micros: " << num_micros_
               << ", Per_batch: " << per_batch_ << ", Global batch size: " << GlobalBatchSize()
               << ", Num Parameters: " << num_param;
  MS_LOG(INFO) << "StaticMEM: " << TSepBytes(StaticMEM) << ", DynamicMEM: " << TSepBytes(DynamicMEM)
               << ", totalMEM: " << TSepBytes(StaticMEM + DynamicMEM);
}

size_t StageComputing::CurrentEstimation() { return GetMemory(); }

bool StageComputing::fits(size_t memory) { return ((MARGIN_FACTOR * memory) < device_capacity_); }

// Suggest a parallel config (dp,mp,pp) + batch (micro, per batch, dp)
// maintain global batch size
// memory function as cost model
Status StageComputing::FindSmallerStage() {
  if (pp_dim_ == 1) {
    return FAILED;
  }
  SaveConfig();

  bool saved = false;
  size_t factor = 2;
  double ratio = static_cast<float>(dp_dim_) / static_cast<float>(mp_dim_);
  for (; pp_dim_ >= 1; pp_dim_ /= factor) {
    if (fits(GetMemory()) && !saved) {
      SaveConfig();
      saved = true;
      MS_LOG(INFO) << "Stage " << pp_dim_ << " is selected";
    }
    float ratio_factor_dp = abs(static_cast<float>(factor * dp_dim_) / mp_dim_ - ratio);
    float ratio_factor_mp = abs(static_cast<float>(dp_dim_) / (factor * mp_dim_) - ratio);
    if (mp_dim_ == 1)
      dp_dim_ *= factor;
    else if (dp_dim_ == 1)
      mp_dim_ *= factor;
    else if (ratio_factor_dp <= ratio_factor_mp)
      dp_dim_ *= factor;
    else
      mp_dim_ *= factor;
  }

  LoadConfig();

  if (!saved) {
    return FAILED;
  }
  return SUCCESS;
}

// Estimation compute
size_t StageComputing::LaunchStageCompute() {
  size_t pp = pp_dim_;
  if (vocab_size_ == PARSING_FAILED || seq_length_ == PARSING_FAILED || expansion_ratio_ == PARSING_FAILED ||
      num_layers_ == PARSING_FAILED || dp_dim_ == PARSING_FAILED || num_micros_ == PARSING_FAILED ||
      per_batch_ == PARSING_FAILED) {
    ParsingException();
    return pp;
  }

  MS_LOG(INFO) << "Current stage number : " << pp;
  MS_LOG(DEBUG) << "Number of devices: " << num_devices_;

  PrintHyperparams();
  if (CurrentEstimation() == PARSING_FAILED) {
    ParsingException();
    return pp;
  }
  if (FindSmallerStage() == FAILED) {
    OOMSuggestion();
    return pp;
  } else {
    FittingSuggestion();
  }

  return pp_dim_;
}

// Suggest a pipeline stage
size_t ParallelSuggestion(const FuncGraphPtr &root, const std::shared_ptr<Graph> &graph) {
  size_t vocab;
  size_t seq;
  size_t heads;
  size_t dp;
  size_t mp;
  size_t pp;
  size_t hidden;
  size_t layers;
  size_t devices;
  size_t capacity;
  size_t micros;
  size_t per_batch;
  size_t er;
  bool opt;
  bool recompute;

  pp = static_cast<size_t>(parallel::ParallelContext::GetInstance()->pipeline_stage_split_num());
  if (root == nullptr || graph == nullptr) {
    MS_LOG(WARNING) << "Null costgraph or recgraph, ParallelSuggestion cannot run.";
    MS_LOG(WARNING) << "SAPP algorithm uses original stage number.";
    return pp;
  }

  std::tie(seq, heads) = GetSeqLengthAndAttentionHeads(root);
  std::tie(dp, mp) = GetDPAndMP(graph, pp);
  std::tie(hidden, vocab) = GetVocabAndHiddenSize(root);
  er = GetExpansionRatio(root);
  layers = GetNumLayers(root);
  capacity = GetDeviceCapacity();
  micros = GetNumMicro(root);

  per_batch = GetPerBatch(root, seq);
  devices = GetNumDevices();
  opt = HasParallelOptimizer(root);
  recompute = HasRecompute(root);

  StageComputing sc(root, graph, devices, capacity, hidden, vocab, seq, heads, layers, er, dp, mp, pp, per_batch,
                    micros, opt, recompute);
  pp = sc.LaunchStageCompute();
  return pp;
}

bool IsGraphFilter(const AnfNodePtr &node) { return !IsValueNode<FuncGraph>(node); }

// Update old stage number with suggestion
void ChangeStageNumber(const FuncGraphPtr &root, size_t new_stage_num) {
  size_t old_stage = static_cast<size_t>(parallel::ParallelContext::GetInstance()->pipeline_stage_split_num());
  if (old_stage == new_stage_num) {
    MS_LOG(INFO) << "Stage number " << new_stage_num << " is the same as the old value. Nothing changed.";
    return;
  }

  if (old_stage % new_stage_num != 0) {
    MS_LOG(WARNING) << "Stage number " << new_stage_num << " is not a divisor of the previous stage number "
                    << old_stage << ". Stage Number is NOT changed.";
    return;
  }

  size_t change_factor = old_stage / new_stage_num;
  MS_LOG(DEBUG) << "Old stage number:" << old_stage << " ; Change factor:" << change_factor;

  FuncGraphPtr main_graph;
  // Get main graph
  auto manager = root->manager();
  if (!root->has_flag(kTraining)) {
    main_graph = root;
  } else {
    for (auto &fg : manager->func_graphs()) {
      for (auto &node : fg->nodes()) {
        if (IsPrimitiveCNode(node, prim::kPrimVirtualDataset)) {
          main_graph = fg;
          break;
        }
      }
    }
  }

  // Get all sub graphs
  auto nodes = DeepScopedGraphSearchWithFilter(main_graph->get_return(), AlwaysInclude, IsGraphFilter);
  std::reverse(nodes.begin(), nodes.end());
  std::vector<FuncGraphPtr> subgraphs;
  for (auto &node : nodes) {
    auto graph = GetValueNode<FuncGraphPtr>(node);
    subgraphs.push_back(graph);
  }

  // Update stage in all sub_graphs
  for (auto &graph : subgraphs) {
    int graph_old_stage = graph->stage();
    if (graph_old_stage != -1) {
      graph->set_stage(graph_old_stage / change_factor);  // Either increase or decrease
    }
  }

  // Update stage in parallel context
  parallel::ParallelContext::GetInstance()->set_pipeline_stage_split_num(old_stage / change_factor);
  MS_LOG(INFO) << "END ChangeStageNumber"
               << ", new stage number: " << (old_stage / change_factor);
}

}  // namespace parallel
}  // namespace mindspore
