/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/dynamic_shape/dynamic_shape.h"
#include <set>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <memory>
#include "base/base.h"

#include "abstract/symbolic_shape/symbol.h"
#include "abstract/symbolic_shape/int_symbol.h"
#include "abstract/symbolic_shape/symbol_info.h"
#include "frontend/jit/ps/action.h"
#include "include/utils/parallel_context.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "ir/anf.h"
#include "ir/graph_utils.h"
#include "include/utils/comm_manager.h"
#include "include/utils/anfalgo.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace parallel {
static SymbolElement GetSymbolInfo(const IntSymbol *int_s) {
  MS_EXCEPTION_IF_NULL(int_s);
  SymbolElement tmp;
  if (int_s->is_const()) {  // static shape element
    tmp.max = int_s->value();
    tmp.min = int_s->value();
    tmp.divisor = int_s->value();
    tmp.remainder = 0;
  } else {
    tmp.max = int_s->range_max();
    tmp.min = int_s->range_min();
    tmp.divisor = int_s->divisor();
    tmp.remainder = int_s->remainder();
  }
  return tmp;
}

Symbols StaticShapesToSymbols(const Shapes &shapes) {
  Symbols symbols;
  for (auto &shape : shapes) {
    Symbol symbol;
    for (auto &ele : shape) {
      if (ele <= 0) {
        MS_LOG(EXCEPTION) << "it is not static shape: " << ShapesToString(shapes);
      }
      SymbolElement symbol_ele;
      symbol_ele.divisor = ele;  // assign the divisor
      symbol.push_back(symbol_ele);
    }
    symbols.push_back(symbol);
  }
  return symbols;
}

bool IsDynamicShape(const Shape &shape) { return (std::count(shape.cbegin(), shape.cend(), -1) >= 1); }

bool IsDynamicShapes(const Shapes &shapes) {
  for (auto &shape : shapes) {
    if (std::count(shape.cbegin(), shape.cend(), -1) >= 1) {
      return True;
    }
  }
  return False;
}

bool IsDynamicShapesList(const std::vector<Shapes> &shapes_list) {
  return std::any_of(shapes_list.cbegin(), shapes_list.cend(),
                     [](const Shapes &shapes) { return IsDynamicShapes(shapes); });
}

void CheckRealDivisorSize(const Shapes &shapes, const Shapes &real_divisor_shapes) {
  if (shapes.size() != real_divisor_shapes.size()) {
    MS_LOG(EXCEPTION) << "the size of shapes is " << shapes.size() << ", but the size of real_divisor_shapes is "
                      << real_divisor_shapes.size() << ", they must be equal";
  }
  for (size_t i = 0; i < shapes.size(); ++i) {
    if (shapes[i].size() != real_divisor_shapes[i].size()) {
      MS_LOG(EXCEPTION) << "the size of shape is " << shapes[i].size() << ", but the size of real_divisor_shapes is "
                        << real_divisor_shapes[i].size() << ", they must be equal, the index is " << i;
    }
  }
}

// real divisor
// 1, For static shape elements, the real divisor of symbol is static shape.
// 2, For dynamic shape elements, if remainder != 0, the real divisor of symbol is the maximum common divisor of divisor
// and remainder, else equal to divisor
// 3, For static shape node, the symbols may be empty. For example, the shapes of make_tuple may be [[1], [1]], but
// the symbols of make_tuple may be [[]], using the static shape as the real divisor of symbol
Shapes GetRealDivisorSymbols(const Shapes &shapes, const Symbols &symbols) {
  // dynamic shape graph may be has static operator, and its symbol may be empty, use shapes in this case
  // static shape
  if (!IsDynamicShapes(shapes)) {
    return shapes;
  }

  if (shapes.size() != symbols.size()) {
    MS_LOG(EXCEPTION) << "the size of shapes is " << shapes.size() << ", but the size of symbols is " << symbols.size()
                      << ", they must be equal";
  }

  Shapes real_divisor_shapes;
  for (size_t i = 0; i < shapes.size(); ++i) {
    // dynamic shape graph may be has static operator, and its symbol may be empty, use shapes in this case
    // static shape
    if (!IsDynamicShape(shapes[i])) {
      real_divisor_shapes.push_back(shapes[i]);
      continue;
    }

    // dynamic shape
    if (shapes[i].size() != symbols[i].size()) {
      MS_LOG(EXCEPTION) << "the size of shape is " << shapes[i].size() << ", but the size of symbol is "
                        << symbols[i].size() << ", they must be equal, the index is " << i;
    }

    Shape real_divisor_shape;
    for (size_t j = 0; j < shapes[i].size(); ++j) {
      // static shape element, use shape
      if (shapes[i][j] > 0) {
        real_divisor_shape.push_back(shapes[i][j]);
        continue;
      }

      // dynamic shape element
      int64_t real_divisor = 1;
      if (symbols[i][j].remainder > 0) {
        real_divisor = std::gcd(symbols[i][j].divisor, symbols[i][j].remainder);
      } else {
        real_divisor = symbols[i][j].divisor;
      }
      real_divisor_shape.push_back(real_divisor);
    }

    real_divisor_shapes.push_back(real_divisor_shape);
  }

  CheckRealDivisorSize(shapes, real_divisor_shapes);

  return real_divisor_shapes;
}

static std::string DivisorOfSymbolToString(const Symbol &symbol) {
  std::string str = "[";
  for (size_t i = 0; i < symbol.size(); ++i) {
    str += std::to_string(symbol[i].divisor);
    if (i < symbol.size() - 1) {
      str += ", ";
    }
  }
  return str + "]";
}

static std::string RemainderOfSymbolToString(const Symbol &symbol) {
  std::string str = "[";
  for (size_t i = 0; i < symbol.size(); ++i) {
    str += std::to_string(symbol[i].remainder);
    if (i < symbol.size() - 1) {
      str += ", ";
    }
  }
  return str + "]";
}

std::string DivisorOfSymbolsToString(const Symbols &symbols) {
  std::string str = "[";
  for (size_t i = 0; i < symbols.size(); ++i) {
    str += DivisorOfSymbolToString(symbols[i]);
    if (i < symbols.size() - 1) {
      str += ", ";
    }
  }
  return str + "]";
}

std::string RemainderOfSymbolsToString(const Symbols &symbols) {
  std::string str = "[";
  for (size_t i = 0; i < symbols.size(); ++i) {
    str += RemainderOfSymbolToString(symbols[i]);
    if (i < symbols.size() - 1) {
      str += ", ";
    }
  }
  return str + "]";
}

void PrintSymbolInfo(const std::vector<symshape::SymbolInfoList> &symbol_infos) {
  for (size_t i = 0; i < symbol_infos.size(); ++i) {
    auto info_list = symbol_infos[i];
    for (size_t j = 0; j < info_list.size(); ++j) {
      MS_LOG(DEBUG) << "SYMBOL, i is " << i << ", j is " << j << ", divisor is " << info_list[j].divisor
                    << ", remainder is " << info_list[j].remainder;
    }
  }
}

bool ForwardHasDynamicShape(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  auto ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  auto all_nodes = TopoSort(ret, SuccDeeperSimple);
  auto graph_set = FindForwardGraphByRootNodes(all_nodes, root);
  if (graph_set.empty()) {
    MS_LOG(INFO) << "Can not find the forward graph, so find the ops in root graph";
    auto fgs = root->manager()->func_graphs();
    for (const auto &fg : fgs) {
      if (common::AnfAlgo::IsDynamicGraph(fg)) {
        return true;
      }
    }
  } else {
    for (const auto &fg : graph_set) {
      if (common::AnfAlgo::IsDynamicGraph(fg)) {
        return true;
      }
    }
  }
  return false;
}

bool IsParallelDynamicShape(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(parallel::ParallelContext::GetInstance());
  std::string parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode != parallel::kAutoParallel && parallel_mode != parallel::kSemiAutoParallel) {
    return false;
  }
  if (func_graph->has_flag(kSkipAutoParallelCompile)) {
    return false;
  }
  if (parallel::ParallelContext::GetInstance()->dynamic_shape_parallel_flag_is_set()) {
    return parallel::ParallelContext::GetInstance()->is_dynamic_shape_parallel();
  }

  bool is_dynamic = ForwardHasDynamicShape(func_graph);
  parallel::ParallelContext::GetInstance()->set_is_dynamic_shape_parallel(is_dynamic);
  return is_dynamic;
}

bool IsForwardDynamicShape() {
  MS_EXCEPTION_IF_NULL(parallel::ParallelContext::GetInstance());
  return parallel::ParallelContext::GetInstance()->is_dynamic_shape_parallel();
}

bool IsSemiOrAutoParallelMode() {
  MS_EXCEPTION_IF_NULL(parallel::ParallelContext::GetInstance());
  std::string parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();
  return (parallel_mode == parallel::kAutoParallel || parallel_mode == parallel::kSemiAutoParallel);
}

static int64_t GetDeviceNum() {
  int64_t device_num = 1;
  if (parallel::ParallelContext::GetInstance()->device_num_is_set()) {
    device_num = parallel::ParallelContext::GetInstance()->device_num();
  } else {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    std::string backend = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    std::string world_group;
    if (backend == kAscendDevice || backend == kDavinciDevice) {
      world_group = parallel::HCCL_WORLD_GROUP;
    } else if (backend == kGPUDevice) {
      world_group = parallel::NCCL_WORLD_GROUP;
    } else {
      MS_LOG(EXCEPTION) << "Invalid communication backend: " << backend
                        << " for semi_auto_parallel/auto_parallel mode,"
                           " currently only support Ascend/GPU backend.";
    }
    uint32_t world_rank_size = 0;
    if (!CommManager::GetInstance().GetRankSize(world_group, &world_rank_size)) {
      MS_LOG(EXCEPTION) << "Get rank size failed";
    }
    device_num = UintToInt(world_rank_size);
  }

  auto pipeline_stage = parallel::ParallelContext::GetInstance()->pipeline_stage_split_num();
  device_num = device_num / pipeline_stage;
  return device_num;
}

// modify symbol info by dataset strategy
// only for data sink is false
std::vector<symshape::SymbolInfoList> ParallelSymbolInfo(const std::vector<symshape::SymbolInfoList> &symbol_infos,
                                                         bool has_dyn_shape) {
  if (!has_dyn_shape || !IsSemiOrAutoParallelMode()) {  // static shape or sink mode no need to handle symbol info here
    return symbol_infos;
  }

  ParallelContext::GetInstance()->set_symbol_infos(symbol_infos);

  auto parallel_symbol_infos = symbol_infos;
  parallel::Strategies dataset_strategy;
  if (!ParallelContext::GetInstance()->dataset_strategy_tensormap().empty() &&
      !ParallelContext::GetInstance()->dataset_strategy_devmat().empty()) {
    auto all_tensor_map = ParallelContext::GetInstance()->dataset_strategy_tensormap();
    auto all_dev_mat = ParallelContext::GetInstance()->dataset_strategy_devmat();
    for (size_t i = 0; i < all_tensor_map.size(); ++i) {
      Shape local_stra;
      auto cur_tensor_map = all_tensor_map.at(i);
      auto cur_dev_mat = all_dev_mat.at(i);
      for (size_t j = 0; j < cur_tensor_map.size(); ++j) {
        size_t shard_size = 1;
        for (size_t k = 0; k < cur_tensor_map.at(j).size(); ++k) {
          auto val = cur_tensor_map.at(j).at(k);
          if (val != -1) {
            auto real_idx = cur_dev_mat.size() - LongToSize(val) - 1;
            shard_size *= LongToSize(cur_dev_mat.at(real_idx));
          }
        }
        local_stra.push_back(shard_size);
      }
      dataset_strategy.push_back(local_stra);
    }
  } else {
    if (!parallel::ParallelContext::GetInstance()->dataset_strategy().empty()) {
      dataset_strategy = parallel::ParallelContext::GetInstance()->dataset_strategy();
    } else {
      bool full_batch = parallel::ParallelContext::GetInstance()->full_batch();
      if (full_batch) {
        return parallel_symbol_infos;
      } else {
        // get device num
        int64_t device_num = GetDeviceNum();

        // set parallel symbol
        for (auto &symbol : parallel_symbol_infos) {
          if (!symbol.empty()) {
            symbol[0].divisor = symbol[0].divisor * device_num;
            symbol[0].remainder = symbol[0].remainder * device_num;
          }
        }
        return parallel_symbol_infos;
      }
    }
  }

  MS_LOG(DEBUG) << "dataset strategy is " << dataset_strategy;
  if (dataset_strategy.size() != parallel_symbol_infos.size()) {
    MS_LOG(EXCEPTION) << "The size of dataset strategy is " << dataset_strategy.size()
                      << ", but the size of symbol info is " << parallel_symbol_infos.size();
  }

  for (size_t i = 0; i < dataset_strategy.size(); ++i) {
    if (dataset_strategy[i].size() != parallel_symbol_infos[i].size()) {
      MS_LOG(EXCEPTION) << "Invalid dataset strategy size for index " << i << ", the size of dataset strategy ele is "
                        << dataset_strategy[i].size() << ", but the size of symbol info ele is "
                        << parallel_symbol_infos[i].size();
    }

    for (size_t j = 0; j < dataset_strategy[i].size(); ++j) {
      parallel_symbol_infos[i][j].divisor = parallel_symbol_infos[i][j].divisor * dataset_strategy[i][j];
      parallel_symbol_infos[i][j].remainder = parallel_symbol_infos[i][j].remainder * dataset_strategy[i][j];
    }
  }

  return parallel_symbol_infos;
}

Symbols GetNodeSymbol(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->abstract());
  Symbols symbols;
  MS_LOG(DEBUG) << ", node is " << node->ToString() << ",full name is " << node->fullname_with_scope();

  auto sym_shape = node->abstract()->GetSymbolicShape();
  if (sym_shape == nullptr) {
    // for static operator in dynamic shape graph, the symbol maybe null
    // construct symbol base on shape
    MS_EXCEPTION_IF_NULL(node->abstract()->GetShape());
    sym_shape = node->abstract()->GetShape()->BuildSymbolicShape();
  }

  if (sym_shape->symbols().empty()) {
    // dynamic operator, and the input is scalar
    symbols.push_back(Symbol{});
    return symbols;
  }

  Symbol int_symbol;
  Symbols list_symbol;
  bool int_symbol_flag = false;
  for (const auto &s : sym_shape->symbols()) {
    // There are two situations in sym_shape->symbols():
    // 1, It is a ListSymbol, its elements are IntSymbols: [IntSymbol, IntSymbol, ..., IntSymbol]
    // 2, It is a vector of ListSymbol, its elements are ListSymbols: [ListSymbols, ListSymbols, ..., ListSymbols]
    // The ListSymbol like this: [s46<[64,inf]|64N|=s1*64-64>, 768], all elements are IntSymbols, but 768 is const
    MS_EXCEPTION_IF_NULL(s);
    if (s->is<IntSymbol>()) {
      auto int_s = s->as<IntSymbol>();
      int_symbol.push_back(GetSymbolInfo(int_s));
      int_symbol_flag = true;
      continue;
    } else if (s->is<ListSymbol>()) {
      auto list_s = s->as<ListSymbol>();
      Symbol tmp;
      for (const auto &ele : list_s->symbols()) {
        if (ele->is<IntSymbol>()) {
          auto int_s = ele->as<IntSymbol>();
          tmp.push_back(GetSymbolInfo(int_s));
        }
      }
      list_symbol.push_back(tmp);
    } else {
      MS_LOG_WITH_NODE(EXCEPTION, node) << "invalid symbol for " << node->fullname_with_scope();
    }
  }

  if (int_symbol_flag) {
    symbols.push_back(int_symbol);
  } else {
    symbols = list_symbol;
  }

  MS_LOG(DEBUG) << "The symbol is " << DivisorOfSymbolsToString(symbols);
  return symbols;
}

void UpdateParamSymbolicShape(const FuncGraphPtr &root) {
  if (!IsForwardDynamicShape()) {
    return;
  }
  auto symbol_infos = ParallelContext::GetInstance()->symbol_infos();
  // when input is None, the parameter is removed from root graph.
  symbol_infos.erase(std::remove_if(symbol_infos.begin(), symbol_infos.end(),
                                    [](const symshape::SymbolInfoList &s) { return s.empty(); }),
                     symbol_infos.end());
  abstract::AbstractBasePtrList params_abs(root->parameters().size());
  (void)std::transform(root->parameters().begin(), root->parameters().end(), params_abs.begin(),
                       [](const AnfNodePtr &p) { return p->abstract(); });
  std::vector<ListSymbolPtr> original_symbolic_shapes;
  if (!symbol_infos.empty()) {
    original_symbolic_shapes = symshape::BuildSymbolicShapeBySymbolInfo(params_abs, symbol_infos);
  }
  for (size_t i = 0; i < params_abs.size(); i++) {
    if (params_abs[i] == nullptr) {
      continue;
    }
    if (i < original_symbolic_shapes.size()) {
      params_abs[i]->SetSymbolicShape(original_symbolic_shapes[i]);
    } else if (params_abs[i]->GetSymbolicShape() != nullptr) {
      params_abs[i]->SetSymbolicShape(nullptr);
    }
  }
  ParallelContext::GetInstance()->set_symbol_infos({});
}

static Status CheckLayoutAndDivisor(const std::vector<std::shared_ptr<TensorLayout>> &tensor_layouts,
                                    const Shapes &divisors) {
  if (tensor_layouts.size() != divisors.size()) {
    return FAILED;
  }

  for (size_t i = 0; i < divisors.size(); ++i) {
    Shape shard = tensor_layouts[i]->shard_strategy();
    Shape divisor = divisors[i];
    if (shard.size() != divisor.size()) {
      return FAILED;
    }
    for (size_t j = 0; j < divisor.size(); ++j) {
      if (divisor[j] % shard[j] != 0) {
        MS_LOG(ERROR) << "the symbol-divisor:" << ShapeToString(divisor)
                      << " can not be divisible by strategy: " << ShapeToString(shard) << ", the layout is "
                      << tensor_layouts[i]->ToString();
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

static Status CheckLayoutFormatForDynamicShape(const std::vector<std::shared_ptr<TensorLayout>> &layouts,
                                               const std::string &name) {
  for (auto &layout : layouts) {
    if (layout->IsInterleavedParallel()) {
      MS_LOG(ERROR) << "it does not support to config interleave parallel in layout for dynamic shape, the op name is "
                    << name;
      return FAILED;
    }
    Shapes tensor_map_before = layout->tensor_map_before();
    for (auto &map_ele : tensor_map_before) {
      if (map_ele.size() > 1) {
        MS_LOG(ERROR) << "it does not support to config multi-map in layout for dynamic shape, the op name is " << name;
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status CheckLayoutForDynamicShape(const std::vector<std::shared_ptr<TensorLayout>> &in_tensor_layouts,
                                  const std::vector<std::shared_ptr<TensorLayout>> &out_tensor_layouts,
                                  const OperatorInfoPtr &op_info) {
  MS_EXCEPTION_IF_NULL(op_info);
  if (!op_info->dynamic_shape_flag()) {
    return SUCCESS;
  }

  if (CheckLayoutFormatForDynamicShape(in_tensor_layouts, op_info->name()) != SUCCESS) {
    MS_LOG(ERROR) << "check input layouts format failed";
    return FAILED;
  }

  if (CheckLayoutFormatForDynamicShape(out_tensor_layouts, op_info->name()) != SUCCESS) {
    MS_LOG(ERROR) << "check output layouts format failed";
    return FAILED;
  }

  Shapes inputs_divisor = op_info->inputs_divisor();
  Shapes outputs_divisor = op_info->outputs_divisor();
  if (CheckLayoutAndDivisor(in_tensor_layouts, inputs_divisor) != SUCCESS) {
    MS_LOG(ERROR) << "input layout is invalid, the op name is " << op_info->name() << ", the inputs shape is "
                  << ShapesToString(op_info->inputs_shape());
    return FAILED;
  }

  // only check out_tensor_layouts if it is not empty
  if (!out_tensor_layouts.empty() && CheckLayoutAndDivisor(out_tensor_layouts, outputs_divisor) != SUCCESS) {
    MS_LOG(ERROR) << "output layout is invalid, the op name is " << op_info->name() << ", the outputs shape is "
                  << ShapesToString(op_info->outputs_shape());
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
