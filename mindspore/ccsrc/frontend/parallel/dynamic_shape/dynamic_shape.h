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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_DYNAMIC_SHAPE_DYNAMIC_SHAPE_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_DYNAMIC_SHAPE_DYNAMIC_SHAPE_H_

#include <set>
#include <utility>
#include <string>
#include <memory>
#include <vector>
#include "ir/value.h"
#include "ir/graph_utils.h"
#include "base/base.h"
#include "utils/hash_map.h"
#include "mindspore/core/symbolic_shape/symbol.h"
#include "mindspore/core/symbolic_shape/symbol_info.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "ops/array_ops.h"

namespace mindspore {
namespace parallel {
struct SymbolElement {
  int64_t max = 1;
  int64_t min = 1;
  int64_t divisor = 1;
  int64_t remainder = 0;
};

using Symbol = std::vector<SymbolElement>;
using Symbols = std::vector<Symbol>;
constexpr size_t INPUT_SYMBOLS_INDEX = 0;
constexpr size_t OUTPUT_SYMBOLS_INDEX = 1;
constexpr size_t INPUT_OUTPUT_SYMBOLS_SIZE = 2;

void PrintSymbolInfo(const std::vector<symshape::SymbolInfoList> &symbol_infos);
std::vector<symshape::SymbolInfoList> ParallelSymbolInfo(const std::vector<symshape::SymbolInfoList> &symbol_infos,
                                                         bool has_dyn_shape);
bool IsParallelDynamicShape(const FuncGraphPtr &func_graph);
Symbols GetNodeSymbol(const AnfNodePtr &node);
Symbols StaticShapesToSymbols(const Shapes &shapes);
std::string DivisorOfSymbolsToString(const Symbols &symbols);
std::string RemainderOfSymbolsToString(const Symbols &symbols);

// get real divisor symbols for input or output
Shapes GetRealDivisorSymbols(const Shapes &shapes, const Symbols &symbols);

void TagDynamicShapeFuncGraph(const FuncGraphPtr &root);
bool InDynamicGraph(const CNodePtr &node);
bool IsDynamicShapesList(const std::vector<Shapes> &shapes_list);
bool IsDynamicShapes(const Shapes &shapes);
bool IsDynamicShape(const Shape &shape);
bool IsSemiOrAutoParallelMode();
void UpdateParamSymbolicShape(const FuncGraphPtr &root);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_DYNAMIC_SHAPE_DYNAMIC_SHAPE_H_
