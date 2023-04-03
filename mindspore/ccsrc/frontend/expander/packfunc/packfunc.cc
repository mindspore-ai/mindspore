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
#include <unordered_map>
#include <vector>
#include <memory>
#include "ops/base_operator.h"
#include "pybind_api/ir/primitive_py.h"
#include "include/common/debug/anf_ir_dump.h"
#include "frontend/expander/packfunc/pack_expander.h"
#include "frontend/expander/packfunc/packfunc.h"
#include "utils/ms_context.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace expander {
#define REGISTER_PRIMITIVE_OP_CPP_INFER_IMPL(name, primitive, OP_INFER_ClASS, is_impl_infer_value) \
  const auto helper_op_infer_##name = abstract::RegisterStandardPrimitiveEvalHelper(               \
    abstract::GetPrimitiveInferMapPtr(), primitive, std::make_shared<OP_INFER_ClASS>(), is_impl_infer_value);

using PackGraphMap = std::unordered_map<abstract::AbstractBasePtrList, FuncGraphPtr,
                                        abstract::AbstractBasePtrListHasher, abstract::AbstractBasePtrListEqual>;

static std::unordered_map<int64_t, PackGraphMap> pack_graph_cache;

FuncGraphPtr ExpandPackFunc(const PrimitivePtr &prim, const abstract::AbstractBasePtrList &abs_list) {
  auto key = GetValue<std::int64_t>(prim->GetAttr("unique_key"));
  auto &graph_map = pack_graph_cache[key];
  auto it = graph_map.find(abs_list);
  if (it != graph_map.end()) {
    return it->second;
  }
  auto prim_py = prim->cast_ptr<PrimitivePy>();
  MS_EXCEPTION_IF_NULL(prim_py);
  auto expander = expander::PackExpander::Instance();
  FuncGraphPtr graph;
  {
    py::gil_scoped_acquire acquire;
    py::object expand_func = prim_py->GetPyObj().attr("__expand__");
    py::object inputs = expander->BeginGraph(abs_list);
    py::object output = expand_func(inputs);
    graph = expander->EndGraph(output);
    graph_map[abs_list] = graph;
  }
  static bool dump_result = (common::GetEnv("MS_DEV_DUMP_PACK") == "on");
  if (dump_result) {
    DumpIR("pack_func_" + std::to_string(key) + ".ir", graph, true);
  }
  return graph;
}

class PackFuncInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    auto abs = InferShapeAndType(nullptr, primitive, input_args);
    return abs->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto abs = InferShapeAndType(nullptr, primitive, input_args);
    return abs->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    auto graph = ExpandPackFunc(primitive, input_args);
    MS_EXCEPTION_IF_NULL(graph);
    // the python primitive object may be used in different places with different inputs, so we
    // cannot save the graph in graph mode. But for pynative mode, this primitive is inferred
    // in forward thread sequentially and deep copied to backend runtime, so we can save graph
    // in attr to save performance.
    if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
      primitive->set_attr("recent_graph", graph);
    }
    return graph->output()->abstract();
  }
};

inline const PrimitivePtr kPrimPackFunc = std::make_shared<Primitive>(kPackFunc);
REGISTER_PRIMITIVE_OP_CPP_INFER_IMPL(PackFunc, kPrimPackFunc, PackFuncInfer, false);
}  // namespace expander
}  // namespace mindspore
