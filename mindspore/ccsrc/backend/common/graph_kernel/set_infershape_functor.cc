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
#include "backend/common/graph_kernel/set_infershape_functor.h"

#include <algorithm>
#include <memory>

#include "mindspore/core/symbolic_shape/symbol_engine.h"
#include "include/common/utils/anfalgo.h"
#include "ir/anf.h"
#include "backend/common/graph_kernel/symbol_engine/jit/transform_visitor.h"
#include "backend/common/graph_kernel/symbol_engine/multi_symbol_engine.h"
#include "backend/common/graph_kernel/symbol_engine/jit/cpp_visitor.h"

namespace mindspore::graphkernel {
BaseShapePtr SymbolEngineInfer::InferShape(const CNodePtr &cnode, const AbstractBasePtrList &args) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(DEBUG) << "Infer shape using symbol engine for cnode: " << cnode->fullname_with_scope();
  auto func_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(cnode);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto output = func_graph->output();
  auto symbol_engine = func_graph->symbol_engine();
  MS_EXCEPTION_IF_NULL(symbol_engine);
  if (!symbol_engine->Infer(args)) {
    MS_LOG(WARNING) << "Infer failed by symbol engine. node " << cnode->fullname_with_scope();
    return nullptr;
  }
  return symbol_engine->QueryShape(output);
}

BaseShapePtr SymbolEngineJitInfer::InferShape(const CNodePtr &cnode, const AbstractBasePtrList &inputs) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(DEBUG) << "Infer shape using symbol engine for cnode: " << cnode->fullname_with_scope();

  // Load library
  if (infer_func_ == nullptr) {
    MS_LOG(DEBUG) << " Start to load function";
    infer_func_ = cpp_visitor_->LoadFunc(func_name_);
  }

  // Prepare inputs
  std::vector<const int64_t *> input_parm(inputs.size());
  std::transform(inputs.begin(), inputs.end(), input_parm.begin(), [](AbstractBasePtr abs) {
    auto base_shape_p = abs->GetShape();
    MS_EXCEPTION_IF_NULL(base_shape_p);
    auto shape_p = base_shape_p->cast<abstract::TensorShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_p);
    return shape_p->shape().data();
  });

  // Execute infer function
  MS_LOG(DEBUG) << " Start to run function: " << func_name_;
  infer_func_(input_parm.data(), output_parm_.data());
  MS_LOG(DEBUG) << "After run function: " << func_name_ << " output: " << out_shapes_;

  abstract::AbstractBasePtr out_abs = cnode->abstract();
  if (out_abs->isa<abstract::AbstractTuple>()) {
    abstract::BaseShapePtrList shapes(out_shapes_.size());
    (void)std::transform(out_shapes_.begin(), out_shapes_.end(), shapes.begin(),
                         [](const ShapeVector &s) { return std::make_shared<abstract::TensorShape>(s); });
    return std::make_shared<abstract::TupleShape>(shapes);
  }
  return std::make_shared<abstract::TensorShape>(out_shapes_.front());
}

void SymbolEngineJitInfer::Init() {
  // Prepare outputs space
  MS_EXCEPTION_IF_CHECK_FAIL(output_symbol_->HasData(), "SymbolEngineJit does not support dynamic rank");
  if (output_symbol_->size() == 0) {
    out_shapes_.resize(1, ShapeVector(1));
    output_parm_.resize(1, out_shapes_[0].data());
    return;
  }
  if (output_symbol_->item(0)->is<IntSymbol>()) {
    out_shapes_.resize(1, ShapeVector(output_symbol_->size()));
    output_parm_.resize(1, out_shapes_[0].data());
  } else {
    out_shapes_.resize(output_symbol_->size());
    output_parm_.resize(out_shapes_.size());
    for (size_t i = 0; i < out_shapes_.size(); ++i) {
      auto out_i = output_symbol_->item_as<ListSymbol>(i);
      MS_EXCEPTION_IF_NULL(out_i);
      out_shapes_[i].resize(out_i->size());
      output_parm_[i] = out_shapes_[i].data();
    }
  }
}

bool Process(const AnfNodePtrList &cnodes, bool use_jit) {
  symshape::CppVisitorPtr cpp_visitor = nullptr;
  if (use_jit) {
    cpp_visitor = std::make_shared<symshape::CppVisitor>();
  }

  bool changed = false;
  for (const auto &cnode : cnodes) {
    if (common::AnfAlgo::IsGraphKernel(cnode) && common::AnfAlgo::IsDynamicShape(cnode)) {
      auto func_graph = GetCNodeFuncGraph(cnode);
      MS_EXCEPTION_IF_NULL(func_graph);
      if (func_graph->symbol_engine() != nullptr) {
        bool jit_succeed = false;
        symshape::TransformVisitor transform_visitor;
        if (use_jit) {
          transform_visitor.Init(func_graph);
          jit_succeed = transform_visitor.Transform(func_graph);
        }
        if (jit_succeed) {
          auto func_name = cpp_visitor->CodeGen(
            transform_visitor.GetShapes(), transform_visitor.GetSymbolTable(),
            (func_graph->has_attr("info_name") ? GetValue<std::string>(func_graph->get_attr("info_name")) : ""));
          MS_LOG(DEBUG) << "Set infershape functor(SymbolEngineJit) for cnode: " << cnode->fullname_with_scope();
          auto output_symbol = func_graph->output()->abstract()->GetSymbolicShape();
          MS_EXCEPTION_IF_NULL(output_symbol);
          common::AnfAlgo::SetNodeAttrSafely(
            "infer_shape_functor",
            std::make_shared<SymbolEngineJitInfer>("symbol_engine_jit_infer_functor", func_name, cpp_visitor,
                                                   output_symbol),
            cnode);
        } else {
          MS_LOG(DEBUG) << "Set infershape functor for cnode: " << cnode->fullname_with_scope();
          common::AnfAlgo::SetNodeAttrSafely("infer_shape_functor",
                                             std::make_shared<SymbolEngineInfer>("symbol_engine_infer_functor"), cnode);
        }
        changed = true;
      }
    }
  }
  if (use_jit) {
    cpp_visitor->Compile();
  }
  return changed;
}

bool SetInferShapeFunctor::Run(const FuncGraphPtr &func_graph) {
  auto cnodes = TopoSort(func_graph->output(), SuccIncoming,
                         [](const AnfNodePtr &node) { return node->isa<CNode>() ? FOLLOW : EXCLUDE; });
  bool use_jit = common::GetEnv("MS_DEV_SYMBOL_JIT") != "off";
#if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER))
  use_jit = false;
#endif
  return Process(cnodes, use_jit);
}
}  // namespace mindspore::graphkernel
