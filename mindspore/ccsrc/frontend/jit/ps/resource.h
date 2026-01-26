/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_RESOURCE_H_
#define MINDSPORE_CCSRC_FRONTEND_JIT_RESOURCE_H_

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <future>
#include <mutex>
#include <utility>
#include <functional>

#include "utils/hash_map.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "utils/any.h"
#include "utils/profile.h"

#include "frontend/jit/ps/resource_base.h"
#include "frontend/jit/ps/static_analysis/prim.h"
#include "frontend/jit/ps/static_analysis/static_analysis.h"
#include "load_mindir/load_model.h"
#include "frontend/jit/ps/compile_cache_manager.h"
#include "include/frontend/jit/ps/resource_interface.h"

namespace mindspore {
namespace pipeline {
const char kStepParallelGraph[] = "step_parallel";
const char kOutput[] = "output";
const char kBuildBackendType[] = "backend_type";
const char kBuildBackendOutput[] = "backend_output";
const char kNoBackend[] = "no_backend";
const char kPynativeGraphId[] = "graph_id";
const char kActorInfo[] = "actor_info";
const char kCompiler[] = "Compiler";
const char kBootstrap[] = "bootstrap";
const char kParse[] = "parse";
const char kSymbolResolve[] = "symbol_resolve";
const char kSetMixedPrecisionFlag[] = "set_mixed_precision_flag";
const char kCombineLikeGraphs[] = "combine_like_graphs";
const char kGraphReusing[] = "graph_reusing";
const char kPreCConv[] = "pre_cconv";
const char kTypeInference[] = "type_inference";
const char kEventMethod[] = "event_method";
const char kAutoMonad[] = "auto_monad";
const char kInline[] = "inline";
const char kAddAttr[] = "add_attr";
const char kPreAutoParallel[] = "pre_auto_parallel";
const char kPipelineSplit[] = "pipeline_split";
const char kDetachBackwardAction[] = "detach_backward";
const char kPipelineParallelScheduler[] = "pipeline_parallel_scheduler";
const char kOptimize[] = "optimize";
const char kAutoMonadReorder[] = "auto_monad_reorder";
const char kGetJitBpropGraph[] = "get_jit_bprop_graph";
const char kRewriterAfterJitBprop[] = "rewriter_after_jit_bprop_graph";
const char kOptAfterJitGrad[] = "opt_after_jit_grad";
const char kWaitDistCommInitDone[] = "wait_dist_comm_init_done";
const char kUnusedParamsEliminate[] = "eliminate_unused_params";
const char kOffloadAnnotation[] = "process_offload_annotation";
const char kValidate[] = "validate";
const char kLoadMindir[] = "load_mindir";
const char kInferMindir[] = "infer_mindir";
const char kModifyMindirGraph[] = "modify_mindir_graph";
const char kDistributedSplit[] = "distribtued_split";
const char kTaskEmit[] = "task_emit";
const char kExecute[] = "execute";
const char kAbstractAnalyze[] = "AbstractAnalyze";
const char kProgramSpecialize[] = "ProgramSpecialize";
const char kCreateBackend[] = "create_backend";
const char kPipelineClean[] = "pipeline_clean";
const char kPyInterpretToExecute[] = "py_interpret_to_execute";
const char kRewriterBeforeOptA[] = "rewriter_before_opt_a";
const char kAddAttrWithInline[] = "add_attr_with_inline";
const char kExpandDumpFlag[] = "expand_dump_flag";
const char kSwitchSimplifyFlag[] = "switch_simplify";
const char kMetaFgExpandFlag[] = "meta_fg_expand";
const char kSetForwardCommIdForCommNodePass[] = "set_forward_comm_id_for_comm_node_pass";
const char kJitOptA[] = "jit_opt_a";
const char kJitOptB[] = "jit_opt_b";
const char kPyInterpretToExecuteAfterOptA[] = "py_interpret_to_execute_after_opt_a";
const char kRewriterAfterOptA[] = "rewriter_after_opt_a";
const char kConvertAfterRewriter[] = "convert_after_rewriter";
const char kOrderPyExecuteAfterRewriter[] = "order_py_execute_after_rewriter";
const char kCconv[] = "cconv";
const char kLoopUnroll[] = "loop_unroll";
const char kJitOptPassAfterCconv[] = "jit_opt_after_cconv";
const char kRemoveDupValue[] = "remove_dup_value";
const char kPartialUnusedArgsEliminate[] = "partial_unused_args_eliminate";
const char kMutableEliminate[] = "mutable_eliminate";
const char kEnvironConv[] = "environ_conv";
const char kTupleTransform[] = "tuple_transform";
const char kAddRecomputation[] = "add_recomputation";
const char kCseAfterRecomputation[] = "cse_after_recomputation";
const char kBackendPass[] = "backend_pass";
const char kSymbolEngineOpt[] = "symbol_engine_optimizer";

using BuiltInTypeMap = mindspore::HashMap<int64_t, mindspore::HashMap<std::string, Any>>;

BuiltInTypeMap &GetMethodMap();

BuiltInTypeMap &GetAttrMap();
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_RESOURCE_H_
