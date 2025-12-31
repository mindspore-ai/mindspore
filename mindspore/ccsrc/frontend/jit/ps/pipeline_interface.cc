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

#include "include/frontend/jit/ps/pipeline_interface.h"

#include <memory>
#include <map>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <functional>

#include "pybind_api/pybind_patch.h"
#include "pybind11/pybind11.h"
#include "ir/func_graph_cloner.h"
#include "utils/ms_context.h"
#include "tools/profiler/profiling.h"
#include "tools/profiler/profiler.h"
#include "frontend/jit/ps/action.h"
#include "frontend/jit/ps/pass.h"
#include "frontend/jit/ps/fallback.h"
#include "frontend/optimizer/irpass.h"
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/step_auto_parallel.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/allreduce_fusion/step_allreduce_fusion.h"
#include "frontend/parallel/pass/handle_group_info.h"
#include "frontend/parallel/step_assigned_parallel.h"
#include "include/utils/pipeline/fallback.h"
#include "include/utils/tensor_py.h"
#include "include/utils/parallel_context.h"
#include "utils/config_manager.h"
#include "include/frontend/jit/ps/action_interface.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "mindspore/ccsrc/utils/ir_dump/dump_proto.h"

namespace mindspore {
namespace pipeline {
namespace {
void RecordInitStatus() {
  static bool printed = false;
  if (!printed) {
    MS_LOG(INFO) << "Status record: system init.";
    printed = true;
  }
}

py::object GetSelfFromArgs(const py::object &args) {
  if (!py::isinstance<py::tuple>(args)) {
    return py::object();
  }
  auto args_tuple = py::cast<py::tuple>(args);
  if (args_tuple.size() == 0) {
    return py::object();
  }
  py::object first_arg = args_tuple[0];
  if (!py::isinstance<Cell>(first_arg)) {
    return py::object();
  }
  return first_arg;
}

kernel::PyExecuteOutputUserDataPtr GetUserDataFromAddress(const py::object &res) {
  const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() >= kCompatible);
  if (!allow_fallback_runtime) {
    return nullptr;
  }

  if (tensor::IsTensorPy(res)) {
    auto res_tensor = tensor::ConvertToTensor(res);
    MS_EXCEPTION_IF_NULL(res_tensor);
    if (res_tensor->has_user_data(kernel::PyExecuteOutputUserData::key)) {
      return res_tensor->GetUserData().get<kernel::PyExecuteOutputUserData>(kernel::PyExecuteOutputUserData::key);
    }
  }
  return nullptr;
}

template <typename T>
py::object GetVectorRefPyDataWithAbstract(const VectorRef &value_list, const abstract::AbstractSequencePtr &seq_abs) {
  auto value_size = value_list.size();
  auto ret = T(value_size);

  const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() >= kCompatible);
  size_t ref_idx = 0;
  for (size_t i = 0; i < seq_abs->size(); ++i) {
    auto elem_abs = seq_abs->elements()[i];
    if (elem_abs->isa<abstract::AbstractNone>() && !allow_fallback_runtime) {
      continue;
    }
    ret[ref_idx] = BaseRefToPyDataWithUserData(value_list[ref_idx], elem_abs);
    ref_idx++;
  }
  if (ref_idx != value_size) {
    MS_LOG(EXCEPTION) << "The size of elements (excluding None) should be equal to " << value_size << ", but got "
                      << ref_idx;
  }
  return ret;
}

py::object GetVectorRefPyData(const VectorRef &value_list, const AbstractBasePtr &abs) {
  if (abs == nullptr || abs->isa<abstract::AbstractCSRTensor>() || abs->isa<abstract::AbstractCOOTensor>() ||
      abs->isa<abstract::AbstractAny>()) {
    return BaseRefToPyData(value_list, abs);
  }
  // Need to consider AbstractAny with vector ref scene later.
  if (!abs->isa<abstract::AbstractSequence>()) {
    MS_LOG(EXCEPTION) << "Can not convert vector ref with abstract " << abs->ToString();
  }
  auto seq_abs = abs->cast<abstract::AbstractSequencePtr>();
  if (seq_abs->dynamic_len()) {
    return BaseRefToPyData(value_list, abs);
  }
  if (seq_abs->isa<abstract::AbstractTuple>()) {
    return GetVectorRefPyDataWithAbstract<py::tuple>(value_list, seq_abs);
  }
  return GetVectorRefPyDataWithAbstract<py::list>(value_list, seq_abs);
}
}  // namespace

py::bool_ VerifyInputSignature(const py::list &input_signature, const py::tuple &inputs) {
  MS_LOG(DEBUG) << "Verify args size:" << inputs.size();
  if (inputs.size() != input_signature.size()) {
    MS_LOG(ERROR) << "Signature size not equal to args size";
    return false;
  }

  size_t count = 0;
  for (auto arg_obj : inputs) {
    std::shared_ptr<tensor::Tensor> m_tensor = nullptr;
    bool is_tensor = false;
    if (tensor::IsTensorPy(arg_obj)) {
      m_tensor = tensor::ConvertToTensor(arg_obj);
      is_tensor = true;
    }
    if (is_tensor && m_tensor == nullptr) {
      MS_LOG(ERROR) << "Verify Tensor error, get ptr is null";
      return false;
    }

    if (m_tensor != nullptr) {
      MS_LOG(DEBUG) << "Verify Tensor";
      auto sig = tensor::ConvertToTensor(input_signature[count]);
      MS_EXCEPTION_IF_NULL(sig);
      ShapeVector sig_shape = sig->shape();
      TypePtr sig_type = sig->Dtype();

      ShapeVector tensor_shape = m_tensor->shape_c();
      if (tensor_shape != sig_shape) {
        MS_LOG(ERROR) << "Python input shape is incompatible with input_signature";
        return false;
      }

      if (*m_tensor->Dtype() != *sig_type) {
        MS_LOG(ERROR) << "Python input type(" << m_tensor->Dtype()->ToString() << ") incompatible with input_signature("
                      << sig_type->ToString() << ")";
        return false;
      }
    }
    count++;
  }

  return true;
}

FuncGraphPtr LoadMindIR(const std::string &file_name, const char *dec_key, const size_t key_len,
                        const std::string &dec_mode, const py::object decrypt) {
  FuncGraphPtr func_graph = nullptr;
  if (dec_mode == "Customized") {
    py::bytes key_bytes(dec_key);
    py::bytes model_stream = decrypt(file_name, key_bytes);
    std::string model_string(model_stream);

    MindIRLoader mindir_loader;
    func_graph = mindir_loader.LoadMindIR(model_string.c_str(), model_string.size());
  } else {
    MindIRLoader mindir_loader(false, reinterpret_cast<const unsigned char *>(dec_key), key_len, dec_mode, false);
    func_graph = mindir_loader.LoadMindIR(file_name);
  }
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("load.ir", func_graph);
  }
#endif
  return func_graph;
}

FuncGraphPtr SplitMindIR(const std::string &file_name) {
  MS_LOG(INFO) << "Start split mindir";
  FuncGraphPtr func_graph = nullptr;
  MindIRLoader mindir_loader;
  func_graph = mindir_loader.LoadMindIR(file_name);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Load MindIR file failed. Please check model file.";
    return nullptr;
  }
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("load.ir", func_graph);
  }
#endif
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  parallel_context->Reset();
  parallel_context->set_parallel_mode(parallel::kAutoParallel);
  parallel_context->set_strategy_search_mode(parallel::kRecursiveProgramming);
  parallel_context->set_direct_split(true);
  parallel_context->set_full_batch(true);
  parallel_context->set_group_ckpt_save_file("group_info");

  FuncGraphManagerPtr func_graph_manager = func_graph->manager();

  MS_LOG(INFO) << "func_graph_manager is not null";
  if (func_graph_manager == nullptr) {
    std::vector<FuncGraphPtr> graphs{func_graph};
    func_graph_manager = std::make_shared<FuncGraphManager>(graphs);
    func_graph_manager->AddFuncGraph(func_graph);
  }
  pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();
  resource->set_manager(func_graph_manager);

  // Get the parameters items and add the value to args_abs.
  auto params = func_graph->parameters();
  auto inputs = func_graph->get_inputs();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    auto input = inputs[i]->abstract();
    (void)parallel::ExtendInputArgsAbstractShape(input, i);
  }
  parallel::StepAutoParallel(func_graph, NULL);
  parallel::StepParallel(func_graph, NULL);
  parallel::StepAllreduceFusion(func_graph, NULL);
  resource->set_func_graph(func_graph);
  resource->set_manager(func_graph->manager());
  opt::irpass::OptimizeIRPassLib irpass;
  opt::OptPassConfig virtual_dataset = opt::OptPassConfig({irpass.virtual_dataset_eliminate_});
  opt::OptPassConfig virtual_output = opt::OptPassConfig({irpass.virtual_output_eliminate_});

  opt::OptPassGroupMap map_parallel_eliminate(
    {{"virtual_dataset", virtual_dataset}, {"virtual_output", virtual_output}});

  auto split_pass_opts = opt::Optimizer::MakeOptimizer("map_parallel_eliminate", resource, map_parallel_eliminate);
  ProfileExecute(MsProfile::GetProfile()->Step("split_pass_opts"),
                 [&split_pass_opts, &func_graph]() { func_graph = split_pass_opts->step(func_graph, true); });

  AbstractBasePtrList args_abs_list;
  (void)std::transform(params.begin(), params.end(), std::back_inserter(args_abs_list),
                       [](const AnfNodePtr &p) -> AbstractBasePtr { return p->abstract(); });
  func_graph = pipeline::Renormalize(resource, func_graph, args_abs_list);

  resource->set_args_abs(args_abs_list);

  MindIRExporter mindir_exporter;
  mindir_exporter.ExportProto(func_graph, "split_net", nullptr);

  parallel::HandleGroupInfo();

  return func_graph;
}

FuncGraphPtr SplitDynamicMindIR(const std::string &file_name, size_t device_num, size_t rank_id, bool sapp) {
  MS_LOG(INFO) << "Start split dynamic mindir for transformer network";
  FuncGraphPtr func_graph = nullptr;
  MindIRLoader mindir_loader;
  func_graph = mindir_loader.LoadMindIR(file_name);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Load MindIR file failed. Please check model file.";
    return nullptr;
  }
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("load.ir", func_graph);
  }
#endif
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  parallel_context->Reset();
  parallel_context->set_parallel_mode(parallel::kAutoParallel);
  parallel_context->set_strategy_search_mode(parallel::kRecursiveProgramming);
  parallel_context->set_direct_split(true);
  parallel_context->set_full_batch(true);
  parallel_context->set_group_ckpt_save_file("group_info");

  for (size_t rank_id_iter = 0; rank_id_iter < device_num; rank_id_iter++) {
    auto tmp_func_graph = mindspore::BasicClone(func_graph);
    FuncGraphManagerPtr func_graph_manager = tmp_func_graph->manager();
    if (func_graph_manager == nullptr) {
      MS_LOG(INFO) << "func_graph_manager is null";
      std::vector<FuncGraphPtr> graphs{tmp_func_graph};
      func_graph_manager = std::make_shared<FuncGraphManager>(graphs);
      func_graph_manager->AddFuncGraph(tmp_func_graph);
    }

    auto inputs = tmp_func_graph->get_inputs();
    for (std::size_t i = 0; i < inputs.size(); i++) {
      auto input = inputs[i]->abstract();
      (void)parallel::ExtendInputArgsAbstractShape(input, i);
    }

    auto res = parallel::StepAssignedParallel(tmp_func_graph, func_graph_manager, device_num, rank_id_iter, sapp);
    if (!res) {
      MS_LOG(ERROR) << "StepAssignedParallel failed. Please check.";
      return nullptr;
    }
    pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();
    resource->set_is_load(false);
    resource->set_manager(func_graph_manager);
    resource->set_func_graph(tmp_func_graph);
    // Get the parameters items and add the value to args_abs.
    auto params = tmp_func_graph->parameters();
    AbstractBasePtrList args_abs_list;
    (void)std::transform(params.begin(), params.end(), std::back_inserter(args_abs_list),
                         [](const AnfNodePtr &p) -> AbstractBasePtr { return p->abstract(); });
    tmp_func_graph = pipeline::Renormalize(resource, tmp_func_graph, args_abs_list);

#ifdef ENABLE_DUMP_IR
    auto re_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(re_context);
    if (re_context->CanDump(kIntroductory)) {
      string renormalize_net_name = "Renomalize_" + std::to_string(rank_id_iter) + ".ir";
      DumpIR(renormalize_net_name, tmp_func_graph);
    }
#endif

    parallel::HandleGroupInfo();
    string net_save_name = "split_net" + std::to_string(rank_id_iter);
    MindIRExporter mindir_exporter;
    res = mindir_exporter.ExportProto(tmp_func_graph, net_save_name, nullptr);
    if (!res) {
      MS_LOG(ERROR) << "Export MindIR file failed failed. Please check.";
      return nullptr;
    }
  }

  return func_graph;
}

void InitPipeline() {
  // set python env flag
  RecordInitStatus();
  mindspore::python_adapter::set_python_env_flag(true);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  CompileConfigManager::GetInstance().CollectCompileConfig();
}

bool RunJitPipeline() {
  bool is_auto_parallel = (parallel::ParallelContext::GetInstance()->parallel_mode() == parallel::kSemiAutoParallel ||
                           parallel::ParallelContext::GetInstance()->parallel_mode() == parallel::kAutoParallel);
  if (is_auto_parallel || common::GetEnv("MS_DEV_JIT_PIPELINE") == "0") {
    return false;
  }
  return true;
}

std::string DumpFuncGraph(const py::object &obj) {
  auto func_graph = obj.cast<FuncGraphPtr>();
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "The python object is not a func_graph";
    return "";
  }
  std::ostringstream out_oss;
  DumpIR(out_oss, func_graph, false, kOff, false);
  return out_oss.str();
}

void PreJit(const py::object &args, const py::object &kwargs) {
  const auto &self = GetSelfFromArgs(args);
  parse::Parser::InitParserEnvironment(self);
}

py::object BaseRefToPyDataWithUserData(const BaseRef &value, const abstract::AbstractBasePtr &abs) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kGraphExecutorPy, runtime::ProfilerEvent::kOutputProcess,
                                     "BaseRefToPyData");
  const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() >= kCompatible);
  if (!allow_fallback_runtime) {
    return BaseRefToPyData(value, abs);
  }
  if (utils::isa<ValuePtr>(value)) {
    // Do not use abs as input to BaseRefToPyData, since the res need to be a tensor to get user data.
    auto res = BaseRefToPyData(value);
    const auto user_data = GetUserDataFromAddress(res);
    if (user_data != nullptr) {
      return user_data->obj;
    } else {
      MS_LOG(DEBUG) << "user data is empty";
    }
  } else if (utils::isa<VectorRef>(value)) {
    auto vec_ref = utils::cast<VectorRef>(value);
    return GetVectorRefPyData(vec_ref, abs);
  }
  return BaseRefToPyData(value, abs);
}
}  // namespace pipeline
}  // namespace mindspore
