/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include <pybind11/operators.h>
#include <stack>
#include "kernel/oplib/oplib.h"
#include "pipeline/jit/ps/pipeline.h"
#include "frontend/operator/composite/composite.h"
#include "pipeline/pynative/pynative_execute.h"
#include "utils/symbolic.h"
#include "include/common/pybind_api/api_register.h"
#include "include/common/utils/python_adapter.h"
#ifndef ENABLE_SECURITY
#include "include/common/utils/summary/event_writer.h"
#endif
#include "include/common/utils/config_manager.h"
#include "include/common/utils/mpi/mpi_config.h"
#include "utils/ms_utils.h"
#include "utils/ms_context.h"
#include "include/common/utils/parallel_context.h"
#include "include/common/utils/offload_context.h"
#include "frontend/parallel/costmodel_context.h"
#if ((defined ENABLE_CPU) && (!defined _WIN32))
#include "include/backend/distributed/ps/util.h"
#endif
#include "include/backend/distributed/ps/ps_context.h"
#include "include/backend/distributed/init.h"
#include "include/backend/distributed/recovery/recovery_context.h"
#include "include/backend/distributed/collective/collective_manager.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "runtime/graph_scheduler/embedding_cache_scheduler.h"
#endif
#include "runtime/hardware/device_context_manager.h"
#include "include/backend/mem_reuse/mem_dynamic_allocator.h"
#include "frontend/parallel/tensor_layout/tensor_transform.h"

#include "pybind_api/gil_scoped_long_running.h"

#ifndef ENABLE_SECURITY
#include "include/backend/debug/profiler/profiling.h"
#endif
#include "include/common/profiler.h"

#include "pipeline/jit/pi/external.h"
#include "include/common/np_dtype/np_dtypes.h"
#include "include/common/amp/amp.h"

namespace py = pybind11;
using GraphExecutorPy = mindspore::pipeline::GraphExecutorPy;
using Pipeline = mindspore::pipeline::Pipeline;
using PrimitivePy = mindspore::PrimitivePy;
using MetaFuncGraph = mindspore::MetaFuncGraph;
#ifndef ENABLE_SECURITY
using EventWriter = mindspore::summary::EventWriter;
#endif  // ENABLE_SECURITY
using OpLib = mindspore::kernel::OpLib;
using ParallelContext = mindspore::parallel::ParallelContext;
using CostModelContext = mindspore::parallel::CostModelContext;
using TensorTransform = mindspore::parallel::TensorTransform;
using OffloadContext = mindspore::OffloadContext;
using mindspore::MsCtxParam;
using PSContext = mindspore::ps::PSContext;
using CollectiveManager = mindspore::distributed::collective::CollectiveManager;
using RecoveryContext = mindspore::distributed::recovery::RecoveryContext;
using DeviceContextManager = mindspore::device::DeviceContextManager;
using DeviceContext = mindspore::device::DeviceContext;

constexpr int PROFILER_RECORD_STAMP = 2;

#ifndef ENABLE_SECURITY
namespace mindspore {
namespace profiler {
void RegProfiler(const py::module *m) {
  (void)py::class_<Profiler, std::shared_ptr<Profiler>>(*m, "Profiler")
    .def_static("get_instance", &Profiler::GetInstance, py::arg("device_name"), "Profiler get_instance.")
    .def("init", &Profiler::Init, py::arg("profiling_path"), py::arg("device_id") = py::int_(0),
         py::arg("profiling_options") = py::str(""), "init")
    .def("start", &Profiler::Start, "start")
    .def("stop", &Profiler::Stop, "stop")
    .def("finalize", &Profiler::Finalize, "finalize")
    .def("sync_enable", &Profiler::SyncEnable, py::arg("enable_flag"))
    .def("data_process_enable", &Profiler::DataProcessEnable, py::arg("enable_flag"))
    .def("step_profiling_enable", &Profiler::StepProfilingEnable, py::arg("enable_flag"),
         "enable or disable step profiling")
    .def("enable_op_time", &Profiler::EnableOpTime, "Enable op_time.")
    .def("enable_profile_memory", &Profiler::EnableProfileMemory, "Enable profile_memory.");
}
void RegProfilerManager(const py::module *m) {
  (void)py::class_<ProfilerManager, std::shared_ptr<ProfilerManager>>(*m, "ProfilerManager")
    .def_static("get_instance", &ProfilerManager::GetInstance, "ProfilerManager get_instance.")
    .def("dynamic_status", &ProfilerManager::GetNetDynamicShapeStatus, "dynamic_status")
    .def("set_profile_framework", &ProfilerManager::SetProfileFramework, py::arg("profile_framework"));
}

// level: 0, for developer user, 1, for general user;
// profile_framework: 0, all host info, 1, host memory, 2, host time;
// start_end: 0, start flag, 1, end flag, 2, no distinguish start and end.
// Default parameter for host profile meaning: for developer user, collect both time and memory, record timestamp.
void RegHostProfile(py::module *m) {
  m->def("_collect_host_info", &CollectHostInfo, py::arg("module_name"), py::arg("event"), py::arg("stage"),
         py::arg("level") = py::int_(0), py::arg("profile_framework") = py::int_(0),
         py::arg("start_end") = py::int_(PROFILER_RECORD_STAMP), py::arg("custom_info") = py::dict())
    .def("get_clock_time", &GetClockTime)
    .def("get_clock_syscnt", &GetClockSyscnt);
}

void RegFrameworkProfiler(py::module *m) {
  m->def(
     "_framework_profiler_step_start", []() { runtime::ProfilerAnalyzer::GetInstance().StartStep(); },
     "Profiler step start")
    .def(
      "_framework_profiler_step_end", []() { runtime::ProfilerAnalyzer::GetInstance().EndStep(); }, "Profiler step end")
    .def(
      "_framework_profiler_clear", []() { runtime::ProfilerAnalyzer::GetInstance().Clear(); },
      "Dump json and clear data")
    .def("_framework_profiler_enable_mi", []() { runtime::ProfilerAnalyzer::GetInstance().EnableMiProfile(); });
}

void RegFrameworkPythonProfileRecorder(py::module *m) {
  (void)py::class_<runtime::PythonProfilerRecorder, std::shared_ptr<runtime::PythonProfilerRecorder>>(
    *m, "PythonProfilerRecorder")
    .def(py::init<const std::string &>())
    .def("record_start", &runtime::PythonProfilerRecorder::record_start, "record_start")
    .def("record_end", &runtime::PythonProfilerRecorder::record_end, "record_end");
}

}  // namespace profiler
}  // namespace mindspore
#endif  // ENABLE_SECURITY

namespace mindspore {
void RegModule(py::module *m) {
  RegTyping(m);
  RegCNode(m);
  RegCell(m);
  RegMetaFuncGraph(m);
  RegFuncGraph(m);
  RegUpdateFuncGraphHyperParams(m);
  RegParamInfo(m);
  RegPrimitive(m);
  RegPrimitiveFunction(m);
  RegSignatureEnumRW(m);
  RegRandomSeededGenerator(m);
  mindspore::tensor::RegMetaTensor(m);
  mindspore::tensor::RegCSRTensor(m);
  mindspore::tensor::RegCOOTensor(m);
  mindspore::tensor::RegRowTensor(m);
  mindspore::tensor::RegMapTensor(m);
  RegValues(m);
  mindspore::initializer::RegRandomNormal(m);
  RegMsContext(m);
  RegSecurity(m);
  RegForkUtils(m);
  RegNumpyTypes(m);
  RegAmpModule(m);
  mindspore::hal::RegStream(m);
  mindspore::hal::RegEvent(m);
  mindspore::hal::RegMemory(m);
  mindspore::pynative::RegPyNativeExecutor(m);
  mindspore::pynative::RegisterPyBoostFunction(m);
  mindspore::pijit::RegPIJitInterface(m);
  mindspore::prim::RegCompositeOpsGroup(m);
#ifndef ENABLE_SECURITY
  mindspore::profiler::RegProfilerManager(m);
  mindspore::profiler::RegProfiler(m);
  mindspore::profiler::RegHostProfile(m);
  mindspore::profiler::RegFrameworkProfiler(m);
  mindspore::profiler::RegFrameworkPythonProfileRecorder(m);
#endif
#ifdef _MSC_VER
  mindspore::abstract::RegPrimitiveFrontEval();
#endif
  mindspore::ops::RegOpEnum(m);
}

void RegModuleHelper(py::module *m) {
  static std::once_flag onlyCalledOnce;
  std::call_once(onlyCalledOnce, RegModule, m);
}
}  // namespace mindspore

// Interface with python
PYBIND11_MODULE(_c_expression, m) {
  // The OMP_NUM_THREADS has no effect when set in backend, so set it here in advance.
  mindspore::common::SetOMPThreadNum();

  m.doc() = "MindSpore c plugin";

  mindspore::RegModuleHelper(&m);
  mindspore::ScopedLongRunning::SetHook(std::make_unique<mindspore::GilScopedLongRunningHook>());

  // Class Pipeline interface
  MS_LOG(INFO) << "Start GraphExecutorPy...";
  (void)py::class_<GraphExecutorPy, std::shared_ptr<GraphExecutorPy>>(m, "GraphExecutor_")
    .def_static("get_instance", &GraphExecutorPy::GetInstance, "Executor get_instance.")
    .def("__call__", &GraphExecutorPy::Run, py::arg("args"), py::arg("phase") = py::str(""), "Executor run function.")
    .def("del_net_res", &GraphExecutorPy::DelNetRes, py::arg("obj"), py::arg("network_id") = py::set(),
         "Delete network resource.")
    .def("get_func_graph", &GraphExecutorPy::GetFuncGraph, py::arg("phase") = py::str(""), "Get graph pointer.")
    .def("get_func_graph_proto", &GraphExecutorPy::GetFuncGraphProto, py::arg("phase") = py::str(""),
         py::arg("type") = py::str("onnx_ir"), py::arg("incremental") = py::bool_(false),
         "Get graph proto string by specifying ir type.")
    .def("get_obfuscate_func_graph_proto", &GraphExecutorPy::GetObfuscateFuncGraphProto, py::arg("phase") = py::str(""),
         py::arg("incremental") = py::bool_(false), py::arg("obf_ratio") = py::float_(1.0),
         py::arg("branch_control_input") = py::int_(0), "Get graph proto of dynamic-obfuscated model.")
    .def("get_params", &GraphExecutorPy::GetParams, py::arg("phase") = py::str(""), "Get Parameters from graph")
    .def("get_random_status", &GraphExecutorPy::GetRandomStatus, py::arg("phase") = py::str(""),
         "Get random status from graph")
    .def("compile", &GraphExecutorPy::Compile, py::arg("obj"), py::arg("args"), py::arg("kwargs"),
         py::arg("phase") = py::str(""), py::arg("use_vm") = py::bool_(false), "Compile obj by executor.")
    .def("updata_param_node_default_input", &GraphExecutorPy::UpdataParamNodeDefaultInput, py::arg("phase"),
         py::arg("params"), "Fetch the inputs of Conv or Matmul for quant export.")
    .def("get_parameter_layout", &GraphExecutorPy::GetParameterLayout, py::arg("phase") = py::str("train"),
         "Get Parameter Tensor Layout Dictionary.")
    .def("flops_collection", &GraphExecutorPy::FlopsCollection, py::arg("phase") = py::str("train"),
         "Get model flops information.")
    .def("get_parallel_graph_info", &GraphExecutorPy::GetParallelGraphInfo, py::arg("phase") = py::str("train"),
         "Get graph info in step_parallel stage.")
    .def("get_parallel_parameter_name_list", &GraphExecutorPy::GetParallelParameterNameList,
         py::arg("phase") = py::str("train"), "Get Parallel Parameter Name List.")
    .def("get_strategy", &GraphExecutorPy::GetCNodeStrategy, py::arg("phase") = py::str("train"),
         "Get CNode Strategy Dictionary.")
    .def("get_num_parallel_ops", &GraphExecutorPy::GetNumOpsInfo, py::arg("phase") = py::str("train"),
         "Get the number of parallel operators.")
    .def("get_allreduce_fusion", &GraphExecutorPy::GetAllreduceFusion, py::arg("phase") = py::str("train"),
         "Get Allreduce Fusion Dictionary.")
    .def("build_data_graph", &GraphExecutorPy::BuildGraph, py::arg("build_params"), py::arg("phase") = py::str("train"),
         "Build data graph.")
    .def("export_graph", &GraphExecutorPy::ExportGraph, py::arg("file_name"), py::arg("phase"),
         py::arg("encrypt") = py::none(), py::arg("key") = nullptr, "Export Graph.")
    .def("has_compiled", &GraphExecutorPy::HasCompiled, py::arg("phase") = py::str(""), "Get if cell compiled.")
    .def("set_py_exe_path", &GraphExecutorPy::PyExePath, py::arg("py_exe_path") = py::str(""),
         "Set python executable path.")
    .def("set_kernel_build_server_dir", &GraphExecutorPy::KernelBuildServerDir,
         py::arg("kernel_build_server_dir") = py::str(""), "Set kernel build server directory path.")
    .def("set_queue_name", &GraphExecutorPy::set_queue_name, py::arg("queue_name") = py::str(""),
         "Set queue name for the graph loaded from compile cache.")
    .def("get_queue_name", &GraphExecutorPy::get_queue_name,
         "Get cached queue name for the graph loaded from compile cache.")
    .def("set_enable_tuple_broaden", &GraphExecutorPy::set_enable_tuple_broaden,
         py::arg("enable_tuple_broaden") = py::bool_(false), "Set tuple broaden enable.")
    .def("set_compile_cache_dep_files", &GraphExecutorPy::set_compile_cache_dep_files,
         py::arg("compile_cache_dep_files") = py::list(), "Set the compilation cache dependent files.")
    .def("set_weights_values", &GraphExecutorPy::set_weights_values, py::arg("weights") = py::dict(),
         "Set values of weights.")
    .def("get_optimize_graph_proto", &GraphExecutorPy::GetOptimizeGraphProto, py::arg("phase") = py::str(""),
         "Get the optimize graph proto string.")
    .def("set_jit_config", &GraphExecutorPy::SetJitConfig, py::arg("jit_config") = py::dict(), "Set the jit config.")
    .def("generate_arguments_key", &GraphExecutorPy::GenerateArgumentsKey, "Generate unique key of argument.")
    .def("check_argument_consistency", &GraphExecutorPy::CheckArgumentsConsistency, "Check equal of arguments.")
    .def("clear_compile_arguments_resource", &GraphExecutorPy::ClearCompileArgumentsResource,
         "Clear resource when phase cached.")
    .def("inc_graph_cell_count", &GraphExecutorPy::IncGraphCellCount, "Increase the count of GraphCell instance.")
    .def("dec_graph_cell_count", &GraphExecutorPy::DecGraphCellCount, "Decrease the count of GraphCell instance.");

  (void)m.def("reset_op_id", &mindspore::pipeline::ResetOpId, "Reset Operator Id");
  (void)m.def("reset_op_id_with_offset", &mindspore::pipeline::ResetOpIdWithOffset, "Reset Operator Id With Offset");
  (void)m.def("init_hccl", &mindspore::pipeline::InitHccl, "Init Hccl");
  (void)m.def("finalize_hccl", &mindspore::pipeline::FinalizeHccl, "Finalize Hccl");
  (void)m.def("get_hccl_rank_id", &mindspore::pipeline::GetHcclRankId, "Get Hccl Rank Id");
  (void)m.def("get_hccl_rank_size", &mindspore::pipeline::GetHcclRankSize, "Get Hccl Rank Size");
  (void)m.def("verify_inputs_signature", &mindspore::pipeline::VerifyInputSignature, "Verify input signature.");
  (void)m.def("init_exec_dataset", &mindspore::pipeline::InitExecDataset, py::arg("queue_name"), py::arg("size"),
              py::arg("batch_size"), py::arg("types"), py::arg("shapes"), py::arg("input_indexs"),
              py::arg("phase") = py::str("dataset"), py::arg("need_run") = py::bool_(true), "Init and exec dataset.");
  (void)m.def("_set_dataset_mode_config", &mindspore::ConfigManager::SetDatasetModeConfig, "API for set dataset mode.");
  (void)m.def("init_pipeline", &mindspore::pipeline::InitPipeline, "Init Pipeline.");
  (void)m.def("load_mindir", &mindspore::pipeline::LoadMindIR, py::arg("file_name"), py::arg("dec_key") = nullptr,
              py::arg("key_len") = py::int_(0), py::arg("dec_mode") = py::str("AES-GCM"),
              py::arg("decrypt") = py::none(), py::arg("obfuscated") = py::bool_(false), "Load model as Graph.");
  (void)m.def("split_mindir", &mindspore::pipeline::SplitMindIR, py::arg("file_name"),
              "Split single mindir to distributed mindir");
  (void)m.def("split_dynamic_mindir", &mindspore::pipeline::SplitDynamicMindIR, py::arg("file_name"),
              py::arg("device_num") = py::int_(8), py::arg("rank_id") = py::int_(0), py::arg("sapp") = py::bool_(true),
              "Split single mindir to distributed mindir");
  (void)m.def("dynamic_obfuscate_mindir", &mindspore::pipeline::DynamicObfuscateMindIR, py::arg("file_name"),
              py::arg("obf_ratio"), py::arg("branch_control_input") = py::int_(0), py::arg("dec_key") = nullptr,
              py::arg("key_len") = py::int_(0), py::arg("dec_mode") = py::str("AES-GCM"),
              "Obfuscate a mindir model by dynamic obfuscation.");
  (void)m.def("init_cluster", &mindspore::distributed::Initialize, "Init Cluster");
  (void)m.def("set_cluster_exit_with_exception", &mindspore::distributed::set_cluster_exit_with_exception,
              "Set this process exits with exception.");

  (void)py::class_<mindspore::MpiConfig, std::shared_ptr<mindspore::MpiConfig>>(m, "MpiConfig")
    .def_static("get_instance", &mindspore::MpiConfig::GetInstance, "Get mpi config instance.")
    .def("get_enable_mpi", &mindspore::MpiConfig::enable_mpi, "Get whether enable mpi.")
    .def("set_enable_mpi", &mindspore::MpiConfig::set_enable_mpi, "Set whether to enable mpi.");

  (void)py::class_<TensorTransform, std::shared_ptr<TensorTransform>>(m, "TensorTransform")
    .def_static("get_instance", &TensorTransform::GetInstance, "Get tensor_transform instance.")
    .def("transform_tensor_sharding", &TensorTransform::TransformOperators, "Transform the tensor sharding.");
  MS_LOG(INFO) << "Start ParallelContext...";
  (void)py::class_<ParallelContext, std::shared_ptr<ParallelContext>>(m, "AutoParallelContext")
    .def_static("get_instance", &ParallelContext::GetInstance, "Get auto parallel context instance.")
    .def("get_device_num", &ParallelContext::device_num, "Get device num.")
    .def("set_hccl_test_avaible", &ParallelContext::set_hccl_test_available, "Set hccl test available.")
    .def("set_device_num", &ParallelContext::set_device_num, "Set device num.")
    .def("get_device_num_is_set", &ParallelContext::device_num_is_set, "Get device num is set.")
    .def("set_fusion_threshold_mb", &ParallelContext::set_fusion_threshold_mb, "Set fusion threshold.")
    .def("set_allgather_fusion_threshold_mb", &ParallelContext::set_allgather_fusion_threshold_mb,
         "Set allgather fusion threshold.")
    .def("set_reducescatter_fusion_threshold_mb", &ParallelContext::set_reducescatter_fusion_threshold_mb,
         "Set reducescatter fusion threshold.")
    .def("fusion_threshold_mb", &ParallelContext::fusion_threshold_mb, "Get allreduce fusion threshold.")
    .def("allgather_fusion_threshold_mb", &ParallelContext::allgather_fusion_threshold_mb,
         "Get allgather fusion threshold.")
    .def("reducescatter_fusion_threshold_mb", &ParallelContext::reducescatter_fusion_threshold_mb,
         "Get reduce_scatter fusion threshold.")
    .def("set_fusion_mode", &ParallelContext::set_fusion_mode, "Get fusion mode.")
    .def("get_fusion_mode", &ParallelContext::get_fusion_mode, "Get fusion mode.")
    .def("get_global_rank", &ParallelContext::global_rank, "Get global rank.")
    .def("set_global_rank", &ParallelContext::set_global_rank, "Set global rank.")
    .def("get_grad_accumulation_shard", &ParallelContext::grad_accumulation_shard, "Get grad_accumulation_shard.")
    .def("set_grad_accumulation_shard", &ParallelContext::set_grad_accumulation_shard, "Set grad_accumulation_shard.")
    .def("get_parallel_optimizer_threshold", &ParallelContext::get_parallel_optimizer_threshold, "Get opt threshold.")
    .def("set_parallel_optimizer_threshold", &ParallelContext::set_parallel_optimizer_threshold, "Set opt threshold.")
    .def("get_global_rank_is_set", &ParallelContext::global_rank_is_set, "Get global rank is set.")
    .def("get_gradients_mean", &ParallelContext::gradients_mean, "Get mirror mean.")
    .def("set_gradients_mean", &ParallelContext::set_gradients_mean, "Set mirror mean.")
    .def("get_gradient_fp32_sync", &ParallelContext::gradient_fp32_sync, "Get cast before mirror.")
    .def("set_gradient_fp32_sync", &ParallelContext::set_gradient_fp32_sync, "Set cast before mirror.")
    .def("get_loss_repeated_mean", &ParallelContext::loss_repeated_mean, "Get loss repeated mean.")
    .def("set_loss_repeated_mean", &ParallelContext::set_loss_repeated_mean, "Set loss repeated mean.")
    .def("get_parallel_mode", &ParallelContext::parallel_mode, "Get parallel mode.")
    .def("set_parallel_mode", &ParallelContext::set_parallel_mode, "Set parallel mode.")
    .def("get_grad_accumulation_step", &ParallelContext::grad_accumulation_step, "Get grad accumulation step.")
    .def("set_grad_accumulation_step", &ParallelContext::set_grad_accumulation_step, "Set grad accumulation step.")
    .def("get_strategy_search_mode", &ParallelContext::strategy_search_mode, "Get strategy search mode.")
    .def("set_strategy_search_mode", &ParallelContext::set_strategy_search_mode, "Set strategy search mode.")
    .def("set_all_reduce_fusion_split_indices", &ParallelContext::SetAllReduceFusionSplitIndices,
         "Set all reduce fusion split indices.")
    .def("get_all_reduce_fusion_split_indices", &ParallelContext::GetAllReduceFusionSplitIndices,
         "Get all reduce fusion split indices.")
    .def("set_all_reduce_fusion_split_sizes", &ParallelContext::SetAllReduceFusionSplitSizes,
         "Set all reduce fusion split sizes.")
    .def("get_all_reduce_fusion_split_sizes", &ParallelContext::GetAllReduceFusionSplitSizes,
         "Get all reduce fusion split sizes.")
    .def("set_enable_all_reduce_fusion", &ParallelContext::set_enable_all_reduce_fusion,
         "Set enable/disable all reduce fusion.")
    .def("get_enable_all_reduce_fusion", &ParallelContext::enable_all_reduce_fusion,
         "Get enable/disable all reduce fusion.")
    .def("set_enable_all_gather_fusion", &ParallelContext::set_enable_all_gather_fusion,
         "Set enable/disable all gather fusion.")
    .def("get_enable_all_gather_fusion", &ParallelContext::enable_all_gather_fusion,
         "Get enable/disable all gather fusion.")
    .def("set_enable_reduce_scatter_fusion", &ParallelContext::set_enable_reduce_scatter_fusion,
         "Set enable/disable reduce scatter fusion.")
    .def("get_enable_reduce_scatter_fusion", &ParallelContext::enable_reduce_scatter_fusion,
         "Get enable/disable reduce scatter fusion.")
    .def("get_parameter_broadcast", &ParallelContext::parameter_broadcast, "Get parameter broadcast.")
    .def("get_parameter_broadcast_is_set", &ParallelContext::parameter_broadcast_is_set,
         "Get parameter broadcast is set.")
    .def("set_parameter_broadcast", &ParallelContext::set_parameter_broadcast, "Set parameter broadcast.")
    .def("set_strategy_ckpt_load_file", &ParallelContext::set_strategy_ckpt_load_file,
         "Set strategy checkpoint load file.")
    .def("set_strategy_ckpt_save_file", &ParallelContext::set_strategy_ckpt_save_file,
         "Set strategy checkpoint save file.")
    .def("get_strategy_ckpt_load_file", &ParallelContext::strategy_ckpt_load_file, "Get strategy checkpoint load file.")
    .def("get_strategy_ckpt_save_file", &ParallelContext::strategy_ckpt_save_file, "Get strategy checkpoint save file.")
    .def("set_group_ckpt_save_file", &ParallelContext::set_group_ckpt_save_file, "Set group checkpoint save file.")
    .def("set_pipeline_stage_split_num", &ParallelContext::set_pipeline_stage_split_num,
         "Set pipeline stage split num.")
    .def("get_pipeline_stage_split_num", &ParallelContext::pipeline_stage_split_num, "Get pipeline stage split num.")
    .def("set_auto_pipeline", &ParallelContext::set_auto_pipeline, "Set the pipeline stage number to automatic.")
    .def("get_auto_pipeline", &ParallelContext::auto_pipeline, "Get whether the pipeline stage number is automatic.")
    .def("set_pipeline_result_broadcast", &ParallelContext::set_pipeline_result_broadcast,
         "Set pipeline result broadcast")
    .def("get_pipeline_result_broadcast", &ParallelContext::pipeline_result_broadcast, "Get pipeline result broadcast")
    .def("set_pipeline_segment_split_num", &ParallelContext::set_pipeline_segment_split_num,
         "Set pipeline segment split num.")
    .def("get_pipeline_segment_split_num", &ParallelContext::pipeline_segment_split_num,
         "Get pipeline segment split num.")
    .def("set_pipeline_interleave", &ParallelContext::set_pipeline_interleave, "Set pipeline interleave.")
    .def("get_pipeline_interleave", &ParallelContext::pipeline_interleave, "Get pipeline interleave.")
    .def("set_pipeline_scheduler", &ParallelContext::set_pipeline_scheduler, "Set pipeline scheduler.")
    .def("get_pipeline_scheduler", &ParallelContext::pipeline_scheduler, "Get pipeline scheduler.")
    .def("set_full_batch", &ParallelContext::set_full_batch, "Set whether load full batch on each device.")
    .def("get_full_batch", &ParallelContext::full_batch, "Get whether load full batch on each device.")
    .def("get_full_batch_is_set", &ParallelContext::full_batch_is_set, "Get whether attr full_batch is set.")
    .def("set_dataset_strategy", &ParallelContext::set_dataset_strategy, "Set dataset sharding strategy.")
    .def("get_dataset_strategy", &ParallelContext::dataset_strategy, "Get dataset sharding strategy.")
    .def("set_stra_file_only_trainable_params", &ParallelContext::set_stra_file_only_trainable_params,
         "Set strategy ckpt only save trainable params.")
    .def("get_stra_file_only_trainable_params", &ParallelContext::stra_file_only_trainable_params,
         "Get strategy ckpt only save trainable params.")
    .def("set_enable_parallel_optimizer", &ParallelContext::set_enable_parallel_optimizer,
         "Set enable/disable parallel optimizer.")
    .def("get_enable_parallel_optimizer", &ParallelContext::enable_parallel_optimizer,
         "Get enable/disable parallel optimizer.")
    .def("set_force_fp32_communication", &ParallelContext::set_force_fp32_communication,
         "Set whether to force fp32 communication value.")
    .def("get_force_fp32_communication", &ParallelContext::force_fp32_communication,
         "Get the switch whether to force  fp32 communication value")
    .def("get_enable_fold_pipeline", &ParallelContext::enable_fold_pipeline, "Get enable/disable fold pipeline.")
    .def("set_communi_parallel_mode", &ParallelContext::set_communi_parallel_mode, "Set communication parallel mode.")
    .def("get_communi_parallel_mode", &ParallelContext::communi_parallel_mode, "Get communication parallel mode.")
    .def("set_optimizer_weight_shard_size", &ParallelContext::set_optimizer_weight_shard_size,
         "Set opt shard group size when not fully use parallel optimizer.")
    .def("get_optimizer_weight_shard_size", &ParallelContext::optimizer_weight_shard_size,
         "Get opt shard group size when not fully use parallel optimizer.")
    .def("set_optimizer_weight_shard_aggregated_save", &ParallelContext::set_optimizer_weight_shard_aggregated_save,
         "Set whether to integrated save weight shard when enable parallel optimizer.")
    .def("get_optimizer_weight_shard_aggregated_save", &ParallelContext::optimizer_weight_shard_aggregated_save,
         "Get whether to integrated save weight shard when enable parallel optimizer.")
    .def("set_enable_alltoall", &ParallelContext::set_enable_all2all, "Set the enabling AllToAll value.")
    .def("get_enable_alltoall", &ParallelContext::enable_all2all, "Get the enabling AllToAll value.")
    .def("set_sharding_propagation", &ParallelContext::set_sharding_propagation,
         "Set sharding strategy propagation value.")
    .def("get_sharding_propagation", &ParallelContext::sharding_propagation, "Get sharding strategy propagation value.")
    .def("set_ops_strategy_json_config", &ParallelContext::set_ops_strategy_json_config,
         "Set ops strategy save&load config.")
    .def("reset", &ParallelContext::Reset, "Reset auto parallel context.");
  MS_LOG(INFO) << "Start CostModelContext...";
  (void)py::class_<CostModelContext, std::shared_ptr<CostModelContext>>(m, "CostModelContext")
    .def_static("get_instance", &CostModelContext::GetInstance, "Get cost_model context instance.")
    .def("set_device_memory_capacity", &CostModelContext::set_device_memory_capacity,
         "Set the capacity of device memory.")
    .def("get_device_memory_capacity", &CostModelContext::device_memory_capacity, "Get the capacity of device memory.")
    .def("set_costmodel_alpha", &CostModelContext::set_costmodel_alpha,
         "Set the parameter cost_model_alpha of the DP algorithm.")
    .def("get_costmodel_alpha", &CostModelContext::costmodel_alpha,
         "Get the parameter cost_model_alpha of the DP algorithm.")
    .def("set_costmodel_beta", &CostModelContext::set_costmodel_beta,
         "Set the parameter cost_model_beta of the DP algorithm.")
    .def("get_costmodel_beta", &CostModelContext::costmodel_beta,
         "Get the parameter cost_model_beta of the DP algorithm.")
    .def("set_costmodel_gamma", &CostModelContext::set_costmodel_gamma,
         "Set the parameter cost_model_gamma of the DP algorithm")
    .def("get_costmodel_gamma", &CostModelContext::costmodel_gamma,
         "Get the parameter cost_model_gamma of the DP algorithm.")
    .def("set_costmodel_communi_threshold", &CostModelContext::set_costmodel_communi_threshold,
         "Set the parameter cost_model_communi_threshold of the DP algorithm.")
    .def("get_costmodel_communi_threshold", &CostModelContext::costmodel_communi_threshold,
         "Get the parameter cost_model_communi_threshold of the DP algorithm.")
    .def("set_costmodel_communi_const", &CostModelContext::set_costmodel_communi_const,
         "Set the parameter cost_model_communi_const of the DP algorithm.")
    .def("get_costmodel_communi_const", &CostModelContext::costmodel_communi_const,
         "Get the parameter cost_model_communi_const of the DP algorithm.")
    .def("set_costmodel_communi_bias", &CostModelContext::set_costmodel_communi_bias,
         "Set the parameter cost_model_communi_bias of the DP algorithm.")
    .def("get_costmodel_communi_bias", &CostModelContext::costmodel_communi_bias,
         "Get the parameter cost_model_communi_bias of the DP algorithm.")
    .def("set_multi_subgraphs", &CostModelContext::set_multi_subgraphs, "Set the parameter is_multi_subgraphs.")
    .def("get_multi_subgraphs", &CostModelContext::is_multi_subgraphs, "Get the parameter is_multi_subgraphs.")
    .def("set_run_phase", &CostModelContext::set_run_phase, "Set the flag run_phase.")
    .def("get_run_phase", &CostModelContext::run_phase, "Get the flag run_phase.")
    .def("set_costmodel_allreduce_fusion_algorithm", &CostModelContext::set_costmodel_allreduce_fusion_algorithm,
         "Set the parameter gradient AllReduce fusion algorithm.")
    .def("get_costmodel_allreduce_fusion_algorithm", &CostModelContext::costmodel_allreduce_fusion_algorithm,
         "Get the parameter gradient AllReduce fusion algorithm.")
    .def("set_costmodel_allreduce_fusion_times", &CostModelContext::set_costmodel_allreduce_fusion_times,
         "Set the parameter gradient AllReduce times.")
    .def("get_costmodel_allreduce_fusion_times", &CostModelContext::costmodel_allreduce_fusion_times,
         "Get the parameter gradient AllReduce times.")
    .def("set_costmodel_allreduce_fusion_tail_percent", &CostModelContext::set_costmodel_allreduce_fusion_tail_percent,
         "Set the parameter gradient AllReduce fusion tail percent.")
    .def("get_costmodel_allreduce_fusion_tail_percent", &CostModelContext::costmodel_allreduce_fusion_tail_percent,
         "Get the parameter gradient AllReduce fusion tail percent.")
    .def("set_costmodel_allreduce_fusion_tail_time", &CostModelContext::set_costmodel_allreduce_fusion_tail_time,
         "Set the parameter gradient AllReduce fusion tail time.")
    .def("get_costmodel_allreduce_fusion_tail_time", &CostModelContext::costmodel_allreduce_fusion_tail_time,
         "Get the parameter gradient AllReduce fusion tail time.")
    .def("set_costmodel_allreduce_fusion_allreduce_inherent_time",
         &CostModelContext::set_costmodel_allreduce_fusion_allreduce_inherent_time,
         "Set the parameter gradient AllReduce fusion allreduce inherent time.")
    .def("get_costmodel_allreduce_fusion_allreduce_inherent_time",
         &CostModelContext::costmodel_allreduce_fusion_allreduce_inherent_time,
         "Get the parameter gradient AllReduce fusion allreduce inherent time.")
    .def("set_costmodel_allreduce_fusion_allreduce_bandwidth",
         &CostModelContext::set_costmodel_allreduce_fusion_allreduce_bandwidth,
         "Set the parameter gradient AllReduce fusion allreduce bandwidth.")
    .def("get_costmodel_allreduce_fusion_allreduce_bandwidth",
         &CostModelContext::costmodel_allreduce_fusion_allreduce_bandwidth,
         "Get the parameter gradient AllReduce fusion allreduce bandwidth.")
    .def("set_costmodel_allreduce_fusion_computation_time_parameter",
         &CostModelContext::set_costmodel_allreduce_fusion_computation_time_parameter,
         "Set the parameter gradient AllReduce fusion computation time parameter.")
    .def("get_costmodel_allreduce_fusion_computation_time_parameter",
         &CostModelContext::costmodel_allreduce_fusion_computation_time_parameter,
         "Get the parameter gradient AllReduce fusion computation time parameter.")
    .def("set_tensor_slice_align_enable", &CostModelContext::set_tensor_slice_alignment_enable,
         "Set the parameter tensor_slice_align_enable in strategy generation.")
    .def("get_tensor_slice_align_enable", &CostModelContext::tensor_slice_alignment_enable,
         "Get the parameter tensor_slice_align_enable in strategy generation.")
    .def("set_tensor_slice_align_size", &CostModelContext::set_tensor_slice_alignment_size,
         "Set the parameter tensor_slice_size in strategy generation.")
    .def("get_tensor_slice_align_size", &CostModelContext::tensor_slice_alignment_size,
         "Get the parameter tensor_slice_size in strategy generation.")
    .def("set_fully_use_devices", &CostModelContext::set_fully_use_device,
         "Set the parameter fully_use_devices in the DP algorithm.")
    .def("get_fully_use_devices", &CostModelContext::fully_use_device,
         "Get the parameter fully_use_devices in the DP algorithm.")
    .def("set_elementwise_op_strategy_follow", &CostModelContext::set_elementwise_stra_follow,
         "Set the parameter elementwise_op_strategy_follow in the DP algorithm.")
    .def("get_elementwise_op_strategy_follow", &CostModelContext::elementwise_stra_follow,
         "Get the parameter elementwise_op_strategy_follow in the DP algorithm.")
    .def("set_dp_algo_enable_approxi", &CostModelContext::set_dp_algo_enable_approxi,
         "Set the flag whether enabling approximation in the DP algorithm.")
    .def("get_dp_algo_enable_approxi", &CostModelContext::dp_algo_enable_approxi,
         "Get the flag whether enabling approximation in the DP algorithm.")
    .def("set_dp_algo_approxi_epsilon", &CostModelContext::set_dp_algo_approxi_epsilon,
         "Set the epsilon which is used in the approximation of DP algorithm.")
    .def("get_dp_algo_approxi_epsilon", &CostModelContext::dp_algo_approxi_epsilon,
         "Get the epsilon which is used in the approximation of DP algorithm.")
    .def("set_rp_matmul_mem_coef", &CostModelContext::set_rp_matmul_mem_coef,
         "Set the matmul memory coef which is used in the RP algorithm.")
    .def("get_rp_matmul_mem_coef", &CostModelContext::rp_matmul_mem_coef,
         "Get the matmul memory coef which is used in the RP algorithm.")
    .def("set_dp_algo_single_loop", &CostModelContext::set_dp_algo_single_loop,
         "Set the flag of generating a single suite of OperatorInfos in for-loop.")
    .def("get_dp_algo_single_loop", &CostModelContext::dp_algo_single_loop,
         "Get the flag of whether or not generating a single suite of OperatorInfos in for-loop.")
    .def("reset_cost_model", &CostModelContext::ResetCostModel, "Reset the CostModelContext.")
    .def("reset_algo_parameters", &CostModelContext::ResetAlgoParameters, "Reset the AlgoParameters.");
  MS_LOG(INFO) << "Start OffloadContext...";
  (void)py::class_<OffloadContext, std::shared_ptr<OffloadContext>>(m, "OffloadContext")
    .def_static("get_instance", &OffloadContext::GetInstance, "Get offload context instance.")
    .def("set_offload_param", &OffloadContext::set_offload_param, "Set the param for offload destination, cpu or disk.")
    .def("offload_param", &OffloadContext::offload_param, "Get the param for offload destination.")
    .def("set_offload_path", &OffloadContext::set_offload_path, "Set the path of offload.")
    .def("offload_path", &OffloadContext::offload_path, "Get the path of offload.")
    .def("set_offload_checkpoint", &OffloadContext::set_offload_checkpoint,
         "Set the checkpoint for offload destination, cpu or disk.")
    .def("offload_checkpoint", &OffloadContext::offload_checkpoint, "Get the checkpoint for offload destination.")
    .def("set_offload_cpu_size", &OffloadContext::set_offload_cpu_size, "Set the cpu memory size for offload.")
    .def("offload_cpu_size", &OffloadContext::offload_cpu_size, "Get the cpu memory size for offload.")
    .def("set_offload_disk_size", &OffloadContext::set_offload_disk_size, "Set the disk size for offload.")
    .def("offload_disk_size", &OffloadContext::offload_disk_size, "Get the disk size for offload.")
    .def("set_enable_aio", &OffloadContext::set_enable_aio, "Set the flag of whether enabling aio.")
    .def("enable_aio", &OffloadContext::enable_aio, "Get the flag of whether enabling aio.")
    .def("set_aio_block_size", &OffloadContext::set_aio_block_size, "Set the size of aio block.")
    .def("aio_block_size", &OffloadContext::aio_block_size, "Get the size of aio block.")
    .def("set_aio_queue_depth", &OffloadContext::set_aio_queue_depth, "Set the depth of aio queue.")
    .def("aio_queue_depth", &OffloadContext::aio_queue_depth, "Get the depth of aio queue.")
    .def("set_enable_pinned_mem", &OffloadContext::set_enable_pinned_mem,
         "Set the flag of whether enabling pinned memory.")
    .def("enable_pinned_mem", &OffloadContext::enable_pinned_mem, "Get the flag of whether enabling pinned memory.")
    .def("set_auto_offload", &OffloadContext::set_auto_offload,
         "Set whether to automatically generate the offload strategy")
    .def("auto_offload", &OffloadContext::auto_offload, "Get the flag of whether auto offload")
    .def("set_host_mem_block_size", &OffloadContext::set_host_mem_block_size, "Set the block size for host memory pool")
    .def("host_mem_block_size", &OffloadContext::host_mem_block_size, "Get the block size of host memory pool")
    .def("set_cpu_ratio", &OffloadContext::set_cpu_ratio, "Set the cpu memory usage ratio for offload strategy")
    .def("cpu_ratio", &OffloadContext::cpu_ratio, "Get the cpu memory usage ratio of offload strategy")
    .def("set_hbm_ratio", &OffloadContext::set_hbm_ratio, "Set the hbm usage ratio for offload strategy")
    .def("hbm_ratio", &OffloadContext::hbm_ratio, "Get the hbm usage ratio of offload strategy");

  (void)py::module::import("atexit").attr("register")(py::cpp_function{[&]() -> void {
    MS_LOG(INFO) << "Start register...";
    mindspore::MsContext::GetInstance()->RegisterCheckEnv(nullptr);
    mindspore::MsContext::GetInstance()->RegisterSetEnv(nullptr);
#ifndef ENABLE_SECURITY
    MS_LOG(INFO) << "Start mindspore.profiler...";
    try {
      py::module profiler = py::module::import("mindspore.profiler").attr("EnvProfiler")();
      (void)profiler.attr("analyse")();
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Failed to parse profiler data." << e.what();
    }
#endif
    MS_LOG(INFO) << "Start EmbeddingCacheScheduler...";
#if defined(__linux__) && defined(WITH_BACKEND)
    mindspore::runtime::EmbeddingCacheScheduler::GetInstance().Finalize(
      !mindspore::distributed::cluster_exit_with_exception());
#endif

#ifdef ENABLE_MINDDATA
    MS_LOG(INFO) << "Start releasing dataset handles...";
    py::module iterators = py::module::import("mindspore.dataset.engine.iterators");
    (void)iterators.attr("_cleanup")();
    MS_LOG(INFO) << "End release dataset handles.";
#endif
    mindspore::pipeline::FinalizeCluster();

    // only in case that c++ calling python interface, ClearResAtexit should be called.
    if (mindspore::python_adapter::IsPythonEnv()) {
      mindspore::pipeline::ClearResAtexit();
    }
  }});

#ifndef ENABLE_SECURITY
  (void)py::class_<EventWriter, std::shared_ptr<EventWriter>>(m, "EventWriter_")
    .def(py::init<const std::string &>())
    .def("GetFileName", &EventWriter::GetFileName, "Get the file name.")
    .def("Open", &EventWriter::Open, "Open the write file.")
    .def("Write", &EventWriter::Write, "Write the serialize event.")
    .def("EventCount", &EventWriter::GetWriteEventCount, "Write event count.")
    .def("Flush", &EventWriter::Flush, "Flush the event.")
    .def("Close", &EventWriter::Close, "Close the write.")
    .def("Shut", &EventWriter::Shut, "Final close the write.");
#endif  // ENABLE_SECURITY

  (void)py::class_<OpLib, std::shared_ptr<OpLib>>(m, "Oplib")
    .def(py::init())
    .def_static("reg_op", &OpLib::RegOp, "Register op info.");

  (void)py::class_<CollectiveManager, std::shared_ptr<CollectiveManager>>(m, "CollectiveManager")
    .def_static("get_instance", &CollectiveManager::instance, "Get collective manager instance.")
    .def("initialized", &CollectiveManager::initialized, "Returns whether distributed module is initialized.")
    .def("create_group", &CollectiveManager::CreateCommunicationGroup, "Create collective group.")
    .def("destroy_group", &CollectiveManager::DestroyCommunicationGroup, "Destroy collective group.")
    .def("get_local_rank_id", &CollectiveManager::GetLocalRankId, "Get the node rank id.")
    .def("get_local_group_size", &CollectiveManager::GetLocalGroupSize, "Get the node rank id.")
    .def("get_world_rank_from_group_rank", &CollectiveManager::GetWorldRankFromGroupRank,
         "Get world rank by group rank.")
    .def("get_group_rank_from_world_rank", &CollectiveManager::GetGroupRankFromWorldRank,
         "Get group rank by world rank.")
    .def("get_rank_id", &CollectiveManager::GetRankId, "Get the node rank id.")
    .def("get_group_size", &CollectiveManager::GetGroupSize, "Get the nodes number in the collective communication.")
    .def("get_group_ranks", &CollectiveManager::GetGroupRanks,
         "Get group ranks for the specified communication group.");

  (void)py::class_<PSContext, std::shared_ptr<PSContext>>(m, "PSContext")
    .def_static("get_instance", &PSContext::instance, "Get PS context instance.")
    .def("set_ps_enable", &PSContext::SetPSEnable, "Set PS mode enabled or disabled.")
    .def("is_ps_mode", &PSContext::is_ps_mode, "Get PS mode enable-disable status.")
    .def("reset", &PSContext::Reset, "Reset PS context attributes.")
    .def("is_worker", &PSContext::is_worker, "Get whether the role of this process is Worker.")
    .def("is_server", &PSContext::is_server, "Get whether the role of this process is PServer.")
    .def("is_scheduler", &PSContext::is_scheduler, "Get whether the role of this process is Scheduler.")
    .def("ps_rank_id", &PSContext::ps_rank_id, "Get Worker and PServer rank id.")
    .def("insert_hash_table_size", &PSContext::InsertHashTableSize, "Insert hash table size.")
    .def("reinsert_hash_table_size", &PSContext::ReInsertHashTableSize,
         "Insert hash table size with new parameter name.")
    .def("insert_accumu_init_info", &PSContext::InsertAccumuInitInfo, "Insert accumulation initialization value.")
    .def("clone_hash_table", &PSContext::CloneHashTable, "Clone a hash table.")
    .def("set_cache_enable", &PSContext::set_cache_enable, "Set ps mode cache enable or not.")
    .def("set_cache_size", &PSContext::set_cache_size, "Set embedding cache size for ps cache mode.")
    .def("cache_enable", &PSContext::cache_enable, "Get ps mode cache enable or not.")
    .def("set_sparse_format", &PSContext::set_sparse_format, "Set the storage format of the embedding table.")
    .def("set_rank_id", &PSContext::set_rank_id, "Set rank id for worker on ps mode.")
    .def("set_server_mode", &PSContext::set_server_mode, "Set server mode.")
    .def("server_mode", &PSContext::server_mode, "Get server mode.")
    .def("set_ms_role", &PSContext::set_ms_role, "Set role for this process.")
    .def("ms_role", &PSContext::ms_role, "Get role for this process.")
    .def("set_worker_num", &PSContext::set_worker_num, "Set worker number.")
    .def("worker_num", &PSContext::worker_num, "Get worker number.")
    .def("set_server_num", &PSContext::set_server_num, "Set server number.")
    .def("server_num", &PSContext::server_num, "Get server number.")
    .def("set_scheduler_ip", &PSContext::set_scheduler_ip, "Set scheduler ip.")
    .def("scheduler_ip", &PSContext::scheduler_ip, "Get scheduler ip.")
    .def("set_scheduler_port", &PSContext::set_scheduler_port, "Set scheduler port.")
    .def("scheduler_port", &PSContext::scheduler_port, "Get scheduler port.")
    .def("set_scheduler_manage_port", &PSContext::set_scheduler_manage_port,
         "Set scheduler manage port used to scale out/in.")
    .def("scheduler_manage_port", &PSContext::scheduler_manage_port, "Get scheduler manage port used to scale out/in.")
    .def("set_enable_ssl", &PSContext::set_enable_ssl, "Set PS SSL mode enabled or disabled.")
    .def("enable_ssl", &PSContext::enable_ssl, "Get PS SSL mode enabled or disabled.")
    .def("set_client_password", &PSContext::set_client_password, "Set the client password to decode the p12 file.")
    .def("client_password", &PSContext::client_password, "Get the client password to decode the p12 file.")
    .def("set_server_password", &PSContext::set_server_password, "Set the server password to decode the p12 file.")
    .def("server_password", &PSContext::server_password, "Get the server password to decode the p12 file.")
    .def("set_config_file_path", &PSContext::set_config_file_path,
         "Set configuration files required by the communication layer.")
    .def("config_file_path", &PSContext::config_file_path,
         "Get configuration files required by the communication layer.")
    .def("enable_distributed_mindrt", &PSContext::enable_distributed_mindrt, "Whether distributed MindRT is enabled.")
    .def("set_checkpoint_load_status", &PSContext::set_checkpoint_load_status, "Set checkpoint load status.")
    .def("store_warm_up_ptr_by_tensor", &PSContext::StoreWarmUpPtrByTensor, "Store warm up host cache by tensor.")
    .def("store_warm_up_ptr_by_tensor_list", &PSContext::StoreWarmUpPtrByTensorList,
         "Store warm up host cache by tensor list");
  (void)m.def("_encrypt", &mindspore::pipeline::PyEncrypt, "Encrypt the data.");
  (void)m.def("_decrypt", &mindspore::pipeline::PyDecrypt, "Decrypt the data.");
  (void)m.def("_decrypt_data", &mindspore::pipeline::PyDecryptData, "Decrypt the bytes data.");
  (void)m.def("_is_cipher_file", &mindspore::pipeline::PyIsCipherFile, "Determine whether the file is encrypted");

  (void)py::class_<RecoveryContext, std::shared_ptr<RecoveryContext>>(m, "RecoveryContext")
    .def_static("get_instance", &RecoveryContext::GetInstance, "Get recovery context instance.")
    .def("enable_recovery", &RecoveryContext::enable_recovery, "Get whether enable recovery.")
    .def("latest_ckpt_file", &RecoveryContext::latest_ckpt_file, "Get latest checkpoint file path.")
    .def("latest_ckpt_epoch", &RecoveryContext::latest_ckpt_epoch, "Get the epoch of latest checkpoint.")
    .def("latest_ckpt_step", &RecoveryContext::latest_ckpt_step, "Get the step of latest checkpoint.")
    .def("set_need_reset", &RecoveryContext::set_need_reset,
         "Set whether should call reset minddata and load ckpt for disaster recovery.")
    .def("need_reset", &RecoveryContext::need_reset,
         "Get whether should call reset minddata and load ckpt for disaster recovery.")
    .def("recovery_path", &RecoveryContext::recovery_path,
         "Get the recovery path used to save that need to be persisted.")
    .def("ckpt_path", &RecoveryContext::GetCkptPath, "Get the recovery path used to save checkpoint.")
    .def("set_ckpt_path", &RecoveryContext::SetCkptPath, "Set the recovery path used to save checkpoint.");

  (void)py::class_<DeviceContextManager, std::shared_ptr<DeviceContextManager>>(m, "DeviceContextManager")
    .def_static("get_instance", &DeviceContextManager::GetInstance, py::return_value_policy::reference,
                "Get device context manager instance.")
    .def("get_device_context", &DeviceContextManager::GetDeviceContext, "Return device context object.");
  (void)py::class_<DeviceContext, std::shared_ptr<DeviceContext>>(m, "DeviceContext")
    .def("initialized", &DeviceContext::initialized, "Return whether this device backend is successfully initialized.");
  DeviceContextManager::GetInstance().RegisterDeviceStatelessFunc(&m);

  (void)m.def("_ms_memory_recycle", &mindspore::pipeline::MemoryRecycle, "Recycle memory used by mindspore.");
  (void)m.def("_bind_device_ctx", &mindspore::pipeline::BindDeviceCtx, "Bind device context to current thread");
  (void)m.def("swap_cache", &mindspore::pipeline::SwapCache, py::arg("host"), py::arg("device"),
              py::arg("block_mapping"), py::arg("is_device_to_host"), "Swap Cache for PageAttention.");
}
