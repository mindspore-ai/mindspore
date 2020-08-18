/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/oplib/oplib.h"
#include "backend/kernel_compiler/oplib/oploader.h"
#include "pipeline/jit/pipeline.h"
#include "frontend/operator/composite/composite.h"
#include "pipeline/pynative/pynative_execute.h"
#include "utils/symbolic.h"
#include "pybind_api/api_register.h"
#include "pipeline/jit/parse/python_adapter.h"
#include "utils/summary/event_writer.h"
#include "utils/config_manager.h"
#include "utils/mpi/mpi_config.h"
#include "frontend/parallel/context.h"
#include "frontend/parallel/costmodel_context.h"
#ifdef ENABLE_GPU_COLLECTIVE
#include "runtime/device/gpu/distribution/collective_init.h"
#else
#include "runtime/device/gpu/distribution/collective_fake_init.h"
#endif
namespace py = pybind11;

using EnvInstance = mindspore::EnvInstance;
using ExecutorPy = mindspore::pipeline::ExecutorPy;
using Pipeline = mindspore::pipeline::Pipeline;
using PrimitivePy = mindspore::PrimitivePy;
using MetaFuncGraph = mindspore::MetaFuncGraph;
using EventWriter = mindspore::summary::EventWriter;
using OpLib = mindspore::kernel::OpLib;
using OpInfoLoaderPy = mindspore::kernel::OpInfoLoaderPy;
using ParallelContext = mindspore::parallel::ParallelContext;
using CostModelContext = mindspore::parallel::CostModelContext;

// Interface with python
PYBIND11_MODULE(_c_expression, m) {
  m.doc() = "MindSpore c plugin";

  auto fns = mindspore::PybindDefineRegister::AllFuncs();
  for (auto &item : fns) {
    item.second(&m);
  }

  // Class Pipeline interface
  (void)py::class_<ExecutorPy, std::shared_ptr<ExecutorPy>>(m, "Executor_")
    .def_static("get_instance", &ExecutorPy::GetInstance, "Executor get_instance.")
    .def("__call__", &ExecutorPy::Run, py::arg("args"), py::arg("phase") = py::str(""), "Executor run function.")
    .def("del_net_res", &ExecutorPy::DelNetRes, py::arg("network_id") = py::str(""), "Delete network resource.")
    .def("get_func_graph", &ExecutorPy::GetFuncGraph, py::arg("phase") = py::str(""), "Get graph pointer.")
    .def("get_func_graph_proto", &ExecutorPy::GetFuncGraphProto, py::arg("phase") = py::str(""),
         py::arg("type") = py::str("onnx_ir"), "Get graph proto string by specifying ir type.")
    .def("compile", &ExecutorPy::Compile, py::arg("obj"), py::arg("args"), py::arg("phase") = py::str(""),
         py::arg("use_vm") = py::bool_(false), "Compile obj by executor.")
    .def("updata_param_node_default_input", &ExecutorPy::UpdataParamNodeDefaultInput, py::arg("phase"),
         py::arg("params"), "Fetch the inputs of Conv or Matmul for quant export.")
    .def("get_parameter_layout", &ExecutorPy::GetParameterLayout, py::arg("phase") = py::str("train"),
         "Get Parameter Tensor Layout Dictionary.")
    .def("get_strategy", &ExecutorPy::GetCNodeStrategy, py::arg("phase") = py::str("train"),
         "Get CNode Strategy Dictionary.")
    .def("get_allreduce_fusion", &ExecutorPy::GetAllreduceFusion, py::arg("phase") = py::str("train"),
         "Get Allreduce Fusion Dictionary.")
    .def("fetch_info_for_quant_export", &ExecutorPy::FetchInfoForQuantExport, py::arg("phase") = py::str("train"),
         "Fetch the inputs of Conv or Matmul for quant export.")
    .def("build_data_graph", &ExecutorPy::BuildGraph, py::arg("build_params"), py::arg("phase") = py::str("train"),
         py::arg("broadcast_params") = py::dict(), "Build data graph.")
    .def("has_compiled", &ExecutorPy::HasCompiled, py::arg("phase") = py::str(""), "get if cell compiled.")
    .def("run_init_graph", &ExecutorPy::RunInitGraph, "Run init Graph.");

  (void)py::class_<EnvInstance, std::shared_ptr<EnvInstance>>(m, "EnvInstance_").def(py::init());

  (void)m.def("generate_key", &mindspore::pipeline::GenerateKey, "Generate the function graph key.");
  (void)m.def("real_run_op", &mindspore::pynative::RunOp, "Run op pynatively.");
  (void)m.def("reset_op_id", &mindspore::pipeline::ResetOpId, "Reset Operator Id");
  (void)m.def("init_hccl", &mindspore::pipeline::InitHccl, "Init Hccl");
  (void)m.def("finalize_hccl", &mindspore::pipeline::FinalizeHccl, "Finalize Hccl");
  (void)m.def("verify_inputs_signature", &mindspore::pipeline::VerifyInputSignature, "Verify input signature.");
  (void)m.def("init_exec_dataset", &mindspore::pipeline::InitExecDataset, py::arg("queue_name"), py::arg("size"),
              py::arg("batch_size"), py::arg("types"), py::arg("shapes"), py::arg("input_indexs"),
              py::arg("phase") = py::str("dataset"), py::arg("need_run") = py::bool_(true), "Init and exec dataset.");
  (void)m.def("random_normal", &mindspore::pipeline::InitRandomNormal, py::arg("mean"), py::arg("stddev"),
              py::arg("outshape"), py::arg("seed"), py::arg("outputtensor"), "InitRandRandom");
  (void)m.def("_set_dataset_mode_config", &mindspore::ConfigManager::SetDatasetModeConfig, "API for set dataset mode.");
  (void)m.def("init_backend", &mindspore::pipeline::InitBackend, "Init Backend.");

  (void)m.def("export_graph", &mindspore::pipeline::ExportGraph, "Export Graph.");

  (void)py::class_<mindspore::MsContext, std::shared_ptr<mindspore::MsContext>>(m, "MSContext")
    .def_static("get_instance", &mindspore::MsContext::GetInstance, "Get ms context instance.")
    .def("get_backend_policy", &mindspore::MsContext::backend_policy, "Get backend policy.")
    .def("set_backend_policy", &mindspore::MsContext::set_backend_policy, "Set backend policy.")
    .def("get_execution_mode", &mindspore::MsContext::execution_mode, "Get execution mode.")
    .def("set_execution_mode", &mindspore::MsContext::set_execution_mode, "Set execution mode.")
    .def("set_precompile_only", &mindspore::MsContext::set_precompile_only, "Set enable precompile only.")
    .def("get_precompile_only", &mindspore::MsContext::precompile_only, "Get enable precompile only.")
    .def("get_device_target", &mindspore::MsContext::device_target, "Get device target.")
    .def("set_device_target", &mindspore::MsContext::set_device_target, "Set device target.")
    .def("get_device_id", &mindspore::MsContext::device_id, "Get device id.")
    .def("set_device_id", &mindspore::MsContext::set_device_id, "Set device id.")
    .def("get_save_graphs_flag", &mindspore::MsContext::save_graphs_flag, "Get whether to save graphs.")
    .def("set_save_graphs_flag", &mindspore::MsContext::set_save_graphs_flag, "Set whether to save graphs.")
    .def("get_auto_mixed_precision_flag", &mindspore::MsContext::auto_mixed_precision_flag,
         "Get whether to enable auto mixed precision.")
    .def("set_auto_mixed_precision_flag", &mindspore::MsContext::set_auto_mixed_precision_flag,
         "Set whether to enable auto mixed precision.")
    .def("get_enable_reduce_precision_flag", &mindspore::MsContext::enable_reduce_precision,
         "Get whether to enable reduce precision.")
    .def("set_enable_reduce_precision_flag", &mindspore::MsContext::set_enable_reduce_precision,
         "Set whether to enable reduce precision.")
    .def("get_save_graphs_path", &mindspore::MsContext::save_graphs_path, "Get save graphs path.")
    .def("set_save_graphs_path", &mindspore::MsContext::set_save_graphs_path, "Set save graphs path.")
    .def("get_enable_dump", &mindspore::MsContext::enable_dump, "Get whether to enable dump.")
    .def("set_enable_dump", &mindspore::MsContext::set_enable_dump, "Set whether to enable dump.")
    .def("get_save_dump_path", &mindspore::MsContext::save_dump_path, "Get path to dump.")
    .def("set_save_dump_path", &mindspore::MsContext::set_save_dump_path, "Set path to dump.")
    .def("set_graph_memory_max_size", &mindspore::MsContext::set_graph_memory_max_size, "set graph memory max size.")
    .def("set_variable_memory_max_size", &mindspore::MsContext::set_variable_memory_max_size,
         "set variable memory max size")
    .def("get_enable_profiling", &mindspore::MsContext::enable_profiling, "Get whether to open profiling.")
    .def("set_enable_profiling", &mindspore::MsContext::set_enable_profiling, "Set whether to open profiling.")
    .def("get_profiling_options", &mindspore::MsContext::profiling_options, "Get options to profiling.")
    .def("set_profiling_options", &mindspore::MsContext::set_profiling_options, "Set options to profiling.")
    .def("get_check_bprop_flag", &mindspore::MsContext::check_bprop_flag, "Get whether to check bprop.")
    .def("set_check_bprop_flag", &mindspore::MsContext::set_check_bprop_flag, "Set whether to check bprop.")
    .def("get_max_device_memory", &mindspore::MsContext::max_device_memory, "Get deivce memory max size.")
    .def("set_max_device_memory", &mindspore::MsContext::set_max_device_memory, "Set deivce memory max size.")
    .def("set_print_file_path", &mindspore::MsContext::set_print_file_path, "Set path to print.")
    .def("set_enable_graph_kernel", &mindspore::MsContext::set_enable_graph_kernel,
         "Set the GraphKernel switch to on or off.")
    .def("get_enable_graph_kernel", &mindspore::MsContext::enable_graph_kernel, "Get the value of GraphKernel switch.")
    .def("get_enable_sparse", &mindspore::MsContext::enable_sparse, "Get whether to enable sparsity.")
    .def("set_enable_sparse", &mindspore::MsContext::set_enable_sparse, "Set whether to enable sparsity.");

  (void)py::class_<mindspore::MpiConfig, std::shared_ptr<mindspore::MpiConfig>>(m, "MpiConfig")
    .def_static("get_instance", &mindspore::MpiConfig::GetInstance, "Get mpi config instance.")
    .def("get_enable_mpi", &mindspore::MpiConfig::enable_mpi, "Get whether enable mpi.")
    .def("set_enable_mpi", &mindspore::MpiConfig::set_enable_mpi, "Set whether to enable mpi.");

  (void)py::class_<ParallelContext, std::shared_ptr<ParallelContext>>(m, "AutoParallelContext")
    .def_static("get_instance", &ParallelContext::GetInstance, "Get auto parallel context instance.")
    .def("get_device_num", &ParallelContext::device_num, "Get device num.")
    .def("set_device_num", &ParallelContext::set_device_num, "Set device num.")
    .def("get_device_num_is_set", &ParallelContext::device_num_is_set, "Get device num is set.")
    .def("get_global_rank", &ParallelContext::global_rank, "Get global rank.")
    .def("set_global_rank", &ParallelContext::set_global_rank, "Set global rank.")
    .def("get_global_rank_is_set", &ParallelContext::global_rank_is_set, "Get global rank is set.")
    .def("get_mirror_mean", &ParallelContext::mirror_mean, "Get mirror mean.")
    .def("set_mirror_mean", &ParallelContext::set_mirror_mean, "Set mirror mean.")
    .def("get_cast_before_mirror", &ParallelContext::cast_before_mirror, "Get cast before mirror.")
    .def("set_cast_before_mirror", &ParallelContext::set_cast_before_mirror, "Set cast before mirror.")
    .def("get_loss_repeated_mean", &ParallelContext::loss_repeated_mean, "Get loss repeated mean.")
    .def("set_loss_repeated_mean", &ParallelContext::set_loss_repeated_mean, "Set loss repeated mean.")
    .def("get_communication_backend", &ParallelContext::communication_backend, "Get communication backend.")
    .def("set_communication_backend", &ParallelContext::set_communication_backend, "Set communication backend.")
    .def("get_parallel_mode", &ParallelContext::parallel_mode, "Get parallel mode.")
    .def("set_parallel_mode", &ParallelContext::set_parallel_mode, "Set parallel mode.")
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
    .def("set_full_batch", &ParallelContext::set_full_batch, "Set whether load full batch on each device.")
    .def("get_full_batch", &ParallelContext::full_batch, "Get whether load full batch on each device.")
    .def("set_enable_parallel_optimizer", &ParallelContext::set_enable_parallel_optimizer,
         "Set enable/disable parallel optimizer.")
    .def("get_enable_parallel_optimizer", &ParallelContext::enable_parallel_optimizer,
         "Get enable/disable parallel optimizer.")
    .def("reset", &ParallelContext::Reset, "Reset auto parallel context.");

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
    .def("reset_cost_model", &CostModelContext::ResetCostModel, "Reset the CostModelContext.")
    .def("reset_algo_parameters", &CostModelContext::ResetAlgoParameters, "Reset the AlgoParameters.");

  (void)py::module::import("atexit").attr("register")(py::cpp_function{[&]() -> void {
    // only in case that c++ calling python interface, ClearResAtexit should be called.
    if (mindspore::parse::python_adapter::IsPythonEnv()) {
      mindspore::pipeline::ClearResAtexit();

#ifdef ENABLE_MINDDATA
      py::module iterators = py::module::import("mindspore.dataset.engine.iterators");
      (void)iterators.attr("_cleanup")();
#endif
    }
  }});

  (void)py::class_<EventWriter, std::shared_ptr<EventWriter>>(m, "EventWriter_")
    .def(py::init<const std::string &>())
    .def("GetFileName", &EventWriter::GetFileName, "Get the file name.")
    .def("Open", &EventWriter::Open, "Open the write file.")
    .def("Write", &EventWriter::Write, "Write the serialize event.")
    .def("EventCount", &EventWriter::GetWriteEventCount, "Write event count.")
    .def("Flush", &EventWriter::Flush, "Flush the event.")
    .def("Close", &EventWriter::Close, "Close the write.")
    .def("Shut", &EventWriter::Shut, "Final close the write.");

  (void)py::class_<OpLib, std::shared_ptr<OpLib>>(m, "Oplib")
    .def(py::init())
    .def_static("reg_op", &OpLib::RegOp, "Register op info.");
#ifdef ENABLE_GPU_COLLECTIVE
  (void)m.def("init_gpu_collective", &mindspore::device::gpu::CollectiveInitializer::InitCollective,
              "Init gpu collective communication mode.");
  (void)m.def("finalize_gpu_collective", &mindspore::device::gpu::CollectiveInitializer::FinalizeCollective,
              "Finalize gpu collective communication mode.");
#else
  (void)m.def("init_gpu_collective", &mindspore::device::gpu::CollectiveFakeInitializer::InitCollective,
              "Init gpu collective communication mode.");
  (void)m.def("finalize_gpu_collective", &mindspore::device::gpu::CollectiveFakeInitializer::FinalizeCollective,
              "Finalize gpu collective communication mode.");

#endif

  (void)py::class_<OpInfoLoaderPy, std::shared_ptr<OpInfoLoaderPy>>(m, "OpInfoLoaderPy")
    .def(py::init())
    .def("get_all_ops_info", &OpInfoLoaderPy::GetAllOpsInfo, "get all ops info.");
}
