/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "pybind_api/frontend/frontend_api.h"
#include <memory>
#include "pybind11/pybind11.h"
#include "include/frontend/jit/ps/pipeline_interface.h"
#include "frontend/jit/ps/pipeline.h"
#include "include/frontend/jit/ps/executor/graph_executor_py.h"
#include "include/frontend/jit/ps/executor/jit_executor_py.h"
#include "frontend/operator/composite/composite.h"
#include "frontend/operator/composite/functional_overload.h"
#include "frontend/jit/pi/external.h"
#include "include/frontend/jit/trace/trace_recorder_interface.h"
#include "include/frontend/pybind_api.h"
#include "pybind_api/frontend/manager.h"

namespace py = pybind11;
using GraphExecutorPy = mindspore::pipeline::GraphExecutorPy;
using JitExecutorPy = mindspore::pipeline::JitExecutorPy;

namespace mindspore {
void RegGraphExecutor(py::module *m) {
  // Class Pipeline interface
  MS_LOG(INFO) << "Start GraphExecutorPy...";
  (void)py::class_<GraphExecutorPy, std::shared_ptr<GraphExecutorPy>>(*m, "GraphExecutor_")
    .def_static("get_instance", &GraphExecutorPy::GetInstance, "Executor get_instance.")
    .def("__call__", &GraphExecutorPy::Run, py::arg("args"), py::arg("phase") = py::str(""), "Executor run function.")
    .def("del_net_res", &GraphExecutorPy::DelNetRes, py::arg("obj"), py::arg("network_id") = py::set(),
         "Delete network resource.")
    .def("get_func_graph", &GraphExecutorPy::GetFuncGraph, py::arg("phase") = py::str(""), "Get graph pointer.")
    .def("check_func_graph_sequence_parameter", &GraphExecutorPy::CheckFuncGraphSequenceParamAbstract,
         py::arg("phase") = py::str(""), "Check the abstract of graph sequence parameter.")
    .def("get_func_graph_proto", &GraphExecutorPy::GetFuncGraphProto, py::arg("phase") = py::str(""),
         py::arg("type") = py::str("onnx_ir"), py::arg("incremental") = py::bool_(false),
         "Get graph proto string by specifying ir type.")
    .def("get_onnx_func_graph_proto", &GraphExecutorPy::GetOnnxFuncGraphProto, py::arg("phase") = py::str(""),
         py::arg("input_names") = py::list(), py::arg("output_names") = py::list(),
         py::arg("opset_version") = py::int_(11), py::arg("export_params") = py::bool_(true),
         py::arg("keep_initializers_as_inputs") = py::bool_(false), py::arg("dynamic_axes") = py::dict(),
         py::arg("extra_save_params") = py::bool_(false), py::arg("save_file_dir") = py::str(""),
         "Get graph proto string by onnx.")
    .def("get_params", &GraphExecutorPy::GetParams, py::arg("phase") = py::str(""), "Get Parameters from graph")
    .def("get_random_status", &GraphExecutorPy::GetRandomStatus, py::arg("phase") = py::str(""),
         "Get random status from graph")
    .def("compile", &GraphExecutorPy::Compile, py::arg("obj"), py::arg("args"), py::arg("kwargs"),
         py::arg("phase") = py::str(""), py::arg("jit_config") = py::dict(), "Compile obj by executor.")
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
    .def("set_real_args", &GraphExecutorPy::SetRealArguments, py::arg("args"), py::arg("kwargs"), "Set run arguments.")
    .def("get_optimize_graph_proto", &GraphExecutorPy::GetOptimizeGraphProto, py::arg("phase") = py::str(""),
         "Get the optimize graph proto string.")
    .def("set_jit_config", &GraphExecutorPy::SetJitConfig, py::arg("jit_config") = py::dict(), "Set the jit config.")
    .def("generate_arguments_key", &GraphExecutorPy::GenerateArgumentsKey, "Generate unique key of argument.")
    .def("check_argument_consistency", &GraphExecutorPy::CheckArgumentsConsistency, "Check equal of arguments.")
    .def("clear_compile_arguments_resource", &GraphExecutorPy::ClearCompileArgumentsResource,
         "Clear resource when phase cached.")
    .def("set_optimize_config", &GraphExecutorPy::SetOptimizeConfig, py::arg("opt_config") = py::list(),
         "Set the optimizer config.")
    .def("get_optimize_config", &GraphExecutorPy::GetOptimizeConfig, "Get the optimize config.")
    .def("set_config_passes", &GraphExecutorPy::SetConfigPasses, py::arg("passes") = py::list(),
         "Set the optimizer passes.")
    .def("get_running_passes", &GraphExecutorPy::GetRunningPasses, "Get the running passes.")
    .def("set_max_call_depth", &GraphExecutorPy::set_max_call_depth, py::arg("max_call_depth") = py::int_(1000),
         "Get the running passes.");
}

void RegJitExecutor(py::module *m) {
  MS_LOG(INFO) << "Start JitExecutorPy...";
  (void)py::class_<JitExecutorPy, std::shared_ptr<JitExecutorPy>>(*m, "JitExecutor_")
    .def_static("get_instance", &JitExecutorPy::GetInstance, "Executor get_instance.")
    .def("__call__", &JitExecutorPy::Run, py::arg("args"), py::arg("phase") = py::str(""), "Executor run function.")
    .def("del_net_res", &JitExecutorPy::DelNetRes, py::arg("obj"), py::arg("network_id") = py::set(),
         "Delete network resource.")
    .def("get_func_graph", &JitExecutorPy::GetFuncGraph, py::arg("phase") = py::str(""), "Get graph pointer.")
    .def("check_func_graph_sequence_parameter", &JitExecutorPy::CheckFuncGraphSequenceParamAbstract,
         py::arg("phase") = py::str(""), "Check the abstract of graph sequence parameter.")
    .def("split_graph", &JitExecutorPy::SplitGraph, "Split graph.")
    .def("compile", &JitExecutorPy::Compile, py::arg("obj"), py::arg("args"), py::arg("kwargs"),
         py::arg("phase") = py::str(""), py::arg("jit_config") = py::dict(), "Compile obj by executor.")
    .def("has_compiled", &JitExecutorPy::HasCompiled, py::arg("phase") = py::str(""),
         "Get if the cell or function has been compiled.")
    .def("set_enable_tuple_broaden", &JitExecutorPy::set_enable_tuple_broaden,
         py::arg("enable_tuple_broaden") = py::bool_(false), "Set tuple broaden enable.")
    .def("generate_arguments_key", &JitExecutorPy::GenerateArgumentsKey, "Generate unique key of argument.")
    .def("clear_compile_arguments_resource", &JitExecutorPy::ClearCompileArgumentsResource,
         "Clear resource when phase cached.")
    .def("get_params", &JitExecutorPy::GetParams, py::arg("phase") = py::str(""), "Get Parameters from graph")
    .def("set_jit_config", &JitExecutorPy::SetJitConfig, py::arg("jit_config") = py::dict(), "Set the jit config.")
    .def("get_jit_config", &JitExecutorPy::GetJitConfig, "Get the jit config.")
    .def("set_queue_name", &JitExecutorPy::set_queue_name, py::arg("queue_name") = py::str(""),
         "Set queue name for the graph loaded from compile cache.")
    .def("get_queue_name", &JitExecutorPy::get_queue_name,
         "Get cached queue name for the graph loaded from compile cache.")
    .def("set_compile_cache_dep_files", &JitExecutorPy::set_compile_cache_dep_files,
         py::arg("compile_cache_dep_files") = py::list(), "Set the compilation cache dependent files.")
    .def("set_weights_values", &JitExecutorPy::set_weights_values, py::arg("weights") = py::dict(),
         "Set values of weights.")
    .def("set_real_args", &JitExecutorPy::SetRealArguments, py::arg("args"), py::arg("kwargs"), "Set run args.")
    .def("check_argument_consistency", &JitExecutorPy::CheckArgumentsConsistency, "Check equal of arguments.")
    .def("get_func_graph_proto", &JitExecutorPy::GetFuncGraphProto, py::arg("phase") = py::str(""),
         py::arg("type") = py::str("onnx_ir"), py::arg("incremental") = py::bool_(false),
         "Get graph proto string by specifying ir type.");
}

void RegFunctional(const py::module *m) {
  (void)py::class_<Functional, std::shared_ptr<Functional>>(*m, "Functional_")
    .def(py::init<py::str &>())
    .def_property_readonly("name", &Functional::name, "Get functional name.");
}

void RegFrontendModule(py::module *m) {
  RegTyping(m);
  RegCNode(m);
  RegCell(m);
  RegMetaFuncGraph(m);
  RegFuncGraph(m);
  mindspore::trace::RegTraceRecorderPy(m);
  RegUpdateFuncGraphHyperParams(m);

  RegParamInfo(m);
  RegPrimitive(m);
  RegPrimitiveFunction(m);
  RegFunctional(m);
  RegSignatureEnumRW(m);
  RegPreJit(m);
  RegValues(m);
  mindspore::pijit::RegPIJitInterface(m);
  mindspore::prim::RegCompositeOpsGroup(m);
#ifdef _MSC_VER
  mindspore::abstract::RegPrimitiveFrontEval();
#endif

  RegGraphExecutor(m);
  RegJitExecutor(m);
  (void)m->def("_run_jit_pipeline", &mindspore::pipeline::RunJitPipeline, "Whether to run the jit pipeline.");
  (void)m->def("dump_func_graph", &mindspore::pipeline::DumpFuncGraph, py::arg("func_graph"),
               "Get the dump string of a func_graph");
  (void)m->def("verify_inputs_signature", &mindspore::pipeline::VerifyInputSignature, "Verify input signature.");
  (void)m->def("init_pipeline", &mindspore::pipeline::InitPipeline, "Init Pipeline.");
  (void)m->def("load_mindir", &mindspore::pipeline::LoadMindIR, py::arg("file_name"), py::arg("dec_key") = nullptr,
               py::arg("key_len") = py::int_(0), py::arg("dec_mode") = py::str("AES-GCM"),
               py::arg("decrypt") = py::none(), "Load model as Graph.");
  (void)m->def("split_mindir", &mindspore::pipeline::SplitMindIR, py::arg("file_name"),
               "Split single mindir to distributed mindir");
  (void)m->def("split_dynamic_mindir", &mindspore::pipeline::SplitDynamicMindIR, py::arg("file_name"),
               py::arg("device_num") = py::int_(8), py::arg("rank_id") = py::int_(0), py::arg("sapp") = py::bool_(true),
               "Split single mindir to distributed mindir");
  RegMemoryRecycle(m);
}
}  // namespace mindspore
