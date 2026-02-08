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

#include "pybind_api/frontend/manager.h"

#include <memory>
#include <map>

#include "include/frontend/jit/ps/pass_interface.h"
#include "include/frontend/jit/ps/resource_interface.h"
#include "include/frontend/jit/ps/executor/executor_py.h"
#include "include/frontend/jit/ps/executor/graph_executor_py.h"
#include "include/frontend/optimizer/ad/grad_interface.h"
#include "include/frontend/jit/ps/executor/jit_executor_py.h"
#include "include/frontend/jit/ps/parse/py_data_convert.h"
#include "include/frontend/jit/ps/debug/trace_interface.h"
#include "include/frontend/parallel/graph_costmodel.h"
#include "include/pynative/utils/pynative_execute.h"
#include "include/pynative/forward/pyboost/converter.h"
#include "utils/config_manager.h"
#include "utils/convert_utils.h"
#include "include/utils/callback.h"
#include "include/backend/common/exec_order/execute_order_tracker.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "utils/distributed_meta.h"
#include "utils/log_adapter.h"
#include "utils/info.h"
#include "utils/llm_manager.h"
#include "include/utils/comm_manager.h"
#include "include/backend/common/ms_device_shape_transfer.h"
#include "utils/interpret_node_recorder.h"
#include "mindspore/ccsrc/utils/ir_dump/dump_proto.h"
#include "include/backend/common/kernel_graph/session_factory.h"
#include "include/backend/backend_manager/backend_manager.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/backend/common/exec_order/kernel_cache.h"
#include "include/pynative/utils/runtime/op_executor.h"
#include "backend/ms_backend/runtime/graph_executor/pipeline/runtime_pipeline.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/graph_scheduler.h"
#include "tools/profiler/profiler.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "tools/profiler/profiling.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/graph_kernel_builder_manager.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_info.h"
#include "include/runtime/hardware_abstract/data_queue/data_queue_mgr.h"

#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/cluster/init.h"
#endif
#ifdef ENABLE_DUMP_IR
#include "ir/cell.h"
#endif

#include "mindspore/ccsrc/pynative/forward/pyboost/auto_generate/tensor_func_utils.h"
#include "include/backend/common/somas/somas.h"
#include "include/utils/pyobj_manager.h"

#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace mindspore {
void RegMemoryRecycle(py::module *m) {
  (void)m->def("_ms_memory_recycle", &mindspore::MemoryRecycle, "Recycle memory used by mindspore.");
}
void RecordExitStatus() { MS_LOG(INFO) << "Status record: system exit."; }

void MemoryRecycle() {
  pipeline::ReclaimOptimizer();
  ad::ClearKPrim();
  ad::ClearPrimBpropOptimizer();
  pipeline::ClearAnalysisResultCacheMgr();
  abstract::AnalysisContext::ClearContext();
  pipeline::CleanCache();
  // clean static variable to prevent from crash. As static variable is released after
  // Python threads is released.
  parse::data_converter::ClearObjectCache();
  pipeline::CleanParserResource();
  trace::ClearTraceStack();
  pynative::PyNativeExecutor::GetInstance()->ClearRes();
  ConfigManager::GetInstance().ResetConfig();
  ScopeManager::GetInstance().ClearScope();
  FuncGraphLoopBreaker::Inst().CleanMetaFuncGraphs();
  FuncGraphLoopBreaker::Inst().BreakLoop();
}

namespace {
void MemTrackerInstanceClear() {
  size_t rank_id = SIZE_MAX;
  if (DistributedMeta::GetInstance()->initialized()) {
    rank_id = DistributedMeta::GetInstance()->global_rank_id();
  }
  device::tracker::MemTrackerManager::GetInstance().Dump(rank_id);
}
}  // namespace

void ClearResPart1() {
  pynative::PyNativeExecutor::GetInstance()->WorkerJoin();
  runtime::OpExecutor::GetInstance().WorkerJoin();
  runtime::RuntimePipeline::GetInstance().WorkerJoin();
  device::DeviceContextManager::GetInstance().WaitTaskFinishOnDevice();
  RecordExitStatus();
  runtime::GraphScheduler::GetInstance().Clear();
  MemTrackerInstanceClear();
  runtime::ProfilerAnalyzer::GetInstance().Clear();
  pipeline::ClearPassConfigure();

  PrimitivePy::ClearHookRes();
  ad::ClearKPrim();
  ad::ClearPrimBpropOptimizer();

  pipeline::ClearPrimitiveEvaluatorMap();
  pipeline::ClearAttrAndMethodMap();
  pipeline::GraphExecutorPy::ClearRes();
  pipeline::JitExecutorPy::ClearRes();
  pipeline::ReclaimOptimizer();
  pipeline::ClearActionConfigure();
}

void ClearResPart2() {
  MS_LOG(INFO) << "Start clear PyNativeExecutor...";
  pynative::PyNativeExecutor::GetInstance()->ClearRes();
  MS_LOG(INFO) << "End clear PyNativeExecutor.";

  MS_LOG(INFO) << "Start clear ConfigManager...";
  ConfigManager::GetInstance().ResetIterNum();
  MS_LOG(INFO) << "End clear ConfigManager.";

  MS_LOG(INFO) << "Start clear BackendManager...";
  backend::BackendManager::GetInstance().Clear();
  MS_LOG(INFO) << "End clear BackendManager...";
  MS_LOG(INFO) << "Start clear CollectiveManager...";
  // for GE, HcclCommDestroy should after RemoveGraph in ClearGraphWrapper in ClearDeviceContexts
  (void)distributed::collective::CollectiveManager::instance()->Finalize();
  MS_LOG(INFO) << "End clear CollectiveManager.";

#if defined(__linux__) && defined(WITH_BACKEND)
  MS_LOG(INFO) << "Start clear ClusterContext...";
  // ClusterContext should be finalized only after all communication groups have been cleared.
  distributed::FinalizeCluster();
  MS_LOG(INFO) << "End clear ClusterContext.";
#endif

  MS_LOG(INFO) << "Start clear snapshot manager...";
  static auto destroy_snapshot_manager_cb = GET_COMMON_CALLBACK(DestroySnapshotMgr, void);
  if (destroy_snapshot_manager_cb != nullptr) {
    destroy_snapshot_manager_cb();
  }
  MS_LOG(INFO) << "End clear snapshot manager.";

  MS_LOG(INFO) << "Start clear device context...";
  device::DeviceContextManager::GetInstance().ClearDeviceContexts();
  MS_LOG(INFO) << "End clear device context.";

  MS_LOG(INFO) << "Start clear AnalysisResultCacheMgr...";
  pipeline::ClearAnalysisResultCacheMgr();
  MS_LOG(INFO) << "End clear AnalysisResultCacheMgr.";

  MS_LOG(INFO) << "Start clear AnalysisContext...";
  abstract::AnalysisContext::ClearContext();
  MS_LOG(INFO) << "End clear AnalysisContext...";

  MS_LOG(INFO) << "Start clear AnalysisSchedule...";
  pipeline::ClearAnalysisSchedule();
  MS_LOG(INFO) << "End clear AnalysisSchedule...";
#ifdef ENABLE_DEBUGGER
  constexpr char kReset[] = "DebuggerReset";
  static auto reset_callback = callback::CommonCallback::GetInstance().GetCallback<void>(kReset);
  if (reset_callback) {
    reset_callback();
  } else {
    MS_LOG(WARNING) << "Failed to get DebuggerReset, data dump function may not work.";
  }
#endif
  pipeline::CleanCache();
}

void ClearResPart3() {
  // clean static variable to prevent from crash. As static variable is released after
  // Python threads is released.
  MS_LOG(INFO) << "Start clear ClearObjectCache...";
  parse::data_converter::ClearObjectCache();
  MS_LOG(INFO) << "End clear ClearObjectCache...";

  MS_LOG(INFO) << "Start clear Parser...";
  pipeline::CleanParserResource();
  MS_LOG(INFO) << "End clear Parser...";

  MS_LOG(INFO) << "Start ClearTraceStack...";
  trace::ClearTraceStack();
  MS_LOG(INFO) << "End ClearTraceStack...";

  MS_LOG(INFO) << "Start clear InterpretNodeRecorder...";
  InterpretNodeRecorder::GetInstance().Clear();
  MS_LOG(INFO) << "End clear InterpretNodeRecorder...";

  MS_LOG(INFO) << "Start clear parallel::entire_costgraph...";
  parallel::entire_costgraph.reset();
  MS_LOG(INFO) << "End clear parallel::entire_costgraph...";

  MS_LOG(INFO) << "Start clear ParserDefaultObjects ...";
  pynative::ParserDefaultObjects::GetInstance().ClearRes();
  MS_LOG(INFO) << "End clear ParserDefaultObjects...";

  LLMManager::GetInstance().Clear();
  PyObjManager::Get().Clear();

  // ResetPythonScope after all py::object is freed.
  MS_LOG(INFO) << "Start clear python_adapter...";
  python_adapter::ResetPythonScope();
  MS_LOG(INFO) << "End clear python_adapter.";
}

void ClearSingleton() {
  MS_LOG(INFO) << "Start clear singleton...";
  profiler::Profiler::Clear();
  auto destroy_wait_sem_cb = GET_COMMON_CALLBACK(TftDestroyWaitSemaphore, void);
  if (destroy_wait_sem_cb != nullptr) {
    destroy_wait_sem_cb();
  }
#ifdef ENABLE_AKG
  kernel::GraphKernelBuildManager::Instance().Clear();
#endif
  somas::SomasManager::Instance().Clear();
  GraphKernelInfoManager::Instance().Clear();
  device::DataQueueMgr::GetInstance().Clear();
  session::SessionFactory::Get().Clear();
  ExecuteOrderTracker::GetInstance().Clear();
  pipeline::ClearOpPrimPyRegister();
  constexpr char kFinalize[] = "DumpJsonParserFinalize";
  static auto finalize_callback = callback::CommonCallback::GetInstance().GetCallback<void>(kFinalize);
  if (finalize_callback) {
    finalize_callback();
  } else {
    MS_LOG(WARNING) << "Failed to get DumpJsonParserFinalize, data dump function may not work.";
  }
  CommManager::Clear();
  runtime::KernelCache::GetInstance().ClearBuffers();

  MS_LOG(INFO) << "End clear singleton.";
}

void ClearResAtexit() {
  MS_LOG(INFO) << "Pipeline clear all resource";
  try {
    MsException::Instance().CheckException();
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Check exception before process exit: " << e.what();
  }
  ClearResPart1();
  ClearResPart2();

  mindspore::trans::FormatHelper::GetInstance().Clear();
  ClearResPart3();
  ClearSingleton();
  //  The premature unloading of the plugin .so triggers the process to exit during the termination phase. Other
  //  components' singletons, static variables, and global variables in MindSpore may inadvertently invoke the plugin
  //  interface, resulting in an undefined coredump.
}
}  // namespace mindspore
