/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/tree_adapter.h"

#if !defined(__APPLE__) && !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && \
  !defined(ANDROID)
#include <sys/prctl.h>
#endif

#include "minddata/dataset/core/client.h"
#include "minddata/dataset/engine/ir/datasetops/root_node.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/opt/optional/tensor_op_fusion_pass.h"
#include "minddata/dataset/engine/opt/pre/cache_transform_pass.h"
#include "minddata/dataset/engine/opt/pre/node_offload_pass.h"
#include "minddata/dataset/engine/opt/post/repeat_pass.h"
#endif
#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/engine/opt/post/auto_worker_pass.h"
#ifdef ENABLE_PYTHON
#include "minddata/dataset/engine/opt/post/generator_node_pass.h"
#endif
#include "minddata/dataset/engine/opt/pre/add_skip_pass.h"
#include "minddata/dataset/engine/opt/pre/cache_validation_pass.h"
#include "minddata/dataset/engine/opt/pre/deep_copy_pass.h"
#include "minddata/dataset/engine/opt/pre/epoch_ctrl_pass.h"
#include "minddata/dataset/engine/opt/pre/getter_pass.h"
#include "minddata/dataset/engine/opt/pre/input_validation_pass.h"
#include "minddata/dataset/engine/opt/pre/insert_map_pass.h"
#include "minddata/dataset/engine/opt/pre/node_removal_pass.h"
#include "minddata/dataset/engine/opt/pre/skip_pushdown_pass.h"
#include "minddata/dataset/engine/perf/info_collector.h"
#if !defined(__APPLE__) && !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && \
  !defined(ANDROID)
#include "minddata/dataset/engine/datasetops/send_bridge_op.h"
#include "minddata/dataset/engine/datasetops/receive_bridge_op.h"
#include "minddata/dataset/core/shared_memory_queue.h"
#include "minddata/dataset/core/message_queue.h"
#include "minddata/dataset/util/gil_scoped.h"
#include "minddata/dataset/util/ftok_key.h"
#endif

namespace mindspore {
namespace dataset {
TreeAdapter::TreeAdapter(UsageFlag usage)
    : usage_(usage),
      launched_(false),
      tree_state_(kCompileStateInit),
      optimize_(common::GetEnv("OPTIMIZE") == "true"),

      // Initialize profiling parameters
      cur_batch_num_(0),
      cur_connector_size_(0),
      cur_connector_capacity_(0) {
#if !defined(__APPLE__) && !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && \
  !defined(ANDROID)
  parent_process_id_ = -1;
  process_id_ = getpid();
  sub_process_id_ = -1;
#endif
}

Status TreeAdapter::PrePass(const std::shared_ptr<DatasetNode> &ir) {
  RETURN_UNEXPECTED_IF_NULL(ir);
  // Vector of actions in pre-pass phase
  std::vector<std::unique_ptr<IRPass>> actions;

  MS_LOG(INFO) << "Running pre pass loops.";
  (void)actions.emplace_back(std::make_unique<InputValidationPass>());
  (void)actions.emplace_back(std::make_unique<CacheValidationPass>());
  (void)actions.emplace_back(std::make_unique<NodeRemovalPass>());
  (void)actions.emplace_back(std::make_unique<InsertMapPass>());
  if (usage_ == kDeReset) {
    (void)actions.emplace_back(std::make_unique<AddSkipPass>());
    if (GlobalContext::config_manager()->fast_recovery()) {
      (void)actions.emplace_back(std::make_unique<SkipPushdownPass>());
    }
  }
  (void)actions.emplace_back(std::make_unique<EpochCtrlPass>());
  if (usage_ == kDeGetter) {
    (void)actions.emplace_back(std::make_unique<GetterPass>());
  }
#ifndef ENABLE_ANDROID
  (void)actions.emplace_back(std::make_unique<CacheTransformPass>());

  std::unique_ptr<NodeOffloadPass> offload = std::make_unique<NodeOffloadPass>();
  // Checks nodes for offload removal
  bool offload_mod = false;
  // Checks ir_tree nodes for offload removal
  RETURN_IF_NOT_OK(offload->Run(ir, &offload_mod));
  // Creates JSON object of offload nodes.
  offload_json_ = offload->GetOffloadJson();
#endif
  // Apply pre-pass actions
  for (auto &action : actions) {
    auto m = false;
    RETURN_IF_NOT_OK(action->Run(ir, &m));
  }

  MS_LOG(INFO) << "Pre pass offload complete.";
  return Status::OK();
}

Status TreeAdapter::Optimize(const std::shared_ptr<DatasetNode> &ir) {
  RETURN_UNEXPECTED_IF_NULL(ir);
  // Vector of optimizations
  std::vector<std::unique_ptr<IRNodePass>> optimizations;
  MS_LOG(INFO) << "Running optimization pass loops";
#ifndef ENABLE_ANDROID
  (void)optimizations.emplace_back(std::make_unique<TensorOpFusionPass>());
#endif
  // Apply optimization pass actions
  for (auto &optimization : optimizations) {
    bool modified = false;
    RETURN_IF_NOT_OK(optimization->Run(ir, &modified));
  }
  MS_LOG(INFO) << "Optimization pass complete.";
  return Status::OK();
}

Status TreeAdapter::PostPass(const std::shared_ptr<DatasetNode> &ir) {
  RETURN_UNEXPECTED_IF_NULL(ir);
  // Vector of actions in post-pass phase
  std::vector<std::unique_ptr<IRPass>> actions;
  MS_LOG(INFO) << "Running post pass loops.";

  // AutoWorkerPass should ideally precede CacheTransForm Pass to avoid complications of the setting
  if (GlobalContext::config_manager()->auto_num_workers() && usage_ == kDeIterator) {
    // skip this for getter pass
    (void)actions.emplace_back(std::make_unique<AutoWorkerPass>());
  }
#ifdef ENABLE_PYTHON
  (void)actions.emplace_back(std::make_unique<GeneratorNodePass>());
#endif
#ifndef ENABLE_ANDROID
  (void)actions.emplace_back(std::make_unique<RepeatPass>());
#endif
  // We will gradually move RepeatPass from ExecutionTree::PrepareTreePostAction to here.

  for (auto &action : actions) {
    auto m = false;
    RETURN_IF_NOT_OK(action->Run(ir, &m));
  }
  MS_LOG(INFO) << "Post passes complete.";
  return Status::OK();
}

Status TreeAdapter::BuildExecutionTreeRecur(const std::shared_ptr<DatasetNode> &ir,
                                            std::shared_ptr<DatasetOp> *const op) {
  RETURN_UNEXPECTED_IF_NULL(ir);
  RETURN_UNEXPECTED_IF_NULL(op);
  RETURN_UNEXPECTED_IF_NULL(tree_);
  // Build the DatasetOp ExecutionTree from the optimized IR tree
  std::vector<std::shared_ptr<DatasetOp>> ops;
  RETURN_IF_NOT_OK(ir->Build(&ops));

  CHECK_FAIL_RETURN_UNEXPECTED(!ops.empty(), "Unable to build node: " + ir->Name());

  (*op) = ops.front();  // return the first op to be added as child by the caller of this function
  RETURN_IF_NOT_OK(tree_->AssociateNode(*op));

  for (size_t i = 1; i < ops.size(); i++) {
    RETURN_IF_NOT_OK(tree_->AssociateNode(ops[i]));
    RETURN_IF_NOT_OK(ops[i - 1]->AddChild(ops[i]));
  }

  // Build the children of IR, once they return, add the return value to *op
  for (const std::shared_ptr<DatasetNode> &child_ir : ir->Children()) {
    std::shared_ptr<DatasetOp> child_op;
    RETURN_IF_NOT_OK(BuildExecutionTreeRecur(child_ir, &child_op));
    RETURN_IF_NOT_OK(ops.back()->AddChild(child_op));  // append children to the last of ops
  }

  return Status::OK();
}

#if !defined(__APPLE__) && !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && \
  !defined(ANDROID)
Status TreeAdapter::InsertSendReceiveOp() {
  RETURN_UNEXPECTED_IF_NULL(tree_);

  // get the key
  key_t key = -1;
  RETURN_IF_NOT_OK(GetKey(&key));

  // create shared memory queue
  SharedMemoryQueue send_queue(key);
  send_queue.SetReleaseFlag(false);
  MS_LOG(INFO) << "Create shared memory queue by key: " << key;

  // create message queue id first
  int msg_queue_id = msgget(key, IPC_CREAT | 0600);
  if (msg_queue_id < 0) {
    RETURN_STATUS_UNEXPECTED("Create send message by key: " + std::to_string(key) +
                             " failed. Errno: " + std::to_string(errno));
  }

  // create message queue
  MessageQueue msg_queue(key, msg_queue_id);
  msg_queue.SetReleaseFlag(false);
  MS_LOG(INFO) << "Create send message queue id: " << std::to_string(msg_queue_id) << " by key: " << std::to_string(key)
               << " success.";

  std::shared_ptr<DatasetOp> data_queue_op = nullptr;
  std::shared_ptr<DatasetOp> data_queue_op_next = nullptr;
  // the tree is reversed by begin iterator, get the data_queue_op and its next op
  for (auto itr = tree_->begin(); itr != tree_->end(); ++itr) {
    if (itr->Name() == kDeviceQueueOp) {
      data_queue_op = itr.get();
      data_queue_op_next = (--itr).get();
      break;
    }
  }

  if (data_queue_op != nullptr) {
    // dataset sink mode, the tree will be converted
    // from:
    // xxDataset -> map -> batch -> data_queue
    // to:
    // xxDataset -> map -> batch -> send -> receive -> data_queue
    CHECK_FAIL_RETURN_UNEXPECTED(data_queue_op_next != nullptr, "The next op of DataQueueOp is nullptr.");
    auto send_bridge_op = std::make_shared<SendBridgeOp>(3, send_queue, msg_queue);
    RETURN_IF_NOT_OK(tree_->AssociateNode(send_bridge_op));
    RETURN_IF_NOT_OK(send_bridge_op->AddChild(data_queue_op_next));
    RETURN_IF_NOT_OK(tree_->AssignRoot(send_bridge_op));

    auto receive_bridge_op = std::make_shared<ReceiveBridgeOp>(3, send_queue, msg_queue);
    RETURN_IF_NOT_OK(tree_->AssociateNode(receive_bridge_op));
    RETURN_IF_NOT_OK(receive_bridge_op->AddChild(tree_->root()));
    RETURN_IF_NOT_OK(tree_->AssignRoot(receive_bridge_op));

    RETURN_IF_NOT_OK(data_queue_op->RemoveChildren());
    RETURN_IF_NOT_OK(data_queue_op->AddChild(tree_->root()));
    RETURN_IF_NOT_OK(tree_->AssignRoot(data_queue_op));
  } else {
    // dataset feed mode, the tree will be converted
    // from:
    // xxDataset -> map -> ... -> batch
    // to:
    // xxDataset -> map -> ... -> batch -> send -> receive
    auto send_bridge_op = std::make_shared<SendBridgeOp>(3, send_queue, msg_queue);
    RETURN_IF_NOT_OK(tree_->AssociateNode(send_bridge_op));
    RETURN_IF_NOT_OK(send_bridge_op->AddChild(tree_->root()));
    RETURN_IF_NOT_OK(tree_->AssignRoot(send_bridge_op));

    auto receive_bridge_op = std::make_shared<ReceiveBridgeOp>(3, send_queue, msg_queue);
    RETURN_IF_NOT_OK(tree_->AssociateNode(receive_bridge_op));
    RETURN_IF_NOT_OK(receive_bridge_op->AddChild(tree_->root()));
    RETURN_IF_NOT_OK(tree_->AssignRoot(receive_bridge_op));
  }

  return Status::OK();
}

Status TreeAdapter::SplitBySendReceiveOp() {
  // split the tree_ to two part
  // part1. xxDataset -> map -> ... -> batch -> send
  // part2. receive -> iter / data_queue
  receive_tree_ = std::make_unique<ExecutionTree>();
  send_tree_ = std::make_unique<ExecutionTree>();

  // dis-assign the root from tree_ and assign the root to the receive tree
  std::shared_ptr<DatasetOp> op = tree_->root();
  RETURN_IF_NOT_OK(tree_->root()->Remove());           // del the head from tree_
  RETURN_IF_NOT_OK(receive_tree_->AssociateNode(op));  // add the head to receive_tree_
  RETURN_IF_NOT_OK(receive_tree_->AssignRoot(op));     // assign the receive_tree_ root

  // current op in receive_tree_
  std::shared_ptr<DatasetOp> receive_op = receive_tree_->root();

  while (tree_ != nullptr && tree_->root() && tree_->root()->Children().size() != 0) {
    // move the last tree to send_tree_
    if (tree_->root()->Name() == kSendBridgeOp) {
      send_tree_ = std::move(tree_);
      break;
    }

    op = tree_->root();
    RETURN_IF_NOT_OK(tree_->root()->Remove());

    // continue add the op to the receive tree
    RETURN_IF_NOT_OK(receive_tree_->AssociateNode(op));
    RETURN_IF_NOT_OK(receive_op->AddChild(op));
    receive_op = op;
  }

  return Status::OK();
}
#endif

Status TreeAdapter::Build(const std::shared_ptr<DatasetNode> &root_ir, int64_t init_epoch) {
  RETURN_UNEXPECTED_IF_NULL(root_ir);
  RETURN_IF_NOT_OK(CollectPipelineInfoStart("Pipeline", "Build"));
  // Create ExecutionTree
  tree_ = std::make_unique<ExecutionTree>();

  // Build the Execution tree from the child of the IR root node, which represent the root of the input IR tree
  std::shared_ptr<DatasetOp> root_op;
  RETURN_IF_NOT_OK(BuildExecutionTreeRecur(root_ir->Children()[0], &root_op));
  RETURN_IF_NOT_OK(tree_->AssignRoot(root_op));

  if (usage_ == kDeReset) {
    RETURN_IF_NOT_OK(AdjustReset(init_epoch));
  }

#if !defined(__APPLE__) && !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && \
  !defined(ANDROID)
  std::string env_independent_dataset = common::GetEnv("MS_INDEPENDENT_DATASET");
  transform(env_independent_dataset.begin(), env_independent_dataset.end(), env_independent_dataset.begin(), ::tolower);
  bool independent_dataset = false;
  if (env_independent_dataset == "true") {
    independent_dataset = true;
    MS_LOG(INFO) << "Environment MS_INDEPENDENT_DATASET is true, dataset will be ran in subprocess.";
  } else if (env_independent_dataset == "false" || env_independent_dataset == "") {
    independent_dataset = false;
    MS_LOG(INFO) << "Environment MS_INDEPENDENT_DATASET is false, dataset will be ran in main process.";
  } else {
    MS_LOG(WARNING) << "Environment MS_INDEPENDENT_DATASET: " << env_independent_dataset
                    << " is configured wrong, dataset will be ran in main process.";
  }
  if (independent_dataset) {
    MS_LOG(INFO) << "The original execution tree: " << *tree_;

    // add the SendBridgeOp and ReceiveBridgeOp to the tree
    RETURN_IF_NOT_OK(InsertSendReceiveOp());

    MS_LOG(INFO) << "After add SendBridgeOp and ReceiveBridgeOp, the execution tree: " << *tree_;

    // split the tree to send_tree_ and receive_tree_
    RETURN_IF_NOT_OK(SplitBySendReceiveOp());

    MS_LOG(INFO) << "After split the execution tree, the send tree: " << *send_tree_;
    MS_LOG(INFO) << "After split the execution tree, the receive tree: " << *receive_tree_;

    // Prepare the tree
    RETURN_IF_NOT_OK(send_tree_->Prepare());
    RETURN_IF_NOT_OK(receive_tree_->Prepare());

    // After the tree is prepared, the col_name_id_map can safely be obtained
    column_name_map_ = receive_tree_->root()->column_name_id_map();
  } else {
    // Prepare the tree
    RETURN_IF_NOT_OK(tree_->Prepare());

    // After the tree is prepared, the col_name_id_map can safely be obtained
    column_name_map_ = tree_->root()->column_name_id_map();
  }
#else
  // Prepare the tree
  RETURN_IF_NOT_OK(tree_->Prepare());

  // After the tree is prepared, the col_name_id_map can safely be obtained
  column_name_map_ = tree_->root()->column_name_id_map();
#endif
  RETURN_IF_NOT_OK(CollectPipelineInfoEnd("Pipeline", "Build"));
  return Status::OK();
}

Status TreeAdapter::Compile(const std::shared_ptr<DatasetNode> &input_ir, int32_t num_epochs, int64_t global_step,
                            int64_t dataset_size) {
  RETURN_UNEXPECTED_IF_NULL(input_ir);
  RETURN_IF_NOT_OK(CollectPipelineInfoStart("Pipeline", "Compile"));
  input_ir_ = input_ir;
  tree_state_ = kCompileStateIRGraphBuilt;
  MS_LOG(INFO) << "Input plan:" << '\n' << *input_ir << '\n';

  // Clone the input IR tree and insert under the root node
  // Create a root node to host the new copy of the input IR tree
  // This is done so that the compilation will process and modify the tree
  // without changing the tree associated with the user code.
  // The tree from the user code is permitted to form a graph where any node
  // is consumed by more than one parent. However, this cloning process here
  // will break the graph into a tree by copying each consumption of a node into a new copy.
  bool m = false;
  DeepCopyPass cloning_tree;
  RETURN_IF_NOT_OK(cloning_tree.Run(input_ir, &m));
  std::shared_ptr<RootNode> root_ir = cloning_tree.Root();
  root_ir->SetNumEpochs(num_epochs);
  root_ir->SetStep(global_step);
  root_ir->SetDatasetSize(dataset_size);

  tree_state_ = kCompileStateIRTreeCloned;
  MS_LOG(INFO) << "Plan before optimization:" << '\n' << *root_ir << '\n';

  // Pre-pass of the IR tree
  RETURN_IF_NOT_OK(PrePass(root_ir));

  // Optional phase of optimization
  if (optimize_) {
    RETURN_IF_NOT_OK(Optimize(root_ir));
  }

  // Post-pass of the IR tree
  RETURN_IF_NOT_OK(PostPass(root_ir));

  tree_state_ = kCompileStateOptimized;
  MS_LOG(INFO) << "Plan after optimization:" << '\n' << *root_ir << '\n';
  // Remember the root node
  root_ir_ = root_ir;

  int64_t init_epoch = dataset_size != -1 ? global_step / dataset_size : 0;
  RETURN_IF_NOT_OK(Build(root_ir_, init_epoch));
  tree_state_ = kCompileStateReady;
  RETURN_IF_NOT_OK(CollectPipelineInfoEnd("Pipeline", "Compile"));
  return Status::OK();
}

Status TreeAdapter::AdjustReset(const int64_t epoch_num) {
  if (GlobalContext::config_manager()->fast_recovery() && epoch_num > 0) {
    MS_LOG(INFO) << "Adjusting dataset pipeline for failover reset to start on epoch: " << (epoch_num + 1);
    for (auto op = tree_->begin(); op != tree_->end(); ++op) {
      RETURN_IF_NOT_OK(op->SetEpoch(epoch_num));
    }
  }
  return Status::OK();
}

Status TreeAdapter::CheckTreeIfNull() {
#if !defined(__APPLE__) && !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && \
  !defined(ANDROID)
  if (tree_ == nullptr && (send_tree_ == nullptr || receive_tree_ == nullptr)) {
    RETURN_STATUS_UNEXPECTED("Tree tree_ && (send_tree_ || receive_tree_) is a nullptr.");
  }
#else
  if (tree_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("Tree tree_ is a nullptr.");
  }
#endif
  return Status::OK();
}

Status TreeAdapter::GetNext(TensorRow *row) {
  RETURN_IF_NOT_OK(CheckTreeIfNull());
  RETURN_UNEXPECTED_IF_NULL(row);
  RETURN_IF_NOT_OK(CollectPipelineInfoStart("Pipeline", "GetNext"));
  row->clear();  // make sure row is empty

  // When cur_db_ is a nullptr, it means this is the first call to get_next, launch ExecutionTree
  if (!launched_) {
    RETURN_IF_NOT_OK(Launch());
  }
  // Record profiling info
#ifndef ENABLE_SECURITY
  uint64_t start_time = 0;
  if (tracing_ != nullptr) {
    start_time = ProfilingTime::GetCurMilliSecond();
  }
#endif

  RETURN_IF_NOT_OK(tree_->root()->GetNextRow(row));  // first buf can't be eof or empty buf with none flag
  if (row->eoe()) {                                  // return empty tensor if 1st buf is a ctrl buf (no rows)
    MS_LOG(INFO) << "End of data iteration.  cur_batch_num_: " << cur_batch_num_;
#ifndef ENABLE_SECURITY
    if (profiling_manager_ != nullptr) {
      tree_->SetEpochEnd();
      profiling_manager_->RecordEndOfEpoch(cur_batch_num_);
    }
#endif
    RETURN_IF_NOT_OK(CollectPipelineInfoEnd("Pipeline", "GetNext", {{"TensorRowFlags", row->FlagName()}}));
    return Status::OK();
  }
  if (row->eof()) {
    tree_->SetFinished();
    std::string err = "EOF buffer encountered. User tries to fetch data beyond the specified number of epochs.";
    RETURN_STATUS_UNEXPECTED(err);
  }

  // Record profiling info
#ifndef ENABLE_SECURITY
  if (tracing_ != nullptr) {
    uint64_t end_time = ProfilingTime::GetCurMilliSecond();
    cur_batch_num_++;
    cur_connector_size_ = tree_->root()->ConnectorSize();
    cur_connector_capacity_ = tree_->root()->ConnectorCapacity();
    // push time is 0ms in dataset iterator since no devices are involved
    tracing_->Record(TIME, TDT_PUSH_TIME, cur_batch_num_, 0, end_time);
    tracing_->Record(TIME, BATCH_TIME, cur_batch_num_, end_time - start_time, end_time);
    tracing_->Record(TIME, PIPELINE_TIME, cur_batch_num_, end_time - start_time, end_time);
    tracing_->Record(CONNECTOR_DEPTH, cur_connector_capacity_, cur_batch_num_, cur_connector_size_, end_time);
  }
#endif
  RETURN_IF_NOT_OK(CollectPipelineInfoEnd("Pipeline", "GetNext", {{"TensorRowFlags", row->FlagName()}}));
  return Status::OK();
}

#if !defined(__APPLE__) && !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && \
  !defined(ANDROID)
void TreeAdapter::SubprocessDaemonLoop() {
  const int log_interval = 10;
  while (true) {
    MS_LOG(INFO) << "[Independent Dataset Process] Process: " << std::to_string(process_id_)
                 << " will become a daemon, doing nothing but looping "
                 << "and waiting for the main process: " << std::to_string(parent_process_id_) << " to exit.";
    sleep(log_interval);

    // the parent had been closed
    if (getppid() != parent_process_id_) {
      MS_LOG(INFO) << "[Independent Dataset Process] Main process: " << std::to_string(parent_process_id_)
                   << " had been closed. Current process: " << std::to_string(process_id_) << " will exit too.";
      exit(0);
    }
  }
}

Status TreeAdapter::LaunchSubprocess() {
  // get the message queue id
  if (tree_->root()->Name() != kSendBridgeOp) {
    MS_LOG(EXCEPTION) << "The send_tree_ root is not SendBridgeOp.";
  }
  auto message_queue = dynamic_cast<SendBridgeOp *>(tree_->root().get())->GetMessageQueue();

  parent_process_id_ = getppid();
  process_id_ = getpid();
  sub_process_id_ = -1;
  auto tid = std::this_thread::get_id();
  std::stringstream ss;
  ss << tid;

  // execute the detached dataset in the sub-process
  MS_LOG(INFO) << "[Independent Dataset Process] Process pid: " << std::to_string(process_id_)
               << ", parent pid: " << std::to_string(parent_process_id_) << ", thread id: " << ss.str()
               << " is started successfully.";

  auto ret = tree_->Launch();
  if (ret != Status::OK()) {
    // here should prompt error because it's in subprocess
    MS_LOG(ERROR) << "[Independent Dataset Process] Process pid: " << std::to_string(process_id_) << ", launch failed.";

    // send the INDEPENDENT ERROR flag to main process
    message_queue.MsgSnd(kWorkerErrorMsg);

    // Infinite loop
    SubprocessDaemonLoop();
  }
  launched_ = true;

  // The the subprocess should be alive
  const int main_thread_log_interval = 10;
  const int main_thread_sleep_interval = 1;
  time_t start = time(nullptr);
  while (!tree_->isFinished()) {
    time_t end = time(nullptr);
    if (difftime(end, start) > main_thread_log_interval) {
      MS_LOG(INFO) << "[Independent Dataset Process] Process pid: " << std::to_string(process_id_)
                   << ", parent pid: " << std::to_string(parent_process_id_) << ", thread id: " << ss.str()
                   << " is alive. Its status is normal. Sleeping ...";
      start = time(nullptr);
    }

    sleep(main_thread_sleep_interval);

    // got error from dataset pipeline
    if (tree_->AllTasks()->GetTaskErrorIfAny() != Status::OK()) {
      MS_LOG(ERROR) << "[Independent Dataset Process] Process pid: " << std::to_string(process_id_)
                    << ", parent pid: " << std::to_string(parent_process_id_) << ", thread id: " << ss.str()
                    << ". Got error info in independent dataset pipeline. Process will exit.";

      // send the INDEPENDENT ERROR flag to main process
      message_queue.MsgSnd(kWorkerErrorMsg);
      break;
    }

    // the message queue had been released by main process
    MessageQueue::State state = dynamic_cast<SendBridgeOp *>(tree_->root().get())->MessageQueueState();
    if (state == MessageQueue::State::kReleased) {
      MS_LOG(INFO) << "[Independent Dataset Process] Process pid: " << std::to_string(process_id_)
                   << ", parent pid: " << std::to_string(parent_process_id_) << ", thread id: " << ss.str()
                   << ". Message queue had been released by main process. Process will exit.";
      tree_->AllTasks()->interrupt_all();
      break;
    }
  }

  // Infinite loop
  SubprocessDaemonLoop();

  // The process may not run to this point
  return Status::OK();
}
#endif

Status TreeAdapter::Launch() {
  RETURN_IF_NOT_OK(CheckTreeIfNull());

#if !defined(__APPLE__) && !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && \
  !defined(ANDROID)
  std::string env_independent_dataset = common::GetEnv("MS_INDEPENDENT_DATASET");
  transform(env_independent_dataset.begin(), env_independent_dataset.end(), env_independent_dataset.begin(), ::tolower);
  bool independent_dataset = false;
  if (env_independent_dataset == "true") {
    independent_dataset = true;
    MS_LOG(INFO) << "Environment MS_INDEPENDENT_DATASET is true, dataset will be ran in subprocess.";
  } else if (env_independent_dataset == "false" || env_independent_dataset == "") {
    independent_dataset = false;
    MS_LOG(INFO) << "Environment MS_INDEPENDENT_DATASET is false, dataset will be ran in main process.";
  }
  if (independent_dataset) {
    // move the send_tree_ to tree_ and launch it
    tree_ = std::move(send_tree_);

    // begin hold gil
    MS_LOG(INFO) << "[Main Dataset Process] Begin acquire gil. Current Py_IsInitialized: " << Py_IsInitialized()
                 << ", PyGILState_Check: " << PyGILState_Check();
    GilAcquireWithCheck gil_acquire_with_check;
    PyOS_BeforeFork();

    // launch the sub-process to detach dataset with send
    pid_t fpid = fork();
    if (fpid < 0) {  // fork sub-process failed
      RETURN_STATUS_UNEXPECTED("Create an independent dataset process failed.");
    } else if (fpid == 0) {  // in sub-process
      PyOS_AfterFork_Child();

      // release the gil in child process
      MS_LOG(INFO) << "[Independent Dataset Process] Begin release gil. Current Py_IsInitialized: "
                   << Py_IsInitialized() << ", PyGILState_Check: " << PyGILState_Check();
      py::gil_scoped_release release;
      MS_LOG(INFO) << "[Independent Dataset Process] End release gil. Current Py_IsInitialized: " << Py_IsInitialized()
                   << ", PyGILState_Check: " << PyGILState_Check();
      if (PyGILState_Check()) {
        MS_LOG(EXCEPTION) << "[Independent Dataset Process] PyGILState_Check: " << PyGILState_Check()
                          << ", it should be 0.";
      }

      prctl(PR_SET_NAME, "independent_dataset_process");  // set the thread name
      prctl(PR_SET_PDEATHSIG, SIGTERM);                   // if the main process exit, the subprocess will exit too.

      // launch the independent dataset process
      RETURN_IF_NOT_OK(LaunchSubprocess());
    }

    PyOS_AfterFork_Parent();

    // In the main thread
    sub_process_id_ = fpid;
    MS_LOG(INFO) << "[Main Dataset Process] Process pid: " << std::to_string(process_id_)
                 << ", sub-process pid: " << std::to_string(sub_process_id_);

    // move the receive_tree_ to tree_ and launch it
    tree_ = std::move(receive_tree_);

    // get the receive_bridge_op and set the subprocess id to it
    for (auto itr = tree_->begin(); itr != tree_->end(); ++itr) {
      if (itr->Name() == kReceiveBridgeOp) {
        auto receive_bridge_op = dynamic_cast<ReceiveBridgeOp *>(itr.get().get());
        receive_bridge_op->SetSubprocessID(sub_process_id_);
        break;
      }
    }
    MS_LOG(INFO) << "[Main Datast Process] Begin release gil. Current Py_IsInitialized: " << Py_IsInitialized()
                 << ", PyGILState_Check: " << PyGILState_Check();
  }
#endif

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__) && !defined(ENABLE_ANDROID)
  // set num threads of opencv only for main process
  int32_t thread_num = get_nprocs();
  if (thread_num == 0) {
    std::string err_msg = "Invalid thread number, got 0.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  constexpr int32_t max_cv_threads_cnt = 8;
  cv::setNumThreads(thread_num > max_cv_threads_cnt ? max_cv_threads_cnt : thread_num);
#endif

  RETURN_IF_NOT_OK(tree_->Launch());
  launched_ = true;
  return Status::OK();
}

nlohmann::json TreeAdapter::GetOffloadJson() { return offload_json_; }
}  // namespace dataset
}  // namespace mindspore
