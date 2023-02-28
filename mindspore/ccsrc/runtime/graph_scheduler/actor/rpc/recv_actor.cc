/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/actor/rpc/recv_actor.h"

#include <memory>
#include <utility>
#include <functional>
#include <condition_variable>
#include "proto/topology.pb.h"
#include "plugin/device/cpu/kernel/rpc/rpc_recv_kernel.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace runtime {
RecvActor::~RecvActor() {
  if (server_) {
    try {
      server_->Finalize();
    } catch (const std::exception &) {
      MS_LOG(ERROR) << "Failed to finalize for tcp server in recv actor.";
    }
    server_ = nullptr;
  }
}

void RecvActor::SetOpcontext(OpContext<DeviceTensor> *const op_context) {
  std::unique_lock<std::mutex> lock(context_mtx_);
  MS_EXCEPTION_IF_NULL(op_context);
  op_context_ = op_context;
}

void RecvActor::ResetOpcontext() {
  std::unique_lock<std::mutex> lock(context_mtx_);
  is_context_valid_ = false;
}

void RecvActor::UpdateStatus() {
  std::unique_lock<std::mutex> lock(context_mtx_);
  is_context_valid_ = true;
  context_cv_.notify_all();
}

void RecvActor::SetRouteInfo(uint32_t, const std::string &, const std::string &recv_src_node_name,
                             const std::string &recv_dst_node_name) {
  (void)rpc_input_node_name_.emplace_back(recv_src_node_name);
  input_inter_process_num_++;
}

bool RecvActor::StartServer() {
  // Step 1: Create a rpc server and start listening.

#ifdef ENABLE_RDMA
  if (common::GetEnv(kEnableRDMA) == "1") {
    std::string ip = common::GetEnv(kRDMAIP);
    uint32_t min_port = ClusterContext::instance()->port_range().first;
    uint32_t max_port = ClusterContext::instance()->port_range().second;
    uint32_t current_port = min_port;
    std::string url = ip + ":" + std::to_string(current_port);

    uint32_t retry_num = 0;
    server_ = std::make_unique<RDMAServer>();
    MS_EXCEPTION_IF_NULL(server_);
    while (!server_->Initialize(url) && retry_num++ < kMaxRetryPortNum && current_port <= max_port) {
      ++current_port;
      MS_LOG(WARNING) << "Failed to initialize RDMAServer with url: " << url
                      << ". Port number maybe occupied. Retry with increased port number: " << current_port;
      url = ip + ":" + std::to_string(current_port);
    }
    if (!kURPCInited) {
      MS_LOG(EXCEPTION) << "Failed to initialize RDMAServer.";
    }
  } else {
    server_ = std::make_unique<TCPServer>();
    MS_EXCEPTION_IF_NULL(server_);
    // Set the memory allocating callback using void* message.
    std::function<void *(size_t size)> allocate_callback =
      std::bind(&RecvActor::AllocateMessage, this, std::placeholders::_1);
    if (!server_->Initialize(allocate_callback)) {
      MS_LOG(EXCEPTION) << "Failed to initialize rpc server for recv actor";
    }
  }
#else
  server_ = std::make_unique<TCPServer>();
  MS_EXCEPTION_IF_NULL(server_);
  // Set the memory allocating callback using void* message.
  std::function<void *(size_t size)> allocate_callback =
    std::bind(&RecvActor::AllocateMessage, this, std::placeholders::_1);
  if (!server_->Initialize(allocate_callback)) {
    MS_LOG(EXCEPTION) << "Failed to initialize rpc server for recv actor";
  }
#endif

  // Step 2: Set the message handler of the server.
  SetMessageHandler();

  ip_ = server_->GetIP();
  port_ = server_->GetPort();
  std::string server_url = ip_ + ":" + std::to_string(port_);
  // Step 3: Register the server address to route table. The server should not be connected before this step is done.
  for (const auto &inter_process_edge_name : inter_process_edge_names_) {
    MS_LOG(INFO) << "Start server for recv actor. Server address: " << server_url
                 << ", remote function id: " << kRemoteFuncId
                 << ", inter-process edge name: " << inter_process_edge_name;
    distributed::cluster::topology::ActorAddress recv_actor_addresss;
    recv_actor_addresss.set_actor_id(inter_process_edge_name);
    recv_actor_addresss.set_ip(ip_);
    recv_actor_addresss.set_port(port_);
    recv_actor_addresss.set_func_id(kRemoteFuncId);
    MS_EXCEPTION_IF_NULL(actor_route_table_proxy_);
    if (!actor_route_table_proxy_->RegisterRoute(inter_process_edge_name, recv_actor_addresss)) {
      MS_LOG(EXCEPTION) << "Failed to register route for " << inter_process_edge_name << " " << server_url
                        << " when starting server.";
    }
  }
  return true;
}

void RecvActor::Clear() {
  if (server_) {
    server_->Finalize();
    server_ = nullptr;
  }
}

void RecvActor::StopRpcAtException() {
  std::unique_lock<std::mutex> lock(context_mtx_);
  if (!is_context_valid_) {
    is_exception_thrown_ = true;
    context_cv_.notify_all();
  }
}

void RecvActor::RunOpInterProcessData(MessageBase *const msg, OpContext<DeviceTensor> *const context) {
  MS_ERROR_IF_NULL_WO_RET_VAL(msg);
  MS_ERROR_IF_NULL_WO_RET_VAL(op_context_);
  MS_ERROR_IF_NULL_WO_RET_VAL(context);
  auto &sequential_num = context->sequential_num_;
  (void)input_op_inter_process_[sequential_num].emplace_back(msg->From().Name());

  auto is_run = CheckRunningCondition(context);
  MS_LOG(INFO) << "Actor(" << GetAID().Name() << ") receive the input op inter-process. Edge is "
               << inter_process_edge_names_ << ". Check running condition:" << is_run;

  // Parse the message from remote peer and set to rpc recv kernel.
  auto recv_kernel_mod = dynamic_cast<kernel::RpcKernelMod *>(kernel_info_->MutableKernelMod());
  MS_ERROR_IF_NULL_WO_RET_VAL(recv_kernel_mod);

  // We set remote data by the interface of the rpc kernel, because currently there's no remote input for a kernel mod.
  recv_kernel_mod->SetRemoteInput(msg);
  if (common::GetEnv(kEnableRDMA) == "1") {
    rdma_buf_ = msg->data;
  }

  if (is_run) {
    Run(context);
  }
  return;
}

bool RecvActor::CheckRunningCondition(const OpContext<DeviceTensor> *context) const {
  MS_EXCEPTION_IF_NULL(context);
  // Step 1: Judge data and control inputs are satisfied.
  bool is_data_and_control_arrow_satisfied = AbstractActor::CheckRunningCondition(context);
  if (!is_data_and_control_arrow_satisfied) {
    return false;
  }

  if (input_inter_process_num_ != 0) {
    // Step 2: Judge inter-process inputs are satisfied.
    const auto &inter_process_iter = input_op_inter_process_.find(context->sequential_num_);
    if (inter_process_iter == input_op_inter_process_.end()) {
      return false;
    }

    const auto &current_inter_process_inputs = inter_process_iter->second;
    if (current_inter_process_inputs.size() < input_inter_process_num_) {
      return false;
    } else if (current_inter_process_inputs.size() > input_inter_process_num_) {
      MS_LOG(ERROR) << "Invalid inter process input num:" << current_inter_process_inputs.size()
                    << " need:" << input_inter_process_num_ << " for actor:" << GetAID();
      return false;
    }
  }
  return true;
}

void RecvActor::EraseInput(const OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  KernelActor::EraseInput(context);
  if (input_op_inter_process_.count(context->sequential_num_) != 0) {
    (void)input_op_inter_process_.erase(context->sequential_num_);
  }
  // Release data allocated by AllocateMessage.
  if (recv_data_ != nullptr) {
    device_contexts_[0]->device_res_manager_->FreeMemory(recv_data_.get());
  }

#ifdef ENABLE_RDMA
  // Release data of URPC by caller.
  if (common::GetEnv(kEnableRDMA) == "1" && rdma_buf_ != nullptr) {
    auto rdma_server = dynamic_cast<RDMAServer *>(server_.get());
    MS_EXCEPTION_IF_NULL(rdma_server);
    auto urpc_alloc = rdma_server->urpc_allocator();
    MS_EXCEPTION_IF_NULL(urpc_alloc);
    urpc_alloc->free(rdma_buf_);
  }
#endif
}

void RecvActor::Run(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(kernel_info_);
  auto recv_kernel_mod = dynamic_cast<kernel::RpcKernelMod *>(kernel_info_->MutableKernelMod());
  MS_EXCEPTION_IF_NULL(recv_kernel_mod);
  auto remote_input = recv_kernel_mod->GetRemoteInput();
  bool need_finalize = false;
  // Preprocess the remote input in case data is dynamic shape.
  PreprocessRemoteInput(remote_input, &need_finalize);
  if (need_finalize) {
    return;
  }
  KernelActor::Run(context);
}

void *RecvActor::AllocateMessage(size_t size) {
  // Block this method until the context is valid.
  std::unique_lock<std::mutex> lock(context_mtx_);
  context_cv_.wait(lock, [this] { return is_context_valid_; });
  lock.unlock();

  return AllocateMemByDeviceRes(size);
}

void *RecvActor::AllocateMemByDeviceRes(size_t size) {
  // Only need to create recv_data_ once.
  // The real data is allocated and freed multiple times as recv_data_->ptr_.
  if (recv_data_ == nullptr) {
    recv_data_ = std::make_shared<CPUDeviceAddress>(nullptr, size);
    MS_ERROR_IF_NULL_W_RET_VAL(recv_data_, nullptr);
  } else {
    recv_data_->SetSize(size);
  }

  MS_ERROR_IF_NULL_W_RET_VAL(device_contexts_[kIndex0], nullptr);
  MS_ERROR_IF_NULL_W_RET_VAL(device_contexts_[kIndex0]->device_res_manager_, nullptr);
  if (!device_contexts_[kIndex0]->device_res_manager_->AllocateMemory(recv_data_.get())) {
    MS_LOG(ERROR) << "Failed to allocate memory size " << size;
    return nullptr;
  }
  return recv_data_->GetMutablePtr();
}

void RecvActor::AddArgSpecForInput(AbstractBasePtrList *args_spec_list, const ShapeVector &shapes, TypeId data_type,
                                   size_t input_index) const {
  MS_EXCEPTION_IF_NULL(args_spec_list);
  MS_EXCEPTION_IF_NULL(kernel_);
  auto input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(kernel_, input_index, false);
  auto real_input = input_node_with_index.first;
  size_t real_input_index = input_node_with_index.second;
  MS_EXCEPTION_IF_NULL(real_input);
  auto output_addr = AnfAlgo::GetMutableOutputAddr(real_input, real_input_index, false);
  MS_EXCEPTION_IF_NULL(output_addr);
  if (output_addr->GetNodeIndex().first == nullptr) {
    output_addr->SetNodeIndex(kernel_, input_index);
  }
  auto out_tensor = std::make_shared<tensor::Tensor>(data_type, shapes);
  MS_EXCEPTION_IF_NULL(out_tensor);
  out_tensor->set_device_address(output_addr, false);
  out_tensor->data_sync();

  auto real_abs = real_input->abstract();
  MS_EXCEPTION_IF_NULL(real_abs);
  auto updated_shape = std::make_shared<abstract::Shape>(shapes);
  MS_EXCEPTION_IF_NULL(updated_shape);
  if (real_abs->isa<abstract::AbstractTensor>()) {
    real_abs->set_value(out_tensor);
    real_abs->set_shape(updated_shape);
  } else if (real_abs->isa<abstract::AbstractTuple>()) {
    if (common::AnfAlgo::IsDynamicSequence(real_input)) {
      MS_LOG(EXCEPTION) << "Invalid dynamic sequence for actor:" << GetAID() << " node:" << real_input->DebugString();
    }
    auto abstract_tuple = real_abs->cast<abstract::AbstractTuplePtr>();
    MS_EXCEPTION_IF_NULL(abstract_tuple);
    MS_EXCEPTION_IF_CHECK_FAIL((real_input_index < abstract_tuple->elements().size()), "Index is out of range.");
    auto tuple_elements = abstract_tuple->elements()[real_input_index];
    MS_EXCEPTION_IF_NULL(tuple_elements);
    tuple_elements->set_value(out_tensor);
    tuple_elements->set_shape(updated_shape);
  }
  common::AnfAlgo::AddArgList(args_spec_list, real_input, real_input_index);

  // The inputs of RpcRecv node are all in device tensor store(weight or value node), framework does not free these
  // device tensors. If these device tensors are not released, they will persist in the same memory size. In dynamic
  // shape scenarios, there will be out of memory bounds problems.
  MS_EXCEPTION_IF_NULL(output_addr);
  auto output_addr_size = AnfAlgo::GetOutputTensorMemSize(real_input, real_input_index);
  if (output_addr_size != output_addr->GetSize()) {
    output_addr->SetSize(output_addr_size);
    MS_EXCEPTION_IF_NULL(device_contexts_[0]);
    MS_EXCEPTION_IF_NULL(device_contexts_[0]->device_res_manager_);
    device_contexts_[0]->device_res_manager_->FreeMemory(output_addr.get());
  }
}

size_t RecvActor::ParseDynamicShapeData(const RpcDataPtr &dynamic_shape_data, size_t data_size,
                                        AbstractBasePtrList *args_spec_list, size_t count) {
  // The data which could be parsed by offset in dynamic shape scenario.
  auto data_to_be_parsed = dynamic_shape_data;
  // The real data offsets which will be used by RpcRecvKernel.
  std::vector<size_t> real_data_offsets;

  // Once the magic header is dynamic shape, each input of the Recv is dynamic shape.
  // So traverse each input and parse the dynamic shape data.
  size_t offset = 0;
  for (size_t i = 0; i < count; i++) {
    if (data_to_be_parsed >= dynamic_shape_data + data_size) {
      MS_LOG(EXCEPTION) << "The dynamic shape data size is invalid.";
    }
    // Step 1: parse the magic header which indicates the dynamic shape.
    std::string dynamic_shape_magic_header(data_to_be_parsed, strlen(kRpcDynamicShapeData));
    if (dynamic_shape_magic_header != kRpcDynamicShapeData) {
      MS_LOG(EXCEPTION) << "The dynamie shape data must have the magic header RPC_DYNAMIC_SHAPE_DATA. But got "
                        << dynamic_shape_magic_header;
    }

    // Step 2: parse the size of serialized protobuf message.
    data_to_be_parsed += strlen(kRpcDynamicShapeData);
    size_t pb_msg_size = 0;
    MS_EXCEPTION_IF_CHECK_FAIL(memcpy_s(&pb_msg_size, sizeof(pb_msg_size), data_to_be_parsed, sizeof(size_t)) == EOK,
                               "memcpy_s protobuf message size failed.");

    // Step 3: deserialize the protobuf message.
    data_to_be_parsed += sizeof(pb_msg_size);
    rpc::DynamicShapeMessage pb_msg;
    (void)pb_msg.ParseFromArray(data_to_be_parsed, SizeToInt(pb_msg_size));

    // Step 4: parse the data shape and
    ShapeVector shapes(pb_msg.shape_vector().begin(), pb_msg.shape_vector().end());
    TypeId data_type = static_cast<TypeId>(pb_msg.type_id());
    data_to_be_parsed += pb_msg_size;

    // Step 5: get the size of real data as recv's input.
    int64_t real_data_size = 1;
    if (!kernel::GetShapeSize(shapes, TypeIdToType(data_type), &real_data_size)) {
      MS_LOG(EXCEPTION) << "Getting shape size for shape " << shapes << " failed.";
    }
    data_to_be_parsed += real_data_size;

    // Step 6: update the abstract.
    AddArgSpecForInput(args_spec_list, shapes, data_type, i);

    offset += strlen(kRpcDynamicShapeData) + sizeof(pb_msg_size) + pb_msg_size;
    real_data_offsets.push_back(offset);
    offset += LongToSize(real_data_size);
  }

  auto recv_kernel_mod = dynamic_cast<kernel::RpcRecvKernelMod *>(kernel_info_->MutableKernelMod());
  MS_EXCEPTION_IF_NULL(recv_kernel_mod);
  recv_kernel_mod->set_real_data_offset(real_data_offsets);
  return offset;
}

void RecvActor::PreprocessRemoteInput(const MessageBase *const msg, bool *need_finalize) {
  MS_EXCEPTION_IF_NULL(msg);
  MS_EXCEPTION_IF_NULL(need_finalize);

  // Parse the void * data.
  size_t data_size = msg->size;
  MS_EXCEPTION_IF_NULL(msg->data);
  std::string msg_magic_header = std::string(static_cast<RpcDataPtr>(msg->data), strlen(kRpcDynamicShapeData));
  RpcDataPtr dynamic_shape_data = static_cast<RpcDataPtr>(msg->data);

  if (data_size <= strlen(kRpcDynamicShapeData)) {
    MS_LOG(DEBUG) << "This is not a dynamic shape data. No need to preprocess.";
    return;
  }
  if (msg_magic_header != kRpcDynamicShapeData) {
    MS_LOG(DEBUG) << "This is not a dynamic shape data. No need to preprocess.";
    return;
  }

  MS_LOG(INFO) << "Preprocess for dynamic shape data.";
  AbstractBasePtrList args_spec_list;
  size_t input_size = common::AnfAlgo::GetInputTensorNum(kernel_);
  size_t dynamic_shape_data_msg_len = ParseDynamicShapeData(dynamic_shape_data, data_size, &args_spec_list, input_size);
  ParseFinalizeReqData(dynamic_shape_data_msg_len, msg, need_finalize);

  // The args_spec_list is updated in ParseDynamicShapeData method. So do the Infer and Resize operation.
  auto eval_result = opt::CppInferShapeAndType(common::AnfAlgo::GetCNodePrimitive(kernel_), args_spec_list);
  kernel_->set_abstract(eval_result);
  auto args = kernel::AbstractArgsFromCNode(kernel_);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel_);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  if (kernel_mod->Resize(args.op, args.inputs, args.outputs, args.depend_tensor_map) ==
      static_cast<int>(kernel::KRET_RESIZE_FAILED)) {
    MS_LOG(EXCEPTION) << "Node " << kernel_->fullname_with_scope() << " Resize() failed.";
  }
}

MessageBase *RecvActor::HandleMessage(MessageBase *const msg) {
  // Block the message handler if the context is invalid.
  std::unique_lock<std::mutex> lock(context_mtx_);
  context_cv_.wait(lock, [this] { return is_context_valid_ || is_exception_thrown_; });
  if (is_exception_thrown_) {
    MS_LOG(WARNING) << "Recv actor stops waiting for op_context at exception.";
    return distributed::rpc::NULL_MSG;
  }
  lock.unlock();
  // Once recv actor is launched, lock the context so that the next step's recv will not be launched in advance.
  ResetOpcontext();

  MS_LOG(INFO) << "Rpc actor recv message for inter-process edge: " << inter_process_edge_names_;

  if (msg == nullptr || op_context_ == nullptr) {
    return distributed::rpc::NULL_MSG;
  }
  ActorDispatcher::Send(GetAID(), &RecvActor::RunOpInterProcessData, msg, op_context_);
  return distributed::rpc::NULL_MSG;
}

void RecvActor::SetMessageHandler() {
  server_->SetMessageHandler(std::bind(&RecvActor::HandleMessage, this, std::placeholders::_1), ++kRemoteFuncId);
}
}  // namespace runtime
}  // namespace mindspore
