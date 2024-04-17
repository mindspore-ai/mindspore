/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/actor/actor_dump.h"
#include <map>
#include <utility>
#include <deque>

#include "runtime/graph_scheduler/scheduler_helper.h"
namespace mindspore {
namespace runtime {
namespace {
std::string GetSplitName(const std::string &name) {
  auto index = name.rfind('/');
  if ((index != std::string::npos) && (index < name.size() - 1)) {
    return name.substr(index + 1);
  }
  return name;
}

void DumpBaseInputInfo(const AbstractActor *actor, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor);
  // Dump device tensor store.
  if (actor->device_tensor_store_keys().size() > 0) {
    ofs << "\t\tdevice_tensor_store_keys:" << actor->device_tensor_store_keys().size() << "\n ";
    for (const auto &device_tensor_store_key : actor->device_tensor_store_keys()) {
      MS_EXCEPTION_IF_NULL(device_tensor_store_key.second);
      ofs << "\t\t\tto_input_index:" << device_tensor_store_key.first
          << "\tfrom_node_name:" << device_tensor_store_key.second->fullname_with_scope() << "\n";
    }
  }

  // Dump input data arrow.
  if (actor->input_data_arrow_aids().size() > 0) {
    ofs << "\t\tinput_data_arrow_actors:" << actor->input_data_arrow_aids().size() << "\n ";
    for (const auto &input_data_arrow_aid : actor->input_data_arrow_aids()) {
      ofs << "\t\t\tfrom_actor_name:" << input_data_arrow_aid.first.Name() << "\n";
    }
  }

  // Dump input control arrow.
  if (actor->input_control_arrow_aids().size() > 0) {
    ofs << "\t\tinput_control_arrow_actors:" << actor->input_control_arrow_aids().size() << "\n ";
    for (const auto &input_control_arrow_aid : actor->input_control_arrow_aids()) {
      ofs << "\t\t\tfrom_actor_name:" << input_control_arrow_aid.first.Name() << "\n";
    }
  }
}

void DumpBaseOutputInfo(const AbstractActor *actor, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor);
  // Dump batch output data arrow.
  size_t batch_output_data_size = 0;
  if (actor->batch_output_data_arrows().size() > 0) {
    ofs << "\t\tbatch_output_data_arrows:" << actor->batch_output_data_arrows().size() << "\n ";
    for (const auto &batch_output_data_arrow : actor->batch_output_data_arrows()) {
      batch_output_data_size += batch_output_data_arrow.second.size();
      ofs << "\t\t\tbatch_to_actor_name:" << batch_output_data_arrow.first
          << "\tbatch_size:" << batch_output_data_arrow.second.size() << "\n";
      for (const auto &data_arrow : batch_output_data_arrow.second) {
        MS_EXCEPTION_IF_NULL(data_arrow);
        ofs << "\t\t\t\tfrom_output_index:" << data_arrow->from_output_index_
            << "\tto_actor_name:" << data_arrow->to_op_id_.Name() << "\tto_input_index:" << data_arrow->to_input_index_
            << "\tflag:" << data_arrow->flag_ << "\n";
      }
    }
  }

  // Dump output data arrow.
  if (actor->output_data_arrows().size() != actor->output_data_nodes().size()) {
    MS_LOG(EXCEPTION) << "The size of output data arrows is not equal to the output nodes, arrow num:"
                      << actor->output_data_arrows().size() << " node num:" << actor->output_data_nodes().size()
                      << " for actor:" << actor->GetAID().Name();
  }
  if (actor->output_data_arrows().size() > 0) {
    ofs << "\t\toutput_data_arrows:" << actor->output_data_arrows().size() - batch_output_data_size << "\n ";
    size_t batch_count = 0;
    for (size_t i = 0; i < actor->output_data_arrows().size(); ++i) {
      auto data_arrow = actor->output_data_arrows()[i];
      auto output_node = actor->output_data_nodes()[i];
      MS_EXCEPTION_IF_NULL(data_arrow);
      if (!TEST_FLAG(data_arrow->flag_, kOutputDataFlagBatch)) {
        std::string node_name = (output_node != nullptr) ? GetSplitName(output_node->fullname_with_scope()) : "";
        ofs << "\t\t\tfrom_output_node:" << node_name << "\tfrom_output_index:" << data_arrow->from_output_index_
            << "\tto_actor_name:" << data_arrow->to_op_id_.Name() << "\tto_input_index:" << data_arrow->to_input_index_
            << "\tflag:" << data_arrow->flag_ << "\n";
      } else {
        ++batch_count;
      }
    }
    if (batch_count != batch_output_data_size) {
      MS_LOG(EXCEPTION) << "Check batch output data error, the expect num:" << batch_output_data_size
                        << ", but get num:" << batch_count << " for " << actor->GetAID().Name();
    }
  }

  // Dump output control arrow.
  const auto &output_control_arrows = actor->output_control_arrows();
  if (output_control_arrows.size() > 0) {
    ofs << "\t\toutput_control_arrows:" << output_control_arrows.size() << "\n ";
    for (const auto &output_control_arrow : output_control_arrows) {
      MS_EXCEPTION_IF_NULL(output_control_arrow);
      ofs << "\t\t\tto_actor_name:" << output_control_arrow->to_op_id_.Name()
          << "\tflag:" << output_control_arrow->flag_ << "\n";
    }
  }
}

void DumpAbstractActor(const AbstractActor *actor, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor);
  if (actor->parent_fusion_actor() != nullptr) {
    ofs << "\t\tparent_fusion_actor:" << actor->parent_fusion_actor()->GetAID().Name() << "\n ";
  }
  // Dump device context.
  if (actor->device_contexts().size() > 0) {
    ofs << "\t\tdevice_contexts:" << actor->device_contexts().size() << "\n ";
    for (const auto &device_context : actor->device_contexts()) {
      if (device_context == nullptr) {
        ofs << "\t\t\tdevice_context:" << device_context << "\n";
        continue;
      }
      ofs << "\t\t\tdevice_context:" << device_context->device_context_key().ToString() << "\n";
    }
  }

  DumpBaseInputInfo(actor, ofs);
  DumpBaseOutputInfo(actor, ofs);

  // Dump internal parameters.
  if (actor->internal_parameters().size() > 0) {
    ofs << "\t\tinternal_parameters:" << actor->internal_parameters().size() << "\n ";
    for (auto &internal_parameter_iter : actor->internal_parameters()) {
      MS_EXCEPTION_IF_NULL(internal_parameter_iter.first.first);
      for (auto &internal_parameter_weakptr : internal_parameter_iter.second) {
        auto internal_parameter = internal_parameter_weakptr.lock();
        MS_EXCEPTION_IF_NULL(internal_parameter);
        ofs << "\t\t\toutput_node:" << internal_parameter_iter.first.first->fullname_with_scope()
            << "\toutput_index:" << internal_parameter_iter.first.second
            << "\tinternal_parameter:" << internal_parameter->DebugString() << "\n";
      }
    }
  }

  // Dump dependent actors.
  if (actor->dependent_actors().size() > 0) {
    ofs << "\t\tdependent_actors:" << actor->dependent_actors().size() << "\n ";
    for (auto &dependent_actor : actor->dependent_actors()) {
      ofs << "\t\t\tdependent_actor_name:" << dependent_actor << "\n ";
    }
  }

  // Dump the memory actor insert position.
  if (actor->memory_alloc_insert_position() != nullptr) {
    ofs << "\t\tmemory_alloc_insert_from_position:" << actor->memory_alloc_insert_position()->GetAID().Name() << "\n";
  }
  if (actor->memory_free_insert_position() != nullptr) {
    ofs << "\t\tmemory_free_insert_to_position:" << actor->memory_free_insert_position()->GetAID().Name() << "\n";
  }
}

void DumpDSActor(const DataSourceActor *actor, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor);
  const auto &actor_name = actor->GetAID().Name();
  ofs << "\tactor_name:" << actor_name << "\tactor_id:" << actor->actor_id() << "\n";

  if (actor->type() == KernelTransformType::kDeviceDataSourceActor) {
    // Dump the member info of device queue data source actor.
    const auto &device_queue_ds_actor = dynamic_cast<const DeviceQueueDataSourceActor *>(actor);
    MS_EXCEPTION_IF_NULL(device_queue_ds_actor);
    const auto &data_kernel = device_queue_ds_actor->data_kernel();
    MS_EXCEPTION_IF_NULL(data_kernel);
    ofs << "\t\tdata_kernel_name:" << data_kernel->fullname_with_scope()
        << "\tinput_number:" << common::AnfAlgo::GetInputTensorNum(data_kernel)
        << "\toutput_number:" << AnfAlgo::GetOutputTensorNum(data_kernel) << "\n";
    for (size_t i = 0; i < AnfAlgo::GetOutputTensorNum(data_kernel); ++i) {
      const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(data_kernel, i, false);
      MS_EXCEPTION_IF_NULL(device_tensor);
      ofs << "\t\t\toutput_index:" << i << "\tptr:" << device_tensor->GetPtr() << "\tsize:" << device_tensor->GetSize()
          << "\tstream id:" << device_tensor->stream_id()
          << "\toriginal_ref_count:" << device_tensor->original_ref_count()
          << "\tdynamic_ref_count:" << device_tensor->dynamic_ref_count() << "\tflag:" << device_tensor->flag()
          << "\n ";
    }
  } else if (actor->type() == KernelTransformType::kHostDataSourceActor) {
    // Dump the member info of host queue data source actor.
    const auto &host_queue_ds_actor = dynamic_cast<const HostQueueDataSourceActor *>(actor);
    MS_EXCEPTION_IF_NULL(host_queue_ds_actor);
    ofs << "\t\tdata_nodes:" << host_queue_ds_actor->data_nodes().size() << "\n";
    for (size_t i = 0; i < host_queue_ds_actor->data_nodes().size(); ++i) {
      const auto &data_node = host_queue_ds_actor->data_nodes()[i];
      MS_EXCEPTION_IF_NULL(data_node.first);
      const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(data_node.first, data_node.second, false);
      MS_EXCEPTION_IF_NULL(device_tensor);
      ofs << "\t\t\tnode_order_number:" << i << "\tnode_name:" << data_node.first->fullname_with_scope()
          << "\tdebug_name:" << data_node.first->DebugString() << "\tindex:" << data_node.second
          << "\tptr:" << device_tensor->GetPtr() << "\tsize:" << device_tensor->GetSize()
          << "\tstream id:" << device_tensor->stream_id()
          << "\toriginal_ref_count:" << device_tensor->original_ref_count()
          << "\tdynamic_ref_count:" << device_tensor->dynamic_ref_count() << "\tflag:" << device_tensor->flag()
          << "\n ";
    }
  }

  DumpAbstractActor(actor, ofs);
  ofs << "\n";
}

void DumpKernelActor(const KernelActor *actor, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor);
  ofs << "\tactor_name:" << actor->GetAID().Name() << "\tactor_id:" << actor->actor_id() << "\n";

  const auto &kernel = actor->kernel();
  MS_EXCEPTION_IF_NULL(kernel);
  auto kernel_info = dynamic_cast<KernelInfo *>(kernel->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  ofs << "\t\tkernel_name:" << kernel->fullname_with_scope() << "\tstream id:" << kernel_info->stream_id()
      << "\tinputs_num:" << common::AnfAlgo::GetInputTensorNum(kernel)
      << "\tignored_input_addresses_num:" << SchedulerHelper::GetIgnoredInputAddressCount(kernel)
      << "\toutputs_num:" << AnfAlgo::GetOutputTensorNum(kernel) << "\tis_dynamic_shape:" << actor->is_dynamic_shape()
      << "\tis_launch_skipped:" << actor->is_launch_skipped() << "\n";
  const auto &somas_outputs = kernel_info->somas_output_result();
  const auto &somas_graph_output_indexes = actor->somas_graph_output_indexes();
  for (size_t i = 0; i < AnfAlgo::GetOutputTensorNum(kernel); ++i) {
    const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(kernel, i, false);
    MS_EXCEPTION_IF_NULL(device_tensor);
    ofs << "\t\t\toutput_index:" << i << "\tptr:" << device_tensor->GetPtr() << "\tsize:" << device_tensor->GetSize()
        << "\tstream id:" << device_tensor->stream_id()
        << "\toriginal_ref_count:" << device_tensor->original_ref_count()
        << "\tdynamic_ref_count:" << device_tensor->dynamic_ref_count() << "\tflag:" << device_tensor->flag()
        << "\tis_somas_enable:" << kernel_info->IsTensorEnableSomas(somas_outputs, i)
        << "\tsomas_offset:" << kernel_info->GetTensorSomasOffset(somas_outputs, i)
        << "\tsomas_aligned_size:" << kernel_info->GetTensorSomasAlignedSize(somas_outputs, i)
        << "\tsoams_whether_graph_output:" << somas_graph_output_indexes.count(i) << "\n ";
  }
  const auto &somas_workspace = kernel_info->somas_workspace_result();
  const auto &workspace_addresses = kernel_info->workspace_address_list();
  for (size_t i = 0; i < workspace_addresses.size(); ++i) {
    auto &device_tensor = workspace_addresses[i];
    MS_EXCEPTION_IF_NULL(device_tensor);
    ofs << "\t\t\tworkspace_index:" << i << "\tptr:" << device_tensor->GetPtr() << "\tsize:" << device_tensor->GetSize()
        << "\tstream id:" << device_tensor->stream_id()
        << "\toriginal_ref_count:" << device_tensor->original_ref_count()
        << "\tdynamic_ref_count:" << device_tensor->dynamic_ref_count() << "\tflag:" << device_tensor->flag()
        << "\tis_somas_enable:" << kernel_info->IsTensorEnableSomas(somas_workspace, i)
        << "\tsomas_offset:" << kernel_info->GetTensorSomasOffset(somas_workspace, i)
        << "\tsomas_aligned_size:" << kernel_info->GetTensorSomasAlignedSize(somas_workspace, i) << "\n ";
  }

  DumpAbstractActor(actor, ofs);

  if (actor->modifiable_ref_input_indexes().size() != 0) {
    ofs << "\t\tmodifiable_ref_input_indexes:" << actor->modifiable_ref_input_indexes().size() << "\n";
    for (auto &ref_input_index : actor->modifiable_ref_input_indexes()) {
      ofs << "\t\t\tmodifiable_ref_input_index:" << ref_input_index << "\n ";
    }
  }
  if (actor->modifiable_ref_output_indexes().size() != 0) {
    ofs << "\t\tmodifiable_ref_output_indexes:" << actor->modifiable_ref_output_indexes().size() << "\n";
    for (auto &ref_output_index : actor->modifiable_ref_output_indexes()) {
      ofs << "\t\t\tmodifiable_ref_output_index:" << ref_output_index << "\n ";
    }
  }

  ofs << "\n";
}

void DumpCustomActor(const CustomActor *actor, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor);
  ofs << "\tactor_name:" << actor->GetAID().Name() << "\tactor_id:" << actor->actor_id() << "\n";
  DumpAbstractActor(actor, ofs);
  ofs << "\n";
}

void DumpSwapActor(const MemorySwapActor *actor, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor);
  ofs << "\tactor_name:" << actor->GetAID().Name() << "\tactor_id:" << actor->actor_id() << "\n";
  DumpAbstractActor(actor, ofs);
  ofs << "\n";
}

void DumpSuperKernelActor(const SuperKernelActor *actor, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor);
  ofs << "\tactor_name:" << actor->GetAID().Name() << "\tactor_id:" << actor->actor_id() << "\n";

  const auto &graph = actor->graph();
  MS_EXCEPTION_IF_NULL(graph);

  ofs << "\t\tgraph_id:" << graph->graph_id() << "\tgraphl_name:" << graph->ToString()
      << "\tis_graph_run_mode:" << graph->is_graph_run_mode() << "\tis_loop_count_sink:" << graph->is_loop_count_sink()
      << "\tinputs_num:" << (graph->input_nodes()).size() << "\tkernels_num:" << (graph->execution_order()).size()
      << "\tis enable zero copy:" << graph->has_flag(kFlagEnableZeroCopyInGraph) << "\n";

  DumpAbstractActor(actor, ofs);
  ofs << "\n";
}

void DumpAnyTypeKernelActor(const AnyTypeKernelActor *actor, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor);
  ofs << "\tactor_name:" << actor->GetAID().Name() << "\tactor_id:" << actor->actor_id() << "\n";
  const auto &graph = actor->graph();
  MS_EXCEPTION_IF_NULL(graph);
  ofs << "\t\tgraph_id:" << graph->graph_id() << "\tgraphl_name:" << graph->ToString()
      << "\tis_graph_run_mode:" << graph->is_graph_run_mode() << "\tis_loop_count_sink:" << graph->is_loop_count_sink()
      << "\tinputs_num:" << (graph->input_nodes()).size() << "\tkernels_num:" << (graph->execution_order()).size()
      << "\tis enable zero copy:" << graph->has_flag(kFlagEnableZeroCopyInGraph) << "\n";

  DumpAbstractActor(actor, ofs);
  ofs << "\n";
}

void DumpMemoryActor(const MemoryAwareActor *actor, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor);
  ofs << "\tactor_name:" << actor->GetAID().Name() << "\tactor_id:" << actor->actor_id() << "\n";

  SomasInfo *somas_info = nullptr;
  if (actor->type() == KernelTransformType::kMemoryAllocActor) {
    auto alloc_actor = dynamic_cast<const MemoryAllocActor *>(actor);
    MS_EXCEPTION_IF_NULL(alloc_actor);
    somas_info = alloc_actor->somas_info();
  } else {
    auto free_actor = dynamic_cast<const MemoryFreeActor *>(actor);
    MS_EXCEPTION_IF_NULL(free_actor);
    somas_info = free_actor->somas_info();
  }

  MS_EXCEPTION_IF_NULL(somas_info);
  ofs << "\t\tgraph_id:" << somas_info->graph_id_ << "\twhole_block_size:" << somas_info->whole_block_size_ << "\n ";

  DumpAbstractActor(actor, ofs);

  ofs << "\n";
}

void DumpCopyActor(const CopyActor *actor, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor);
  ofs << "\tactor_name:" << actor->GetAID().Name() << "\tactor_id:" << actor->actor_id() << "\n";

  auto &device_tensor = actor->output();
  if (device_tensor != nullptr) {
    ofs << "\t\toutput_index:" << 0 << "\tptr:" << device_tensor->GetPtr() << "\tsize:" << device_tensor->GetSize()
        << "\tstream id:" << device_tensor->stream_id()
        << "\toriginal_ref_count:" << device_tensor->original_ref_count()
        << "\tdynamic_ref_count:" << device_tensor->dynamic_ref_count() << "\tflag:" << device_tensor->flag() << "\n ";
  }

  DumpAbstractActor(actor, ofs);

  ofs << "\t\tis_need_update_output_size:" << actor->is_need_update_output_size() << "\n ";
  ofs << "\n";
}

void DumpFusionActor(const FusionActor *actor, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor);
  ofs << "\tactor_name:" << actor->GetAID().Name() << "\tactor_id:" << actor->actor_id() << "\n";
  ofs << "\t\tsub actors:" << actor->sub_actors().size() << "\n";
  for (auto &sub_actor : actor->sub_actors()) {
    ofs << "\t\t\tsub_actor_name:" << sub_actor.first << "\n";
  }

  DumpAbstractActor(actor, ofs);

  if (actor->real_input_data().size() > 0) {
    ofs << "\t\treal_input_data:" << actor->real_input_data().size() << "\n";
    for (auto &real_input_data : actor->real_input_data()) {
      MS_EXCEPTION_IF_NULL(real_input_data.first);
      ofs << "\t\t\treal_input_data_actor:" << real_input_data.first->GetAID().Name()
          << "\tinput_index:" << real_input_data.second << "\n";
    }
  }

  if (actor->real_input_controls().size() > 0) {
    ofs << "\t\treal_input_controls:" << actor->real_input_controls().size() << "\n";
    for (auto &batch_real_input_control : actor->real_input_controls()) {
      ofs << "\t\t\torigin_input_control_actor:" << batch_real_input_control.first << "\n";
      for (auto &real_input_control : batch_real_input_control.second) {
        MS_EXCEPTION_IF_NULL(real_input_control);
        ofs << "\t\t\t\tto_real_control_actor:" << real_input_control->GetAID().Name() << "\n";
      }
    }
  }

  ofs << "\n";
}

void DumpFormalParameterDeviceTensor(const ControlActor *actor, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor);
  const auto &formal_parameter_device_tensors = actor->ref_formal_parameter_device_tensors();
  if (!formal_parameter_device_tensors.empty()) {
    ofs << "\t\tref_formal_parameter_device_tensors:" << formal_parameter_device_tensors.size() << "\n ";
    for (const auto &formal_parameter_device_tensor : formal_parameter_device_tensors) {
      for (const auto &device_tensor : formal_parameter_device_tensor.second) {
        MS_EXCEPTION_IF_NULL(device_tensor);
        auto ref_node = device_tensor->GetNodeIndex();
        MS_EXCEPTION_IF_NULL(ref_node.first);
        ofs << "\t\t\tref_position:" << formal_parameter_device_tensor.first
            << "\tref_node_name:" << ref_node.first->fullname_with_scope()
            << "\tref_node_debug_name:" << ref_node.first->DebugString() << "\n";
      }
    }
  }

  const auto &ref_node_formal_parameter_device_tensors = actor->ref_node_formal_parameter_device_tensors();
  if (!ref_node_formal_parameter_device_tensors.empty()) {
    ofs << "\t\tref_node_formal_parameter_device_tensors:" << ref_node_formal_parameter_device_tensors.size() << "\n ";
    for (const auto &ref_node_formal_parameter_device_tensor : ref_node_formal_parameter_device_tensors) {
      for (const auto &device_tensor : ref_node_formal_parameter_device_tensor.second) {
        MS_EXCEPTION_IF_NULL(device_tensor);
        auto ref_node = device_tensor->GetNodeIndex();
        MS_EXCEPTION_IF_NULL(ref_node.first);
        ofs << "\t\t\tref_position:" << ref_node_formal_parameter_device_tensor.first
            << "\tref_node_name:" << ref_node.first->fullname_with_scope()
            << "\tref_node_debug_name:" << ref_node.first->DebugString() << "\n";
      }
    }
  }
}

void DumpControlActor(const ControlActor *actor, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor);
  DumpAbstractActor(actor, ofs);
  const auto &local_partials = actor->local_partials();
  if (local_partials.size() > 0) {
    ofs << "\t\tlocal partial num:" << local_partials.size() << "\n ";
    for (const auto &local_partial : local_partials) {
      // Skip the dead node partial.
      MS_EXCEPTION_IF_NULL(local_partial.second);
      if (local_partial.second->func_graph_ == nullptr) {
        continue;
      }
      ofs << "\t\t\tlocal partial index:" << local_partial.first
          << "\tgraph:" << local_partial.second->func_graph_->ToString()
          << "\tparameter num:" << local_partial.second->device_tensors_.size() << "\n";
    }
  }

  if (actor->input_partial_arrow_aids().size() > 0) {
    ofs << "\t\tinput_partial_arrow_actor:" << actor->input_partial_arrow_aids().size() << "\n ";
    for (const auto &input_partial_arrow_aid : actor->input_partial_arrow_aids()) {
      ofs << "\t\t\tfrom_actor_name:" << input_partial_arrow_aid.Name() << "\n";
    }
  }

  if (actor->input_branch_id_arrow_aids().size() > 0) {
    ofs << "\t\tinput_branch_id_arrow_actor:" << actor->input_branch_id_arrow_aids().size() << "\n ";
    for (const auto &input_branch_id_arrow_aid : actor->input_branch_id_arrow_aids()) {
      ofs << "\t\t\tfrom_actor_name:" << input_branch_id_arrow_aid.Name() << "\n";
    }
  }

  const auto &output_partial_arrows = actor->output_partial_arrows();
  if (output_partial_arrows.size() > 0) {
    ofs << "\t\toutput_partial_arrows:" << output_partial_arrows.size() << "\n ";
    for (const auto &partial_arrow : output_partial_arrows) {
      MS_EXCEPTION_IF_NULL(partial_arrow);
      ofs << "\t\t\tfrom_output_index:" << partial_arrow->from_output_index_
          << "\tto_actor_name:" << partial_arrow->to_op_id_.Name()
          << "\tto_input_index:" << partial_arrow->to_input_index_ << "\n";
    }
  }

  const auto &output_branch_id_arrows = actor->output_branch_id_arrows();
  if (output_branch_id_arrows.size() > 0) {
    ofs << "\t\toutput_branch_id_arrows:" << output_branch_id_arrows.size() << "\n ";
    for (const auto &aid : output_branch_id_arrows) {
      ofs << "\t\t\tto_actor_name:" << aid.Name() << "\n";
    }
  }

  DumpFormalParameterDeviceTensor(actor, ofs);
}

void DumpSwitchActor(const SwitchActor *actor, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor);
  ofs << "\tactor_name:" << actor->GetAID().Name() << "\tactor_id:" << actor->actor_id() << '\n';
  DumpControlActor(actor, ofs);
}

void DumpGatherActor(const GatherActor *actor, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor);
  ofs << "\tactor_name:" << actor->GetAID().Name() << "\tactor_id:" << actor->actor_id() << '\n';
  ofs << "\t\tbranch id:" << actor->branch_id() << '\n';
  DumpControlActor(actor, ofs);

  ofs << "\t\toutput index:" << '\n';
  const auto &dynamic_len_index = actor->dynamic_len_index();
  for (const auto &func_to_index : dynamic_len_index) {
    const auto &func_graph = func_to_index.first;
    MS_EXCEPTION_IF_NULL(func_graph);
    ofs << "\t\t\tfunc_graph:" << func_graph->ToString() << '\n';
    const auto &index_list = func_to_index.second;
    for (size_t i = 0; i < index_list.size(); ++i) {
      ofs << "\t\t\t\treal index:" << i << "  is dynamic len:" << index_list[i].second << " relative index:";
      for (const auto &index : index_list[i].first) {
        ofs << index << " ";
      }
      ofs << '\n';
    }
  }

  const auto &output_data_with_branch_id_arrows = actor->output_data_with_branch_id_arrows();
  if (output_data_with_branch_id_arrows.size() > 0) {
    ofs << "\t\toutput_data_with_branch_id_arrows:" << output_data_with_branch_id_arrows.size() << "\n ";
    for (const auto &output_data_with_branch_id_arrow : output_data_with_branch_id_arrows) {
      MS_EXCEPTION_IF_NULL(output_data_with_branch_id_arrow.first);
      ofs << "\t\t\tbranch funcgraph:" << output_data_with_branch_id_arrow.first->ToString() << "\n";
      for (const auto &arrow : output_data_with_branch_id_arrow.second) {
        ofs << "\t\t\t\tto actor:" << arrow << "\n";
      }
    }
  }
}

void DumpEntranceActor(const EntranceActor *actor, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor);
  ofs << "\tactor_name:" << actor->GetAID().Name() << "\tactor_id:" << actor->actor_id() << '\n';
  DumpControlActor(actor, ofs);

  if (actor->loop_body_input_control_arrow_aids().size() > 0) {
    ofs << "\t\tinput_loop_body_control_arrow_actors:" << actor->loop_body_input_control_arrow_aids().size() << "\n ";
    for (const auto &loop_body_input_control_arrow_aid : actor->loop_body_input_control_arrow_aids()) {
      ofs << "\t\t\tfrom_actor_name:" << loop_body_input_control_arrow_aid.Name() << "\n";
    }
  }
}

void DumpExitActor(const ExitActor *actor, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor);
  ofs << "\tactor_name:" << actor->GetAID().Name() << "\tactor_id:" << actor->actor_id() << '\n';
  DumpControlActor(actor, ofs);

  ofs << "\t\toutput index:" << '\n';
  const auto &dynamic_len_index = actor->output_branch_dynamic_len_index();
  for (const auto &func_to_index : dynamic_len_index) {
    const auto &branch_id = func_to_index.first;
    ofs << "\t\t\tbranch_id:" << branch_id << '\n';
    const auto &index_list = func_to_index.second;
    for (size_t i = 0; i < index_list.size(); ++i) {
      ofs << "\t\t\t\treal index:" << i << "  is dynamic len:" << index_list[i].second << " relative index:";
      for (const auto &index : index_list[i].first) {
        ofs << index << " ";
      }
      ofs << '\n';
    }
  }

  const auto &output_branch_data_arrows = actor->output_branch_data_arrows();
  if (output_branch_data_arrows.size() > 0) {
    ofs << "\t\toutput_branch_data_arrows:" << output_branch_data_arrows.size() << "\n ";
    for (const auto &output_branch_data_arrow : output_branch_data_arrows) {
      ofs << "\t\t\tbranch id:" << output_branch_data_arrow.first << "\n";
      for (const auto &arrow : output_branch_data_arrow.second) {
        MS_EXCEPTION_IF_NULL(arrow);
        ofs << "\t\t\t\tfrom_output_index:" << arrow->from_output_index_
            << "\tto_actor_name:" << arrow->to_op_id_.Name() << "\tto_input_index:" << arrow->to_input_index_ << "\n";
      }
    }
  }

  const auto &output_branch_partial_arrows = actor->output_branch_partial_arrows();
  if (output_branch_partial_arrows.size() > 0) {
    ofs << "\t\toutput_branch_partial_arrows:" << output_branch_partial_arrows.size() << "\n ";
    for (const auto &output_branch_partial_arrow : output_branch_partial_arrows) {
      ofs << "\t\t\tbranch id:" << output_branch_partial_arrow.first << "\n";
      for (const auto &arrow : output_branch_partial_arrow.second) {
        MS_EXCEPTION_IF_NULL(arrow);
        ofs << "\t\t\t\tfrom_output_index:" << arrow->from_output_index_
            << "\tto_actor_name:" << arrow->to_op_id_.Name() << "\tto_input_index:" << arrow->to_input_index_ << "\n";
      }
    }
  }

  const auto &output_branch_control_arrows = actor->output_branch_control_arrows();
  if (output_branch_control_arrows.size() > 0) {
    ofs << "\t\toutput_branch_control_arrows:" << output_branch_control_arrows.size() << "\n ";
    for (const auto &output_branch_control_arrow : output_branch_control_arrows) {
      ofs << "\t\t\tbranch id:" << output_branch_control_arrow.first << "\n";
      for (const auto &arrow : output_branch_control_arrow.second) {
        ofs << "\t\t\t\tto actor:" << arrow << "\n";
      }
    }
  }

  const auto &is_need_copy_device_tensors = actor->is_need_copy_device_tensors();
  if (is_need_copy_device_tensors.size() > 0) {
    ofs << "\t\twhether_need_copy_device_tensors:" << is_need_copy_device_tensors.size() << "\n ";
    for (size_t i = 0; i < is_need_copy_device_tensors.size(); ++i) {
      ofs << "\t\t\tdevice_tensor_position:" << i << "\tis_need_copy:" << is_need_copy_device_tensors[i] << "\n";
    }
  }
}

void DumpStackActor(const StackActor *actor, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor);
  ofs << "\tactor_name:" << actor->GetAID().Name() << "\tactor_id:" << actor->actor_id() << '\n';
  ofs << "\t\tinput stack data num:" << actor->input_stack_data_num() << '\n';
  ofs << "\t\tinput stack partial num:" << actor->input_stack_partials_num() << '\n';
  ofs << "\t\tinput stack control num:" << actor->input_stack_controls_num() << '\n';
  DumpControlActor(actor, ofs);
}

void DumpSwitchActors(const std::vector<SwitchActorPtr> &actors, std::ofstream &ofs) {
  ofs << "\n\n[Switch actors:" << actors.size() << "]\n";
  for (const auto &switch_actor : actors) {
    DumpSwitchActor(switch_actor.get(), ofs);
    ofs << "\n";
  }
}

void DumpGatherActors(const std::vector<GatherActorPtr> &actors, std::ofstream &ofs) {
  ofs << "\n\n[Gather actors:" << actors.size() << "]\n";
  for (const auto &gather_actor : actors) {
    DumpGatherActor(gather_actor.get(), ofs);
    ofs << "\n";
  }
}

void DumpEntranceActors(const std::vector<EntranceActorPtr> &actors, std::ofstream &ofs) {
  ofs << "\n\n[Entrance actors:" << actors.size() << "]\n";
  for (const auto &entrance_actor : actors) {
    DumpEntranceActor(entrance_actor.get(), ofs);
    ofs << "\n";
  }
}

void DumpExitActors(const std::vector<ExitActorPtr> &actors, std::ofstream &ofs) {
  ofs << "\n\n[Exit actors:" << actors.size() << "]\n";
  for (const auto &exit_actor : actors) {
    DumpExitActor(exit_actor.get(), ofs);
    ofs << "\n";
  }
}

void DumpStackActors(const std::vector<StackActorPtr> &actors, std::ofstream &ofs) {
  ofs << "\n\n[Stack actors:" << actors.size() << "]\n";
  for (const auto &stack_actor : actors) {
    DumpStackActor(stack_actor.get(), ofs);
    ofs << "\n";
  }
}
}  // namespace

void DumpDataPrepareActor(const DataPrepareActorPtr &actor, std::ofstream &ofs) {
  ofs << "\n\n[Data prepare actor:" << (actor != nullptr ? 1 : 0) << "]\n";
  if (actor == nullptr) {
    return;
  }

  ofs << "\tactor_name:" << actor->GetAID().Name() << "\tactor_id:" << actor->actor_id() << "\n";
  DumpAbstractActor(actor.get(), ofs);

  ofs << "\t\tcontinuous_memory_nodes:" << actor->continuous_memory_nodes().size() << "\n ";
  for (const auto &iter : actor->continuous_memory_nodes()) {
    MS_EXCEPTION_IF_NULL(iter.first.first);
    MS_EXCEPTION_IF_NULL(iter.first.second);
    ofs << "\t\t\tnode_name:" << iter.first.first->fullname_with_scope()
        << "\tdevice_context:" << iter.first.second->device_context_key().ToString()
        << "\tis_input_need:" << iter.second.first << "\tis_output_need:" << iter.second.second << "\n";
  }
}

void DumpLoopCountActor(const LoopCountActorPtr &actor, std::ofstream &ofs) {
  ofs << "\n\n[Loop count actor:" << (actor != nullptr ? 1 : 0) << "]\n";
  if (actor == nullptr) {
    return;
  }

  ofs << "\tactor_name:" << actor->GetAID().Name() << "\tactor_id:" << actor->actor_id()
      << "\tloop_count:" << actor->loop_count() << "\n";
  DumpAbstractActor(actor.get(), ofs);

  ofs << "\t\t\tto_data_prepare_actor:" << actor->data_prepare_aid().Name() << "\n";
  for (auto &entrance_aid : actor->entrance_aids()) {
    ofs << "\t\t\tto_entrance_actor:" << entrance_aid.Name() << "\n";
  }
}

void DumpOutputActor(const OutputActorPtr &actor, std::ofstream &ofs) {
  ofs << "\n\n[Output actor:" << (actor != nullptr ? 1 : 0) << "]\n";
  if (actor == nullptr) {
    return;
  }

  ofs << "\tactor_name:" << actor->GetAID().Name() << "\tactor_id:" << actor->actor_id()
      << "\tloop_count:" << actor->loop_count() << "\toutputs_num:" << actor->outputs_num() << "\n";

  DumpAbstractActor(actor.get(), ofs);
}

void DumpDSActors(const std::vector<DataSourceActorPtr> &actors, std::ofstream &ofs) {
  ofs << "\n\n[Data source actors:" << actors.size() << "]\n";
  for (const auto &data_source_actor : actors) {
    DumpDSActor(data_source_actor.get(), ofs);
  }
}

void DumpKernelActors(const std::vector<KernelActorPtr> &actors, std::ofstream &ofs) {
  ofs << "\n\n[Kernel actors:" << actors.size() << "]\n";
  for (const auto &kernel_actor : actors) {
    DumpKernelActor(kernel_actor.get(), ofs);
  }
}

void DumpKernelInferActors(const std::vector<KernelInferActorPtr> &actors, std::ofstream &ofs) {
  ofs << "\n\n[Kernel infer actors:" << actors.size() << "]\n";
  for (const auto &kernel_infer_actor : actors) {
    DumpKernelActor(kernel_infer_actor.get(), ofs);
  }
}

void DumpKernelResizeActors(const std::vector<KernelResizeActorPtr> &actors, std::ofstream &ofs) {
  ofs << "\n\n[Kernel resize actors:" << actors.size() << "]\n";
  for (const auto &kernel_resize_actor : actors) {
    DumpKernelActor(kernel_resize_actor.get(), ofs);
  }
}

void DumpCustomActors(const std::vector<CustomActorPtr> &actors, std::ofstream &ofs) {
  ofs << "\n\n[Custom actors:" << actors.size() << "]\n";
  for (const auto &custom_actor : actors) {
    DumpCustomActor(custom_actor.get(), ofs);
  }
}

void DumpSwapActors(const std::vector<std::vector<MemSwapActorPtr>> &actors, std::ofstream &ofs) {
  size_t swap_actor_num = 0;
  (void)std::for_each(actors.cbegin(), actors.cend(),
                      [&swap_actor_num](const std::vector<MemSwapActorPtr> &actor) { swap_actor_num += actor.size(); });
  ofs << "\n\n[Swap actors:" << swap_actor_num << "]\n";
  for (const auto &as : actors) {
    for (const auto &swap_actor : as) {
      if (swap_actor == nullptr) {
        continue;
      }
      DumpSwapActor(swap_actor.get(), ofs);
    }
  }
}

void DumpSuperKernelActors(const std::vector<SuperKernelActorPtr> &actors, std::ofstream &ofs) {
  ofs << "\n\n[Super kernel actors:" << actors.size() << "]\n";
  for (const auto &super_kernel_actor : actors) {
    DumpSuperKernelActor(super_kernel_actor.get(), ofs);
  }
}

void DumpAnyTypeKernelActors(const std::vector<AnyTypeKernelActorPtr> &actors, std::ofstream &ofs) {
  ofs << "\n\n[Any Type kernel actors:" << actors.size() << "]\n";
  for (const auto &actor : actors) {
    DumpAnyTypeKernelActor(actor.get(), ofs);
  }
}

void DumpNoInputKernelActors(const std::vector<AbstractActorPtr> &actors, std::ofstream &ofs) {
  ofs << "\n\n[No input kernel actors:" << actors.size() << "]\n";
  for (const auto &actor : actors) {
    MS_EXCEPTION_IF_NULL(actor);
    if (actor->type() == KernelTransformType::kKernelActor) {
      auto kernel_actor = dynamic_cast<const KernelActor *>(actor.get());
      MS_EXCEPTION_IF_NULL(kernel_actor);
      DumpKernelActor(kernel_actor, ofs);
    } else if (actor->type() == KernelTransformType::kSuperKernelActor) {
      auto super_kernel_actor = dynamic_cast<const SuperKernelActor *>(actor.get());
      MS_EXCEPTION_IF_NULL(super_kernel_actor);
      DumpSuperKernelActor(super_kernel_actor, ofs);
    } else if (actor->type() == KernelTransformType::kCustomActor) {
      auto custom_actor = dynamic_cast<const CustomActor *>(actor.get());
      MS_EXCEPTION_IF_NULL(custom_actor);
      DumpCustomActor(custom_actor, ofs);
    }
  }
}

void DumpMemoryActors(const std::vector<MemoryAwareActorPtr> &actors, std::ofstream &ofs) {
  ofs << "\n\n[Memory actors:" << actors.size() << "]\n";
  for (const auto &memory_actor : actors) {
    DumpMemoryActor(memory_actor.get(), ofs);
  }
}

void DumpCopyActors(const std::vector<CopyActorPtr> &actors, std::ofstream &ofs) {
  ofs << "\n\n[Copy actors:" << actors.size() << "]\n";
  for (const auto &copy_actor : actors) {
    DumpCopyActor(copy_actor.get(), ofs);
  }
}

void DumpFusionActors(const std::vector<FusionActorPtr> &actors, std::ofstream &ofs) {
  ofs << "\n\n[Fusion actors:" << actors.size() << "]\n";
  for (const auto &fusion_actor : actors) {
    DumpFusionActor(fusion_actor.get(), ofs);
  }
}

void DumpControlActors(const ControlActorSetPtr &control_actor_set, std::ofstream &ofs) {
  if (control_actor_set == nullptr) {
    return;
  }

  ofs << "\n\n[Control actors]\n";
  DumpEntranceActors(control_actor_set->entrance_actors_, ofs);
  DumpSwitchActors(control_actor_set->switch_actors_, ofs);
  DumpGatherActors(control_actor_set->gather_actors_, ofs);
  DumpStackActors(control_actor_set->stack_actors_, ofs);
  DumpExitActors(control_actor_set->exit_actors_, ofs);
}

namespace {
std::string GetActorSubName(AbstractActor *actor) {
  MS_EXCEPTION_IF_NULL(actor);
  if (actor->type() == KernelTransformType::kCopyActor) {
    return std::string("CopyActor");
  }
  const auto &name = actor->GetAID().Name();
  std::string kernel_graph_name;
  if (actor->type() == KernelTransformType::kKernelActor) {
    const auto &kernel_actor = dynamic_cast<KernelActor *>(actor);
    if (kernel_actor != nullptr && kernel_actor->kernel() != nullptr &&
        kernel_actor->kernel()->func_graph() != nullptr) {
      kernel_graph_name = kernel_actor->kernel()->func_graph()->ToString() + ":";
    }
  }
  if (name.find("/") == std::string::npos) {
    return kernel_graph_name + name;
  }
  const auto &pos = name.find_last_of("/");
  return kernel_graph_name + name.substr(pos + 1);
}
using ActorInputMap = std::map<size_t, std::tuple<std::string, BaseShapePtr, TypePtr>>;
void AddInputActorInfo(ActorInputMap *actor_inputs, AbstractActor *input_actor, const AbstractActor *const actor,
                       const ActorInfoMap &actor_info, size_t from_index, size_t to_index) {
  MS_EXCEPTION_IF_NULL(actor_inputs);
  MS_EXCEPTION_IF_NULL(actor);
  if (actor_inputs->find(to_index) != actor_inputs->end()) {
    MS_LOG(INFO) << "Invalid index:" << to_index << " for actor:" << actor->GetAID()
                 << " input aid:" << input_actor->GetAID() << " same to:" << std::get<0>((*actor_inputs)[to_index]);
    return;
  }
  const auto &input_iter = actor_info.find(input_actor);
  if (input_iter != actor_info.end()) {
    const auto &input_name =
      "%" + std::to_string(std::get<0>(input_iter->second)) + "[" + std::to_string(from_index) + "]";
    const auto &input_shapes = std::get<1>(input_iter->second);
    const auto &input_types = std::get<2>(input_iter->second);
    auto shape =
      ((from_index < input_shapes.size() && input_shapes[from_index] != nullptr) ? input_shapes[from_index] : nullptr);
    auto type =
      ((from_index < input_types.size() && input_types[from_index] != nullptr) ? input_types[from_index] : nullptr);
    (*actor_inputs)[to_index] = {input_name, shape, type};
  } else {
    (*actor_inputs)[to_index] = {input_actor->GetAID().Name() + "[" + std::to_string(from_index) + "]", nullptr,
                                 nullptr};
  }
}

size_t GetFromIndexInHostQueueDataSourceActor(AbstractActor *input_actor, const DataArrow *const data_arrow) {
  MS_EXCEPTION_IF_NULL(input_actor);
  MS_EXCEPTION_IF_NULL(data_arrow);
  const auto &host_ds_actor = dynamic_cast<HostQueueDataSourceActor *>(input_actor);
  MS_EXCEPTION_IF_NULL(host_ds_actor);
  const auto &iter =
    std::find_if(host_ds_actor->output_data_arrows().begin(), host_ds_actor->output_data_arrows().end(),
                 [data_arrow](const auto &arrow) { return arrow.get() == data_arrow; });
  if (iter == host_ds_actor->output_data_arrows().end()) {
    MS_LOG(INFO) << "Failed to find output data arrow from index" << data_arrow->from_output_index_
                 << " to aid:" << data_arrow->to_op_id_ << " to index:" << data_arrow->to_op_id_
                 << " in host data source actor:" << input_actor->GetAID();
    return IntToSize(data_arrow->from_output_index_);
  }
  size_t node_index = LongToSize(iter - host_ds_actor->output_data_arrows().begin());
  if (node_index >= host_ds_actor->output_data_nodes().size()) {
    MS_LOG(INFO) << "Invalid node index:" << node_index << " total:" << host_ds_actor->output_data_nodes().size()
                 << " for actor:" << input_actor->GetAID();
    return IntToSize(data_arrow->from_output_index_);
  }
  return host_ds_actor->FetchNodePosition(
    {host_ds_actor->output_data_nodes()[node_index], IntToSize(data_arrow->from_output_index_)});
}

size_t GetFromIndexInSuperKernelActor(AbstractActor *input_actor, const AbstractActor *const actor,
                                      const DataArrow *const data_arrow) {
  MS_EXCEPTION_IF_NULL(input_actor);
  MS_EXCEPTION_IF_NULL(data_arrow);
  if (input_actor->output_data_arrows().size() != input_actor->output_data_nodes().size()) {
    MS_LOG(INFO) << "For actor:" << input_actor->GetAID()
                 << " output arrow size:" << input_actor->output_data_arrows().size()
                 << " not equal to node size:" << input_actor->output_data_nodes().size();
    return IntToSize(data_arrow->from_output_index_);
  }
  const auto &super_kernel_actor = dynamic_cast<SuperKernelActor *>(input_actor);
  MS_EXCEPTION_IF_NULL(super_kernel_actor);
  const auto &graph = super_kernel_actor->graph();
  if (graph == nullptr) {
    MS_LOG(INFO) << "Failed to get graph in actor:" << input_actor->GetAID();
    return IntToSize(data_arrow->from_output_index_);
  }
  const auto &output_pairs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
  const auto &data_iter =
    std::find_if(input_actor->output_data_arrows().begin(), input_actor->output_data_arrows().end(),
                 [data_arrow](const auto &arrow) { return arrow.get() == data_arrow; });
  if (data_iter == input_actor->output_data_arrows().end()) {
    MS_LOG(INFO) << "Failed to find output data arrow from index" << data_arrow->from_output_index_
                 << " to aid:" << data_arrow->to_op_id_ << " to index:" << data_arrow->to_op_id_
                 << " in host data source actor:" << input_actor->GetAID();
    return IntToSize(data_arrow->from_output_index_);
  }
  size_t node_index = LongToSize(data_iter - input_actor->output_data_arrows().begin());
  if (node_index >= input_actor->output_data_nodes().size() ||
      input_actor->output_data_nodes()[node_index] == nullptr) {
    MS_LOG(INFO) << "Invalid node index:" << node_index << " total:" << input_actor->output_data_nodes().size()
                 << " for actor:" << input_actor->GetAID() << " graph:" << graph->ToString();
    return IntToSize(data_arrow->from_output_index_);
  }
  const auto &output_iter =
    std::find(output_pairs.begin(), output_pairs.end(),
              std::make_pair(input_actor->output_data_nodes()[node_index], IntToSize(data_arrow->from_output_index_)));
  if (output_iter == output_pairs.end()) {
    MS_LOG(INFO) << "Failed to find output node:" << input_actor->output_data_nodes()[node_index]->fullname_with_scope()
                 << " in graph:" << graph->ToString() << " for actor:" << actor->GetAID();
    return IntToSize(data_arrow->from_output_index_);
  }
  return output_iter - output_pairs.begin();
}

void FetchInputActor(std::string input_aid, ActorInputMap *actor_inputs, const AbstractActor *const actor,
                     const ActorInfoMap &actor_info, const DataArrow *const data_arrow) {
  MS_EXCEPTION_IF_NULL(data_arrow);
  size_t to_index = IntToSize(data_arrow->to_input_index_);
  size_t from_index = IntToSize(data_arrow->from_output_index_);
  auto input_actor = FetchActor(input_aid);
  if (input_actor == nullptr) {
    MS_LOG(INFO) << "Failed to fetch input actor:" << input_aid;
    return;
  }
  if (input_actor->type() == KernelTransformType::kHostDataSourceActor) {
    from_index = GetFromIndexInHostQueueDataSourceActor(input_actor, data_arrow);
  } else if (input_actor->type() == KernelTransformType::kSuperKernelActor) {
    from_index = GetFromIndexInSuperKernelActor(input_actor, actor, data_arrow);
  } else if (actor->type() != KernelTransformType::kFusionActor &&
             data_arrow->to_op_id_.Name().find(kFusionActorNameSuffix) != std::string::npos) {
    const auto &fusion_aid = data_arrow->to_op_id_.Name();
    const auto &from_actor = FetchActor(fusion_aid);
    if (from_actor == nullptr) {
      MS_LOG(INFO) << "Failed to fetch actor:" << fusion_aid;
      return;
    }
    input_actor = from_actor;
    from_index = to_index;
    const auto &fusion_actor = dynamic_cast<FusionActor *>(from_actor);
    MS_EXCEPTION_IF_NULL(fusion_actor);
    const auto &real_input_data = fusion_actor->real_input_data();
    if (to_index >= real_input_data.size()) {
      MS_LOG(INFO) << "Failed to find to index in fusion actor:" << input_aid << " for actor:" << actor->GetAID()
                   << " to index:" << to_index;
      return;
    }
    to_index = real_input_data[to_index].second;
  }
  AddInputActorInfo(actor_inputs, input_actor, actor, actor_info, from_index, to_index);
}

void FetchInputDeviceTensorStore(const AnfNodePtr &key, size_t index, const AbstractActor *const actor,
                                 ActorInputMap *actor_inputs) {
  MS_EXCEPTION_IF_NULL(key);
  MS_EXCEPTION_IF_NULL(actor_inputs);
  std::string input_name = "%";
  if (key->isa<Parameter>()) {
    input_name += key->DebugString(0);
  } else if (key->isa<ValueNode>()) {
    const auto &value_node = key->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    if (value_node->value() == nullptr) {
      input_name += value_node->DebugString();
    } else {
      if (value_node->value()->isa<Scalar>()) {
        input_name = value_node->value()->DumpText();
      } else {
        input_name = value_node->value()->ToString();
      }
    }
  } else {
    input_name += key->DebugString();
  }
  if (actor_inputs->find(index) != actor_inputs->end()) {
    MS_LOG(INFO) << "Invalid index:" << index << " for actor:" << actor->GetAID() << " input aid:" << key->DebugString()
                 << " same to:" << std::get<0>((*actor_inputs)[index]);
    return;
  }
  (*actor_inputs)[index] = {input_name, key->Shape() == nullptr ? nullptr : key->Shape(),
                            key->Type() == nullptr ? nullptr : key->Type()};
}

void FetchInputForHostQueueDSActor(AbstractActor *actor, ActorInputMap *actor_inputs) {
  MS_EXCEPTION_IF_NULL(actor);
  MS_EXCEPTION_IF_NULL(actor_inputs);
  const auto &ds_actor = dynamic_cast<HostQueueDataSourceActor *>(actor);
  MS_EXCEPTION_IF_NULL(ds_actor);
  for (size_t i = 0; i < ds_actor->data_nodes().size(); ++i) {
    const auto &node_pair = ds_actor->data_nodes()[i];
    if (node_pair.first == nullptr) {
      (*actor_inputs)[i] = {"null", nullptr, nullptr};
      continue;
    }
    auto device_address = AnfAlgo::GetMutableOutputAddr(node_pair.first, node_pair.second, false);
    if (device_address == nullptr || device_address->kernel_tensor() == nullptr) {
      (*actor_inputs)[i] = {"null", nullptr, nullptr};
      continue;
    }
    const auto &kernel_tensor = device_address->kernel_tensor();
    (*actor_inputs)[i] = {node_pair.first->DebugString(0),
                          kernel_tensor->GetShape() == nullptr ? nullptr : kernel_tensor->GetShape(),
                          kernel_tensor->GetType() == nullptr ? nullptr : kernel_tensor->GetType()};
  }
}

void FetchInputData(AbstractActor *actor, ActorInputMap *actor_inputs, ActorInfoMap *actor_info) {
  MS_EXCEPTION_IF_NULL(actor);
  MS_EXCEPTION_IF_NULL(actor_info);
  MS_EXCEPTION_IF_NULL(actor_inputs);
  for (const auto &pair : actor->input_data_arrow_aids()) {
    if (pair.second == nullptr) {
      MS_LOG(INFO) << "Invalid input data arrow for actor:" << actor->GetAID() << " input actor:" << pair.first;
      continue;
    }
    FetchInputActor(pair.first.Name(), actor_inputs, actor, *actor_info, pair.second);
  }

  for (const auto &pair : actor->device_tensor_store_keys()) {
    MS_EXCEPTION_IF_NULL(pair.second);
    FetchInputDeviceTensorStore(pair.second, pair.first, actor, actor_inputs);
  }

  if (actor->type() == KernelTransformType::kHostDataSourceActor) {
    FetchInputForHostQueueDSActor(actor, actor_inputs);
  }
}

void FetchOutputInfo(AbstractActor *actor, std::vector<BaseShapePtr> *output_shapes, std::vector<TypePtr> *output_types,
                     const ActorInputMap &actor_inputs) {
  MS_EXCEPTION_IF_NULL(actor);
  MS_EXCEPTION_IF_NULL(output_shapes);
  MS_EXCEPTION_IF_NULL(output_types);
  if (actor->type() == KernelTransformType::kKernelActor) {
    const auto &kernel_actor = dynamic_cast<KernelActor *>(actor);
    if (kernel_actor != nullptr && kernel_actor->kernel() != nullptr &&
        kernel_actor->kernel()->kernel_info() != nullptr) {
      const auto &kernel_info = dynamic_cast<KernelInfo *>(kernel_actor->kernel()->kernel_info());
      MS_EXCEPTION_IF_NULL(kernel_info);
      const auto &device_addresses = kernel_info->output_address_list();
      for (const auto &device_address : device_addresses) {
        if (device_address != nullptr && device_address->kernel_tensor() != nullptr) {
          output_shapes->emplace_back(device_address->kernel_tensor()->GetShape());
          output_types->emplace_back(device_address->kernel_tensor()->GetType());
        }
      }
    }
  } else if (actor->type() == KernelTransformType::kSuperKernelActor) {
    if (actor->output_data_arrows().size() != actor->output_data_nodes().size()) {
      MS_LOG(INFO) << "For actor:" << actor->GetAID() << " output arrow size:" << actor->output_data_arrows().size()
                   << " not equal to node size:" << actor->output_data_nodes().size();
      return;
    }
    const auto &super_kernel_actor = dynamic_cast<SuperKernelActor *>(actor);
    MS_EXCEPTION_IF_NULL(super_kernel_actor);
    const auto &graph = super_kernel_actor->graph();
    if (graph == nullptr) {
      MS_LOG(INFO) << "Failed to get graph in actor:" << actor->GetAID();
      return;
    }
    const auto &output_pairs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
    for (size_t i = 0; i < output_pairs.size(); ++i) {
      const auto &output_pair = output_pairs[i];
      MS_EXCEPTION_IF_NULL(output_pair.first);
      const auto &node_index = common::AnfAlgo::VisitKernelWithReturnType(output_pair.first, output_pair.second, false);
      MS_EXCEPTION_IF_NULL(node_index.first);
      auto device_address = AnfAlgo::GetMutableOutputAddr(node_index.first, node_index.second, false);
      if (device_address == nullptr || device_address->kernel_tensor() == nullptr) {
        MS_LOG(INFO) << "For actor:" << actor->GetAID() << " output node:" << node_index.first->fullname_with_scope()
                     << " has invalid device address:" << device_address;
        output_shapes->emplace_back(nullptr);
        output_types->emplace_back(nullptr);
        continue;
      }
      output_shapes->emplace_back(device_address->kernel_tensor()->GetShape());
      output_types->emplace_back(device_address->kernel_tensor()->GetType());
    }
  } else {
    for_each(actor_inputs.begin(), actor_inputs.end(), [output_shapes, output_types](const auto &pair) {
      output_shapes->emplace_back(std::get<1>(pair.second));
      output_types->emplace_back(std::get<2>(pair.second));
    });
  }
}

void DumpActorInfo(AbstractActor *actor, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor);
  if (actor->type() == KernelTransformType::kKernelActor || actor->type() == KernelTransformType::kSuperKernelActor ||
      actor->type() == KernelTransformType::kOutputActor) {
    ofs << "\t# device context: ";
    std::for_each(actor->device_contexts().begin(), actor->device_contexts().end(), [&ofs](const auto &device_context) {
      ofs << (device_context == nullptr ? "null" : device_context->device_context_key().ToString()) << " ";
    });
    ofs << "\n";
  } else if (actor->type() == KernelTransformType::kCopyActor) {
    if (actor->device_contexts().size() >= 2 && actor->device_contexts()[0] != nullptr &&
        actor->device_contexts()[1] != nullptr) {
      ofs << "\t# device context: " << actor->device_contexts()[0]->device_context_key().ToString() << " -> "
          << actor->device_contexts()[1]->device_context_key().ToString() << "\n";
    }
  }
}
}  // namespace

std::vector<AbstractActor *> TopoSortForActor(AbstractActor *root) {
  std::vector<AbstractActor *> actors;
  auto seen = NewSeenGeneration();
  std::deque<AbstractActor *> todo;
  (void)todo.emplace_back(root);

  mindspore::HashMap<AbstractActor *, SeenNum> seen_map;
  mindspore::HashMap<AbstractActor *, SeenNum> extra_seen_map;
  seen_map[root] = 0;
  extra_seen_map[root] = 0;
  while (!todo.empty()) {
    AbstractActor *actor = todo.back();
    if (extra_seen_map[actor] == seen) {
      todo.pop_back();
      continue;
    }
    if (seen_map[actor] == seen) {
      extra_seen_map[actor] = seen;
      (void)actors.emplace_back(actor);
      todo.pop_back();
      continue;
    }
    seen_map[actor] = seen;
    std::vector<std::string> input_aids;
    std::for_each(
      actor->input_data_arrow_aids().begin(), actor->input_data_arrow_aids().end(),
      [&input_aids, actor](const auto &pair) {
        input_aids.emplace_back((actor->type() != KernelTransformType::kFusionActor && pair.second != nullptr &&
                                 pair.second->to_op_id_.Name().find(kFusionActorNameSuffix) != std::string::npos)
                                  ? pair.second->to_op_id_.Name()
                                  : pair.first.Name());
      });
    std::for_each(
      actor->input_control_arrow_aids().begin(), actor->input_control_arrow_aids().end(),
      [&input_aids, actor](const auto &pair) {
        input_aids.emplace_back((actor->type() != KernelTransformType::kFusionActor && pair.second != nullptr &&
                                 pair.second->to_op_id_.Name().find(kFusionActorNameSuffix) != std::string::npos)
                                  ? pair.second->to_op_id_.Name()
                                  : pair.first.Name());
      });
    for (auto aid : input_aids) {
      const auto &input_actor = FetchActor(aid);
      if (input_actor == nullptr) {
        MS_LOG(INFO) << "Failed to get actor:" << aid;
        continue;
      }
      if (seen_map.find(input_actor) == seen_map.end()) {
        seen_map[input_actor] = 0;
      }
      if (extra_seen_map.find(input_actor) == extra_seen_map.end()) {
        extra_seen_map[input_actor] = 0;
      }
      if (extra_seen_map[input_actor] == seen) {
        continue;
      }
      if (seen_map[input_actor] != seen) {
        (void)todo.emplace_back(input_actor);
        continue;
      }
      // Loop count has a cycle input and skip it.
      if (input_actor != root && input_actor->type() != KernelTransformType::kLoopCountActor) {
        MS_LOG(EXCEPTION) << "Actor cycle exists in actor:" << input_actor->GetAID();
      }
    }
  }
  return actors;
}

void DumpActorInfo(AbstractActor *actor, size_t index, ActorInfoMap *actor_info, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor);
  MS_EXCEPTION_IF_NULL(actor_info);
  ActorInputMap actor_inputs;
  FetchInputData(actor, &actor_inputs, actor_info);

  std::vector<BaseShapePtr> output_shapes;
  std::vector<TypePtr> output_types;
  FetchOutputInfo(actor, &output_shapes, &output_types, actor_inputs);
  (*actor_info)[actor] = {index, output_shapes, output_types};

  // Dump input data.
  ofs << "%" << index << " = " << GetActorSubName(actor) << "(";
  for (const auto &pair : actor_inputs) {
    ofs << std::get<0>(pair.second);
    if (pair.first < actor_inputs.size() - 1) {
      ofs << ", ";
    }
  }

  // Dump input control.
  if (!actor->input_control_arrow_aids().empty()) {
    ofs << ") op control(";
    for (const auto &pair : actor->input_control_arrow_aids()) {
      auto aid = pair.first.Name();
      if (actor->type() != KernelTransformType::kFusionActor && pair.second != nullptr &&
          pair.second->to_op_id_.Name().find(kFusionActorNameSuffix) != std::string::npos) {
        aid = pair.second->to_op_id_.Name();
      }
      const auto &input_actor = FetchActor(aid);
      ofs << "%";
      if ((*actor_info).find(input_actor) != (*actor_info).end()) {
        ofs << std::get<0>((*actor_info)[input_actor]);
      } else {
        ofs << aid;
      }
      if (pair != actor->input_control_arrow_aids().back()) {
        ofs << ", ";
      }
    }
  }
  ofs << ")\n";

  if (actor->type() == KernelTransformType::kDataPrepareActor) {
    return;
  }
  // Dump device context;
  DumpActorInfo(actor, ofs);

  // Dump output info.
  std::string shape = "\t# shape : ";
  std::string type = "\t# type : ";
  for (const auto &pair : actor_inputs) {
    shape = shape + "<" + (std::get<1>(pair.second) == nullptr ? "null" : std::get<1>(pair.second)->ToString()) + "> ";
    type = type + "<" + (std::get<2>(pair.second) == nullptr ? "null" : std::get<2>(pair.second)->ToString()) + "> ";
  }
  shape += "-> ";
  type += "-> ";
  for_each(output_shapes.begin(), output_shapes.end(), [&shape](const auto &shape_ptr) {
    shape = shape + "<" + (shape_ptr == nullptr ? "null" : shape_ptr->ToString()) + "> ";
  });
  for_each(output_types.begin(), output_types.end(), [&type](const auto &type_ptr) {
    type = type + "<" + (type_ptr == nullptr ? "null" : type_ptr->ToString()) + "> ";
  });
  ofs << shape << "\n" << type << "\n";
}
}  // namespace runtime
}  // namespace mindspore
