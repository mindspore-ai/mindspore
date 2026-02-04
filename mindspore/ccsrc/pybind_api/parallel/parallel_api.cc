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
#include "pybind_api/parallel/parallel_api.h"
#include <memory>
#include "pybind11/pybind11.h"
#include "include/utils/parallel_context.h"
#include "include/frontend/parallel/parallel_strategy_checkpoint.h"
#include "include/frontend/parallel/device_manager.h"
#include "include/utils/offload_context.h"
#include "include/frontend/parallel/costmodel_context.h"
#include "include/frontend/parallel/tensor_layout/tensor_transform.h"

using ParallelContext = mindspore::parallel::ParallelContext;
using StrategyInfo = mindspore::parallel::StrategyInfo;
using StrategyLayout = mindspore::parallel::StrategyLayout;
using ParallelCommManager = mindspore::parallel::ParallelCommManager;
using CostModelContext = mindspore::parallel::CostModelContext;
using TensorTransform = mindspore::parallel::TensorTransform;
using OffloadContext = mindspore::OffloadContext;

namespace py = pybind11;
namespace mindspore {
void RegParallelModule(py::module *m) {
  (void)py::class_<TensorTransform, std::shared_ptr<TensorTransform>>(*m, "TensorTransform")
    .def_static("get_instance", &TensorTransform::GetInstance, "Get tensor_transform instance.")
    .def("transform_tensor_sharding", &TensorTransform::TransformOperators, "Transform the tensor sharding.");
  MS_LOG(INFO) << "Start ParallelContext...";
  (void)py::class_<ParallelContext, std::shared_ptr<ParallelContext>>(*m, "AutoParallelContext")
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
    .def("get_zero3", &ParallelContext::zero3, "Get zero3.")
    .def("set_zero3", &ParallelContext::set_zero3, "Set zero3.")
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
    .def("set_dump_local_norm", &ParallelContext::set_dump_local_norm, "Set dump_local_norm.")
    .def("get_dump_local_norm", &ParallelContext::dump_local_norm, "Get dump_local_norm.")
    .def("set_dump_local_norm_path", &ParallelContext::set_dump_local_norm_path, "Set dump_local_norm_path.")
    .def("get_dump_local_norm_path", &ParallelContext::dump_local_norm_path, "Get dump_local_norm_path.")
    .def("set_dump_device_local_norm", &ParallelContext::set_dump_device_local_norm, "Set dump_device_local_norm.")
    .def("get_dump_device_local_norm", &ParallelContext::dump_device_local_norm, "Get dump_device_local_norm.")
    .def("set_pipeline_interleave", &ParallelContext::set_pipeline_interleave, "Set pipeline interleave.")
    .def("get_pipeline_interleave", &ParallelContext::pipeline_interleave, "Get pipeline interleave.")
    .def("set_pipeline_scheduler", &ParallelContext::set_pipeline_scheduler, "Set pipeline scheduler.")
    .def("get_pipeline_scheduler", &ParallelContext::pipeline_scheduler, "Get pipeline scheduler.")
    .def("set_full_batch", &ParallelContext::set_full_batch, "Set whether load full batch on each device.")
    .def("get_full_batch", &ParallelContext::full_batch, "Get whether load full batch on each device.")
    .def("get_full_batch_is_set", &ParallelContext::full_batch_is_set, "Get whether attr full_batch is set.")
    .def("set_dataset_strategy", &ParallelContext::set_dataset_strategy, "Set dataset sharding strategy.")
    .def("set_dataset_layout", &ParallelContext::set_dataset_layout, "Set dataset sharding layout.")
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
    .def("set_init_param_in_compile", &ParallelContext::set_init_param_in_compile, "Set init param in compile.")
    .def("get_init_param_in_compile", &ParallelContext::init_param_in_compile, "Get the init param in compile.")
    .def("set_auto_parallel_new_interface", &ParallelContext::set_auto_parallel_new_interface, "Set interface flag.")
    .def("get_auto_parallel_new_interface", &ParallelContext::auto_parallel_new_interface, "Get interface flag.")
    .def("reset", &ParallelContext::Reset, "Reset auto parallel context.");
  MS_LOG(INFO) << "Start StrategyInfo...";
  (void)py::class_<StrategyInfo, std::shared_ptr<StrategyInfo>>(*m, "StrategyInfo")
    .def(py::init<>())
    .def("__str__", &StrategyInfo::ToString)
    .def_property_readonly("dev_matrix", &StrategyInfo::dev_matrix)
    .def_property_readonly("tensor_map", &StrategyInfo::tensor_map)
    .def_property_readonly("tensor_shape", &StrategyInfo::tensor_shape)
    .def_property_readonly("tensor_type", &StrategyInfo::tensor_type)
    .def_property_readonly("field", &StrategyInfo::field)
    .def_property_readonly("opt_weight_shard_step", &StrategyInfo::opt_weight_shard_step)
    .def_property_readonly("opt_weight_shard_size", &StrategyInfo::opt_weight_shard_size)
    .def_property_readonly("param_split_shape", &StrategyInfo::param_split_shape)
    .def_property_readonly("indices_offset", &StrategyInfo::indices_offset)
    .def_property_readonly("stage_id", &StrategyInfo::stage_id)
    .def_property_readonly("pipeline_stages", &StrategyInfo::pipeline_stages)
    .def_property_readonly("rank_list", &StrategyInfo::rank_list);
  MS_LOG(INFO) << "Start StrategyLayout...";
  (void)py::class_<StrategyLayout, std::shared_ptr<StrategyLayout>>(*m, "StrategyLayout")
    .def_static("get_instance", &StrategyLayout::GetInstance, "Get global_strategy_layout instance.")
    .def("enable_save_strategy_online", &StrategyLayout::enable_save_strategy_online,
         "Set save_strategy_online of global network.")
    .def("global_network_layout", &StrategyLayout::global_network_layout, "Get global network rank param strategy.")
    .def("local_network_layout", &StrategyLayout::local_network_layout, "Get local network rank param strategy.")
    .def("clear_strategy_metadata", &StrategyLayout::clear_strategy_metadata, "Clear global network strategy info.");
  MS_LOG(INFO) << "Start ParallelCommManager...";
  (void)py::class_<ParallelCommManager, std::shared_ptr<ParallelCommManager>>(*m, "ParallelCommManager")
    .def_static("get_instance", &ParallelCommManager::GetInstance, "Get parallel comm manager instance.")
    .def("set_hccl_groups", &ParallelCommManager::SetHcclGroups, "Record hccl_groups created by rank_list.")
    .def("hccl_groups", &ParallelCommManager::HcclGroups, "Weather hccl_groups has been created by rank_list.");
  MS_LOG(INFO) << "Start CostModelContext...";
  (void)py::class_<CostModelContext, std::shared_ptr<CostModelContext>>(*m, "CostModelContext")
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
  (void)py::class_<OffloadContext, std::shared_ptr<OffloadContext>>(*m, "OffloadContext")
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
}
}  // namespace mindspore
