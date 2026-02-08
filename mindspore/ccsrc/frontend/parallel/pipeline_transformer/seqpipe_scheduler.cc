
/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <memory>
#include <queue>
#include <stack>

#include "frontend/parallel/pipeline_transformer/seqpipe_scheduler.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/node_check.h"
#include "include/utils/utils.h"
#include "ir/anf.h"
#include "ir/func_graph_flag.h"
#include "ir/graph_utils.h"
#include "primitive/array_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/other_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace parallel {
constexpr auto kMaskCache = "mask_cache";
constexpr auto kSeqChunk = "seq_chunk";
constexpr auto kSend = "Send";
constexpr auto kReceive = "Receive";
constexpr auto kCell = "Cell";

bool CmpInsideMicroSeqpipe(const Border &b_i, const Border &b_j) {
  auto order_i1 = b_i.border->GetPrimalAttr(ORDER);
  auto order_j1 = b_j.border->GetPrimalAttr(ORDER);
  MS_EXCEPTION_IF_NULL(order_i1);
  MS_EXCEPTION_IF_NULL(order_j1);
  return (GetValue<int64_t>(order_i1) < GetValue<int64_t>(order_j1));
}

std::pair<Border, Border> SeqpipeScheduler::SeqpipeBorder(const std::vector<Border> &borders, int64_t seq_chunk,
                                                          int64_t chunk, int64_t micro) {
  std::vector<Border> candidates;
  std::copy_if(borders.begin(), borders.end(), std::back_inserter(candidates),
               [&chunk, &micro, &seq_chunk](const auto &b) {
                 return (b.chunk == chunk && b.micro == micro && b.seq_chunk == seq_chunk);
               });
  if (candidates.empty()) {
    MS_LOG(EXCEPTION) << "Can not find border of the pipeline, stage id:" << stage_ << ", chunk:" << chunk
                      << ", micro:" << micro << ", seq_chunk:" << seq_chunk;
  }
  if (candidates.size() > 1) {
    std::sort(candidates.begin(), candidates.end(), CmpInsideMicroSeqpipe);
    for (size_t index = 0; index < candidates.size() - 1; ++index) {
      auto prior = candidates[index];
      auto last = candidates[index + 1];
      ControlOrder(prior, last);
    }
  }
  return std::make_pair(candidates.front(), candidates.back());
}

std::vector<std::vector<std::vector<BorderPair>>> SeqpipeScheduler::BorderMap(const std::vector<Border> &borders) {
  size_t chunk_num = LongToSize(chunk_num_);
  size_t micro_size = LongToSize(micro_size_);
  size_t seq_chunk_size = LongToSize(seq_chunk_size_);
  std::vector<std::vector<std::vector<BorderPair>>> sorted_borders(
    chunk_num, std::vector<std::vector<BorderPair>>(micro_size, std::vector<BorderPair>(seq_chunk_size)));
  for (size_t chunk = 0; chunk < chunk_num; ++chunk) {
    for (size_t micro = 0; micro < micro_size; ++micro) {
      for (size_t seq_chunk = 0; seq_chunk < seq_chunk_size; ++seq_chunk) {
        auto border = SeqpipeBorder(borders, seq_chunk, chunk, micro);
        sorted_borders[chunk][micro][seq_chunk] = border;
      }
    }
  }
  return sorted_borders;
}

void SeqpipeScheduler::ComputeCycleList() {
  size_t cycle_size = LongToSize(micro_size_ / stage_num_);
  if (micro_size_ < stage_num_) {
    cycle_list_.push_back(LongToSize(micro_size_));
    cycle_size = 1;
  } else {
    cycle_list_.resize(cycle_size, LongToSize(stage_num_));
  }
  size_t cycled_index = cycle_size * LongToSize(stage_num_);
  size_t idx = 0;
  while (cycled_index + idx < LongToSize(micro_size_)) {
    cycle_list_[idx % cycle_size] += 1;
    idx++;
  }
}

void SeqvppScheduler::ComputeCycleList() {
  if (chunk_num_ <= 1) {
    MS_LOG(EXCEPTION) << "For seq vpp scheduler, vpp should large than 1.";
  }
  if (stage_num_ < seq_chunk_size_) {
    MS_LOG(EXCEPTION) << "For seq vpp scheduler, pp should large than seq_chunk_size.";
  }
  if (micro_size_ * seq_chunk_size_ < stage_num_) {
    MS_LOG(EXCEPTION) << "For seq vpp scheduler, micro_num * seq_chunk_size should be greater than or equal to pp.";
  }
  size_t cycle_value = LongToSize(stage_num_ / seq_chunk_size_);
  size_t cycle_size = LongToSize(micro_size_ / cycle_value);
  cycle_list_.resize(cycle_size, cycle_value);

  size_t cycled_index = cycle_size * cycle_value;
  size_t idx = 0;
  while (cycled_index + idx < LongToSize(micro_size_)) {
    cycle_list_[idx % cycle_size] += 1;
    idx++;
  }
}

std::vector<Triplet> SeqpipeScheduler::MakeExecuteOrder(const std::vector<size_t> &micro_index_list,
                                                        const std::vector<size_t> &chunk_index_list,
                                                        size_t warm_up_size) {
  std::vector<Triplet> excute_order;
  std::queue<size_t> micro_queue;
  std::unordered_map<size_t, std::stack<size_t>> chunk_stack;
  std::unordered_map<size_t, std::stack<size_t>> seq_chunk_stack;
  size_t chunk_num = LongToSize(chunk_num_);
  size_t micro_size = LongToSize(micro_size_);
  size_t seq_chunk_size = LongToSize(seq_chunk_size_);
  for (size_t index = 0; index < micro_size * chunk_num * seq_chunk_size; ++index) {
    size_t seq_chunk = index % seq_chunk_size;
    size_t micro = micro_index_list[index];
    size_t chunk = chunk_index_list[index];
    if (micro < micro_size) {
      micro_queue.push(micro);
      chunk_stack[micro].push(chunk);
      seq_chunk_stack[micro].push(seq_chunk);
      excute_order.push_back({seq_chunk, micro, chunk, false});
      MS_LOG(INFO) << "seq_chunk:" << seq_chunk << ", micro:" << micro << ", chunk:" << chunk << ", is fp";
    }
    if (index >= warm_up_size && !micro_queue.empty()) {
      size_t b_micro = micro_queue.front();
      micro_queue.pop();
      size_t b_chunk = chunk_stack[b_micro].top();
      chunk_stack[b_micro].pop();
      size_t b_seq_chunk = seq_chunk_stack[b_micro].top();
      seq_chunk_stack[b_micro].pop();
      excute_order.push_back({b_seq_chunk, b_micro, b_chunk, true});
      MS_LOG(INFO) << "seq_chunk:" << b_seq_chunk << ", micro:" << b_micro << ", chunk:" << b_chunk << ", is bp";
    }
  }
  while (!micro_queue.empty()) {
    size_t b_micro = micro_queue.front();
    micro_queue.pop();
    size_t b_chunk = chunk_stack[b_micro].top();
    chunk_stack[b_micro].pop();
    size_t b_seq_chunk = seq_chunk_stack[b_micro].top();
    seq_chunk_stack[b_micro].pop();
    excute_order.push_back({b_seq_chunk, b_micro, b_chunk, true});
    MS_LOG(INFO) << "seq_chunk:" << b_seq_chunk << ", micro:" << b_micro << ", chunk:" << b_chunk << ", is bp";
  }
  return excute_order;
}

std::vector<Triplet> SeqpipeScheduler::ExecuteOrder(size_t warm_up_size) {
  std::vector<size_t> micro_index_list;
  std::vector<size_t> chunk_index_list;
  size_t updated_micro_num = 0;
  for (size_t cycle_len : cycle_list_) {
    for (size_t chunk_id = 0; chunk_id < LongToSize(chunk_num_); ++chunk_id) {
      for (size_t micro_id = 0; micro_id < cycle_len; ++micro_id) {
        for (size_t seq_id = 0; seq_id < LongToSize(seq_chunk_size_); ++seq_id) {
          micro_index_list.push_back(micro_id + updated_micro_num);
          chunk_index_list.push_back(chunk_id);
        }
      }
    }
    updated_micro_num += cycle_len;
  }

  if (warm_up_size > fp_block_size_ - 1) {
    warm_up_size = fp_block_size_ - 1;
  }
  return MakeExecuteOrder(micro_index_list, chunk_index_list, warm_up_size);
}

std::vector<Triplet> SeqpipeScheduler::FpBpExecuteOrder(bool is_bp) {
  std::vector<Triplet> result;
  for (size_t index = 0; index < execute_order_.size(); ++index) {
    if (execute_order_[index].is_bp == is_bp) {
      result.push_back(execute_order_[index]);
    }
  }
  return result;
}

void SeqpipeScheduler::GetBorderNode() {
  PipelineScheduler::GetBorderNode();
  GetCleanAssigns();
  auto set_seq_chunk = [&](std::vector<Border> borders, bool is_bp) {
    for (auto &node : borders) {
      if (node.border->HasPrimalAttr(kSeqChunk)) {
        node.seq_chunk = GetValue<int64_t>(node.border->GetPrimalAttr(kSeqChunk));
      }
      node.border->AddAttr(kSeqChunk, MakeValue<int64_t>(node.seq_chunk));
      node.border->AddAttr(kChunk, MakeValue<int64_t>(node.chunk));
      node.border->AddAttr(kMicro, MakeValue<int64_t>(node.micro));
      node.border->AddAttr(kIsBp, MakeValue<bool>(is_bp));
    }
  };
  set_seq_chunk(fwd_begin_, false);
  set_seq_chunk(fwd_end_, false);
  set_seq_chunk(fwd_cell_, false);
  set_seq_chunk(bwd_begin_, true);
  set_seq_chunk(bwd_end_, true);
  set_seq_chunk(bwd_cell_, true);
}

void SeqpipeScheduler::SendRecvControl(const std::pair<BorderStruct, BorderStruct> &send,
                                       const std::pair<BorderStruct, BorderStruct> &recv) {
  auto prior = stage_ % INT64_TWO != 0 ? send : recv;
  auto post = stage_ % INT64_TWO != 0 ? recv : send;
  ControlOrder(prior.second, post.first, "send_recv");
}

void SeqpipeScheduler::SpecialControl(const std::pair<BorderStruct, BorderStruct> &origin_recv,
                                      const std::pair<BorderStruct, BorderStruct> &send,
                                      const std::pair<BorderStruct, BorderStruct> &recv,
                                      const std::pair<BorderStruct, BorderStruct> &prior_cell, size_t index) {
  if (IsPrimitiveCNode(origin_recv.second.border, prim::kPrimReceive) &&
      IsPrimitiveCNode(send.first.border, prim::kPrimSend)) {
    SchedulerNode origin_receive_node = {"receive", origin_recv.first.seq_chunk, origin_recv.first.micro,
                                         origin_recv.first.chunk,
                                         origin_recv.first.border->HasPrimalAttr(kPrimalAttrForwardUniqueId)};
    SchedulerNode send_node = {"send", send.second.seq_chunk, send.second.micro, send.second.chunk,
                               send.second.border->HasPrimalAttr(kPrimalAttrForwardUniqueId)};
    InsertSchedulerNode(origin_receive_node, send_node, index);
    ControlOrder(origin_recv.second, send.first, "calm_down");
  }
  if (IsPrimitiveCNode(origin_recv.second.border, prim::kPrimReceive) &&
      IsPrimitiveCNode(recv.first.border, prim::kPrimReceive)) {
    SchedulerNode origin_receive_node = {"receive", origin_recv.first.seq_chunk, origin_recv.first.micro,
                                         origin_recv.first.chunk,
                                         origin_recv.first.border->HasPrimalAttr(kPrimalAttrForwardUniqueId)};
    SchedulerNode receive_node = {"receive", recv.first.seq_chunk, recv.first.micro, recv.first.chunk,
                                  recv.first.border->HasPrimalAttr(kPrimalAttrForwardUniqueId)};
    InsertSchedulerNode(origin_receive_node, receive_node, index);
    ControlOrder(origin_recv.second, recv.first, "calm_down");
  }
  if (IsPrimitiveCNode(origin_recv.first.border, prim::kPrimReceive)) {
    ControlOrder(prior_cell.second, origin_recv.first, "calm_down");
  }
  if (IsPrimitiveCNode(recv.first.border, prim::kPrimReceive) &&
      IsPrimitiveCNode(send.second.border, prim::kPrimSend)) {
    SchedulerNode send_node = {"send", send.second.seq_chunk, send.second.micro, send.second.chunk,
                               send.second.border->HasPrimalAttr(kPrimalAttrForwardUniqueId)};
    SchedulerNode receive_node = {"receive", recv.first.seq_chunk, recv.first.micro, recv.first.chunk,
                                  recv.first.border->HasPrimalAttr(kPrimalAttrForwardUniqueId)};
    if (small_micro_handle_stage_ && index == warm_up_size_ - 1) {
      ControlOrder(send.second, recv.first, "small_micro");
      InsertSchedulerNode(send_node, receive_node, index);
      return;
    }
    auto prior = stage_ % INT64_TWO != 0 ? send_node : receive_node;
    auto post = stage_ % INT64_TWO != 0 ? receive_node : send_node;
    InsertSchedulerNode(prior, post, index);
    SendRecvControl(send, recv);
  }
}

void SeqpipeScheduler::ComputeBias() { bias_ = chunk_num_ == 1 ? 1 : 2; }

void SeqsmartvppScheduler::ComputeBias() { bias_ = 1; }

void SeqpipeScheduler::ExtractDataStruct() {
  auto max_seq = std::max_element(fwd_cell_.begin(), fwd_cell_.end(),
                                  [](const auto &a, const auto &b) { return a.seq_chunk < b.seq_chunk; });
  if (max_seq != fwd_cell_.end()) {
    seq_chunk_size_ = max_seq->seq_chunk + 1;
  }
  fp_block_size_ = LongToSize(seq_chunk_size_ * micro_size_ * chunk_num_);
  sorted_fwd_begin_ = BorderMap(fwd_begin_);
  sorted_fwd_end_ = BorderMap(fwd_end_);
  sorted_fwd_cell_ = BorderMap(fwd_cell_);
  sorted_bwd_begin_ = BorderMap(bwd_begin_);
  sorted_bwd_end_ = BorderMap(bwd_end_);
  sorted_bwd_cell_ = BorderMap(bwd_cell_);
  ComputeCycleList();
  ComputeBias();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto pp_1f1b_overlap = ms_context->get_param<std::string>(MS_CTX_PP_1F1B_OVERLAP);
  bp_fp_inline_ = bias_ > 1 && !pp_1f1b_overlap.empty();  // ToDo: Add parallel context config
  int64_t cycle_list0 = SizeToLong(cycle_list_[0]);
  warm_up_size_ = LongToSize((stage_num_ - 1 - stage_) * bias_ + (chunk_num_ - 1) * cycle_list0 * seq_chunk_size_ +
                             seq_chunk_size_ - 1);
  small_micro_handle_stage_ = (warm_up_size_ == fp_block_size_ - 1) && (stage_ != 0);
  if (warm_up_size_ > fp_block_size_ - 1) {
    warm_up_size_ = fp_block_size_ - 1;
    before_small_micro_handle_stage_ = true;
  }
  execute_order_ = ExecuteOrder(warm_up_size_);
  scheduler_node_order_.resize(execute_order_.size());
  fp_execute_order_ = FpBpExecuteOrder(false);
  bp_execute_order_ = FpBpExecuteOrder(true);
  calm_down_index_ = kSizeTwo * fp_block_size_ - warm_up_size_;
}

size_t SeqpipeScheduler::GetOrderIndex(size_t seq_chunk, size_t micro, size_t chunk, bool is_bp, std::string type) {
  if (type == "fp") {
    return size_t(
      std::find(fp_execute_order_.begin(), fp_execute_order_.end(), Triplet({seq_chunk, micro, chunk, is_bp})) -
      fp_execute_order_.begin());
  }
  if (type == "bp") {
    return size_t(
      std::find(bp_execute_order_.begin(), bp_execute_order_.end(), Triplet({seq_chunk, micro, chunk, is_bp})) -
      bp_execute_order_.begin());
  }
  return size_t(std::find(execute_order_.begin(), execute_order_.end(), Triplet({seq_chunk, micro, chunk, is_bp})) -
                execute_order_.begin());
}

BorderPair SeqpipeScheduler::ControlAdvancedRecv(size_t index, size_t recv_node_index) {
  auto chunk = execute_order_[index].chunk;
  auto micro = execute_order_[index].micro;
  auto seq_chunk = execute_order_[index].seq_chunk;
  auto act_cell =
    execute_order_[index].is_bp ? sorted_bwd_cell_[chunk][micro][seq_chunk] : sorted_fwd_cell_[chunk][micro][seq_chunk];
  chunk = execute_order_[recv_node_index].chunk;
  micro = execute_order_[recv_node_index].micro;
  seq_chunk = execute_order_[recv_node_index].seq_chunk;
  auto recv_node = execute_order_[recv_node_index].is_bp ? sorted_bwd_begin_[chunk][micro][seq_chunk]
                                                         : sorted_fwd_begin_[chunk][micro][seq_chunk];
  if (IsPrimitiveCNode(recv_node.second.border, prim::kPrimReceive) && index != recv_node_index) {
    ControlOrder(recv_node.second, act_cell.first, "pre_recv_call");
  }
  return recv_node;
}

int64_t SeqpipeScheduler::CurrentChunkSize(size_t recv_node_index) {
  int64_t current_chunk_size = 0;
  size_t accu_len = 0;
  for (size_t cyc_len : cycle_list_) {
    if (execute_order_[recv_node_index].micro < accu_len) {
      break;
    }
    current_chunk_size = SizeToLong(cyc_len);
    accu_len += cyc_len;
  }
  return current_chunk_size;
}

BorderPair SeqpipeScheduler::GetBorderNodeRecv(size_t index) {
  size_t recv_node_index = index;
  if (chunk_num_ > 1 && index >= LongToSize(warm_up_size_ + 1) && index + 1 < LongToSize(calm_down_index_)) {
    recv_node_index = index + 1;
    MS_LOG(INFO) << "recv_node_index in 1f1b:" << recv_node_index << ", index:" << index;
  }
  if (chunk_num_ > 1 && small_micro_handle_stage_ && index == LongToSize(warm_up_size_)) {
    recv_node_index = index + 1;
    MS_LOG(INFO) << "recv_node_index in 1f1b:" << recv_node_index << ", index:" << index;
  }
  int64_t current_chunk_size = CurrentChunkSize(recv_node_index);
  int64_t first_stage_pre_fetch_index =
    seq_chunk_size_ * current_chunk_size < stage_num_ ? seq_chunk_size_ * current_chunk_size : stage_num_;
  if (chunk_num_ > 1 && stage_ == 0 && recv_node_index >= LongToSize(first_stage_pre_fetch_index) &&
      !execute_order_[recv_node_index].is_bp) {
    // distance to pre_fetched in fp_execute_order_ is seq_chunk_size_ * current_chunk_size - stage_num
    auto fp_index = GetOrderIndex(execute_order_[recv_node_index].seq_chunk, execute_order_[recv_node_index].micro,
                                  execute_order_[recv_node_index].chunk, false, "fp");
    auto prefetched_fp_index =
      fp_index + LongToSize(seq_chunk_size_ * current_chunk_size - first_stage_pre_fetch_index);
    if (prefetched_fp_index < fp_execute_order_.size()) {
      recv_node_index =
        GetOrderIndex(fp_execute_order_[prefetched_fp_index].seq_chunk, fp_execute_order_[prefetched_fp_index].micro,
                      fp_execute_order_[prefetched_fp_index].chunk, false);
      MS_LOG(INFO) << "recv_node_index:" << recv_node_index << ", prefetched_fp_index:" << prefetched_fp_index;
    }
  }
  if (chunk_num_ > 1 && stage_ == stage_num_ - 1 && recv_node_index >= last_stage_pre_fetch_index_ &&
      execute_order_[recv_node_index].is_bp) {
    auto bp_index = GetOrderIndex(execute_order_[recv_node_index].seq_chunk, execute_order_[recv_node_index].micro,
                                  execute_order_[recv_node_index].chunk, true, "bp");
    auto prefetched_bp_index = bp_index - last_stage_pre_fetch_bp_index_ + last_stage_pre_fetched_bp_index_ -
                               LongToSize(seq_chunk_size_) * (cycle_list_[0] - LongToSize(current_chunk_size));
    if (prefetched_bp_index < bp_execute_order_.size()) {
      recv_node_index =
        GetOrderIndex(bp_execute_order_[prefetched_bp_index].seq_chunk, bp_execute_order_[prefetched_bp_index].micro,
                      bp_execute_order_[prefetched_bp_index].chunk, true);
      MS_LOG(INFO) << "recv_node_index:" << recv_node_index << ", prefetched_bp_index:" << prefetched_bp_index;
    }
  }

  return BorderRecv(index, recv_node_index);
}

BorderPair SeqvppScheduler::GetBorderNodeRecv(size_t index) {
  size_t recv_node_index = index;
  if (index >= LongToSize(warm_up_size_ + 1) && index + 1 < LongToSize(calm_down_index_)) {
    recv_node_index = index + 1;
    MS_LOG(INFO) << "recv_node_index in 1f1b:" << recv_node_index << ", index:" << index;
  }

  if (stage_ == 0 && index >= LongToSize(stage_num_) && !execute_order_[index - 1].is_bp) {
    size_t pre_fp_index = GetOrderIndex(execute_order_[index - 1].seq_chunk, execute_order_[index - 1].micro,
                                        execute_order_[index - 1].chunk, false, "fp");
    size_t last_stage_send_node_index = pre_fp_index - LongToSize(stage_num_ - 1);
    if (pre_fp_index + 1 >= LongToSize(stage_num_) && !fp_execute_order_[last_stage_send_node_index].is_bp &&
        fp_execute_order_[last_stage_send_node_index].chunk + 1 < LongToSize(chunk_num_)) {
      recv_node_index = GetOrderIndex(fp_execute_order_[last_stage_send_node_index].seq_chunk,
                                      fp_execute_order_[last_stage_send_node_index].micro,
                                      fp_execute_order_[last_stage_send_node_index].chunk + 1, false);
      MS_LOG(INFO) << "recv_node_index:" << recv_node_index
                   << ", last_stage_send_node_index:" << last_stage_send_node_index << ", index is:" << index;
    }
  }

  if (stage_ == stage_num_ - 1 && execute_order_[index - 1].is_bp) {
    size_t pre_bp_index = GetOrderIndex(execute_order_[index - 1].seq_chunk, execute_order_[index - 1].micro,
                                        execute_order_[index - 1].chunk, true, "bp");
    size_t first_stage_send_node_index = pre_bp_index - LongToSize(stage_num_ - 1);
    if (pre_bp_index + 1 >= LongToSize(stage_num_) && bp_execute_order_[first_stage_send_node_index].is_bp &&
        bp_execute_order_[first_stage_send_node_index].chunk > 0) {
      recv_node_index = GetOrderIndex(bp_execute_order_[first_stage_send_node_index].seq_chunk,
                                      bp_execute_order_[first_stage_send_node_index].micro,
                                      bp_execute_order_[first_stage_send_node_index].chunk - 1, true);
      MS_LOG(INFO) << "recv_node_index:" << recv_node_index
                   << ", first_stage_send_node_index:" << first_stage_send_node_index << ", index is:" << index;
    }
  }

  return BorderRecv(index, recv_node_index);
}

void SeqsmartvppScheduler::ControlSpecialPreRecv(size_t index) {
  size_t last_stage_warm_up_size = warm_up_size_ + 1 - LongToSize(stage_num_);
  size_t border_fp_index = GetOrderIndex(execute_order_[index + 1].seq_chunk, execute_order_[index + 1].micro,
                                         execute_order_[index + 1].chunk, false, "fp");
  size_t border_last_stage_send_node_index = border_fp_index - LongToSize(stage_num_ / INT64_TWO);
  auto prior_call = sorted_fwd_cell_[execute_order_[index - 1].chunk][execute_order_[index - 1].micro]
                                    [execute_order_[index - 1].seq_chunk];
  auto next_recv =
    sorted_bwd_begin_[execute_order_[index].chunk][execute_order_[index].micro][execute_order_[index].seq_chunk];
  for (size_t sp_idx = last_stage_warm_up_size; sp_idx < border_last_stage_send_node_index; ++sp_idx) {
    if (fp_execute_order_[sp_idx].chunk + 1 < LongToSize(chunk_num_)) {
      size_t sp_recv_node_index = GetOrderIndex(fp_execute_order_[sp_idx].seq_chunk, fp_execute_order_[sp_idx].micro,
                                                fp_execute_order_[sp_idx].chunk + 1, false);
      MS_LOG(INFO) << "sp_recv_node_index:" << sp_recv_node_index << ", sp_idx:" << sp_idx << ", index is:" << index;
      advanced_recv_indexs_.push_back(sp_recv_node_index);

      // prior call to recv && recv to next recv
      auto sp_pre_recv =
        sorted_fwd_begin_[execute_order_[sp_recv_node_index].chunk][execute_order_[sp_recv_node_index].micro]
                         [execute_order_[sp_recv_node_index].seq_chunk];
      if (IsPrimitiveCNode(sp_pre_recv.first.border, prim::kPrimReceive)) {
        ControlOrder(prior_call.second, sp_pre_recv.first, "sp_call_pre_recv");
        ControlOrder(sp_pre_recv.second, next_recv.first, "sp_pre_recv_recv");
      }
      if (!special_recv_indexs_.empty()) {
        auto pre_sp_pre_recv = sorted_fwd_cell_[execute_order_[special_recv_indexs_.back()].chunk]
                                               [execute_order_[special_recv_indexs_.back()].micro]
                                               [execute_order_[special_recv_indexs_.back()].seq_chunk];
        ControlOrder(pre_sp_pre_recv.second, sp_pre_recv.first, "sp_pre_recv_recv");
      }
      special_recv_indexs_.push_back(sp_recv_node_index);
    }
  }
}

BorderPair SeqsmartvppScheduler::GetBorderNodeRecv(size_t index) {
  size_t recv_node_index = index;
  if (stage_ == 0 && index == warm_up_size_ + 1) {
    ControlSpecialPreRecv(index);
  }
  size_t warm_up_size_stage0 = LongToSize(stage_ * bias_ + warm_up_size_);
  size_t calm_down_index_stage0 = kSizeTwo * fp_block_size_ - warm_up_size_stage0;
  bool is_in_full_1f1b_range = index >= warm_up_size_stage0 + 1 && index + 1 < calm_down_index_stage0;
  if (stage_ == 0 && index >= LongToSize(stage_num_) && !execute_order_[index].is_bp) {
    size_t fp_index = GetOrderIndex(execute_order_[index].seq_chunk, execute_order_[index].micro,
                                    execute_order_[index].chunk, false, "fp");
    size_t last_stage_send_node_index =
      is_in_full_1f1b_range ? fp_index - LongToSize(stage_num_ / INT64_TWO) : fp_index - LongToSize(stage_num_);
    MS_LOG(INFO) << "last_stage_send_node_index:" << last_stage_send_node_index << ", index:" << index;
    if (fp_index >= LongToSize(stage_num_) &&
        fp_execute_order_[last_stage_send_node_index].chunk + 1 < LongToSize(chunk_num_)) {
      recv_node_index = GetOrderIndex(fp_execute_order_[last_stage_send_node_index].seq_chunk,
                                      fp_execute_order_[last_stage_send_node_index].micro,
                                      fp_execute_order_[last_stage_send_node_index].chunk + 1, false);
      MS_LOG(INFO) << "recv_node_index:" << recv_node_index
                   << ", last_stage_send_node_index:" << last_stage_send_node_index << ", index is:" << index;
    }
  }

  if (stage_ == stage_num_ - 1 && execute_order_[index].is_bp) {
    size_t bp_index = GetOrderIndex(execute_order_[index].seq_chunk, execute_order_[index].micro,
                                    execute_order_[index].chunk, true, "bp");

    size_t first_stage_send_node_index =
      is_in_full_1f1b_range ? bp_index - LongToSize(stage_num_ / INT64_TWO) : bp_index - LongToSize(stage_num_);
    MS_LOG(INFO) << "first_stage_send_node_index:" << first_stage_send_node_index << ", index:" << index;
    // Assure the first_stage_send_node_index is correct.
    if ((is_in_full_1f1b_range || index + 1 >= calm_down_index_stage0 + LongToSize(stage_num_)) &&
        bp_index >= LongToSize(stage_num_ / INT64_TWO) && bp_execute_order_[first_stage_send_node_index].chunk > 0) {
      recv_node_index = GetOrderIndex(bp_execute_order_[first_stage_send_node_index].seq_chunk,
                                      bp_execute_order_[first_stage_send_node_index].micro,
                                      bp_execute_order_[first_stage_send_node_index].chunk - 1, true);
      MS_LOG(INFO) << "recv_node_index:" << recv_node_index
                   << ", first_stage_send_node_index:" << first_stage_send_node_index << ", index is:" << index;
    }
  }

  return BorderRecv(index, recv_node_index);
}

BorderPair SeqpipeScheduler::BorderRecv(size_t index, size_t recv_node_index) {
  if (recv_node_index >= execute_order_.size() || std::find(advanced_recv_indexs_.begin(), advanced_recv_indexs_.end(),
                                                            recv_node_index) != advanced_recv_indexs_.end()) {
    return BorderPair({Border({nullptr, 0, 0, 0}), Border({nullptr, 0, 0, 0})});
  }
  if (recv_node_index != index) {
    advanced_recv_indexs_.push_back(recv_node_index);
  }
  return ControlAdvancedRecv(index, recv_node_index);
}

BorderPair SeqpipeScheduler::GetBorderNode(const std::string &border_type, size_t index) {
  auto chunk = execute_order_[index].chunk;
  auto micro = execute_order_[index].micro;
  auto seq_chunk = execute_order_[index].seq_chunk;
  if (border_type == kSend) {
    auto send_node =
      execute_order_[index].is_bp ? sorted_bwd_end_[chunk][micro][seq_chunk] : sorted_fwd_end_[chunk][micro][seq_chunk];
    return send_node;
  }
  if (border_type == kReceive) {
    if (recv_nodes_map_.count(index) > 0) {
      return recv_nodes_map_[index];
    }
    auto recv = GetBorderNodeRecv(index);
    recv_nodes_map_[index] = recv;
    return recv;
  }
  auto cell_node =
    execute_order_[index].is_bp ? sorted_bwd_cell_[chunk][micro][seq_chunk] : sorted_fwd_cell_[chunk][micro][seq_chunk];
  return cell_node;
}

void SeqpipeScheduler::ComputePrefetchInfo() {
  size_t warm_up_size_stage0 = LongToSize(stage_ * bias_ + warm_up_size_);
  size_t bp_delta = 1;
  if (stage_ == stage_num_ - 1 && fp_block_size_ < LongToSize(stage_num_) + warm_up_size_) {
    size_t fp_1f1b_size = fp_block_size_ - warm_up_size_;
    warm_up_size_stage0 -= (LongToSize(stage_num_) - fp_1f1b_size);
    bp_delta = 0;
  }
  // stage0 first bp send position is warm_up_size_stage0 + 1.
  // Thus, the last stage doing bp receive position should match the stage0 send position.
  if (chunk_num_ > 1 && stage_ == stage_num_ - 1) {
    last_stage_pre_fetched_bp_index_ =
      GetOrderIndex(LongToSize(seq_chunk_size_ - 1), 0, LongToSize(chunk_num_ - INT64_TWO), true, "bp");
    auto pre_recv = execute_order_[warm_up_size_stage0 + 1];
    last_stage_pre_fetch_bp_index_ =
      GetOrderIndex(pre_recv.seq_chunk, pre_recv.micro, pre_recv.chunk, true, "bp") + bp_delta;
    auto pre_recv_bp = bp_execute_order_[last_stage_pre_fetch_bp_index_];
    last_stage_pre_fetch_index_ = GetOrderIndex(pre_recv_bp.seq_chunk, pre_recv_bp.micro, pre_recv_bp.chunk, true);
  }
  MS_LOG(INFO) << "pre_fetch_idx:" << last_stage_pre_fetch_index_
               << ", pre_fetch_bp_idx:" << last_stage_pre_fetch_bp_index_ << ", pre_fetched_bp_idx"
               << last_stage_pre_fetched_bp_index_;
}

void SeqpipeScheduler::InsertSchedulerNode(SchedulerNode prior_node, SchedulerNode next_node, size_t index) {
  if (scheduler_node_order_[index].count(prior_node) == 0) {
    scheduler_node_order_[index][prior_node] = 0;
  }
  if (scheduler_node_order_[index].count(next_node) == 0) {
    scheduler_node_order_[index][next_node] = scheduler_node_order_[index][prior_node] + 1;
  } else if (scheduler_node_order_[index][next_node] <= scheduler_node_order_[index][prior_node]) {
    auto delta = scheduler_node_order_[index][prior_node] + 1 - scheduler_node_order_[index][next_node];
    for (auto &iter : scheduler_node_order_[index]) {
      if (iter.second > scheduler_node_order_[index][next_node]) {
        iter.second += delta;
      }
    }
    scheduler_node_order_[index][next_node] = scheduler_node_order_[index][prior_node] + 1;
  }
}

void SeqpipeScheduler::ControlSendRecvOrder(const BorderPair &send, const BorderPair &post_recv, size_t index) {
  bool special_control = chunk_num_ > 1 && ((index + 2 < calm_down_index_ && index == warm_up_size_) ||
                                            (small_micro_handle_stage_ && index == warm_up_size_ - 1));
  if (special_control) {
    auto idx_node = execute_order_[index + 1];
    auto origin_recv = execute_order_[index + 1].is_bp
                         ? sorted_bwd_begin_[idx_node.chunk][idx_node.micro][idx_node.seq_chunk]
                         : sorted_fwd_begin_[idx_node.chunk][idx_node.micro][idx_node.seq_chunk];
    SpecialControl(origin_recv, send, post_recv, GetBorderNode(kCell, index), index);
  }
  if (!special_control && IsPrimitiveCNode(post_recv.first.border, prim::kPrimReceive) &&
      IsPrimitiveCNode(send.second.border, prim::kPrimSend)) {
    SchedulerNode send_node = {"send", send.second.seq_chunk, send.second.micro, send.second.chunk,
                               send.second.border->HasPrimalAttr(kPrimalAttrForwardUniqueId)};
    SchedulerNode receive_node = {"receive", post_recv.first.seq_chunk, post_recv.first.micro, post_recv.first.chunk,
                                  post_recv.first.border->HasPrimalAttr(kPrimalAttrForwardUniqueId)};
    if (warm_up_size_ == fp_block_size_ - 1 && index == warm_up_size_ && before_small_micro_handle_stage_) {
      ControlOrder(send.second, post_recv.first, "small_micro");
      InsertSchedulerNode(send_node, receive_node, index);
    } else {
      SendRecvControl(send, post_recv);
      auto prior = stage_ % INT64_TWO != 0 ? send_node : receive_node;
      auto post = stage_ % INT64_TWO != 0 ? receive_node : send_node;
      InsertSchedulerNode(prior, post, index);
    }
  }
}

void SeqsmartvppScheduler::ControlSendRecvOrder(const BorderPair &send, const BorderPair &post_recv, size_t index) {
  if (IsPrimitiveCNode(post_recv.first.border, prim::kPrimReceive) &&
      IsPrimitiveCNode(send.second.border, prim::kPrimSend)) {
    SchedulerNode send_node = {"send", send.second.seq_chunk, send.second.micro, send.second.chunk,
                               send.second.border->HasPrimalAttr(kPrimalAttrForwardUniqueId)};
    SchedulerNode receive_node = {"receive", post_recv.first.seq_chunk, post_recv.first.micro, post_recv.first.chunk,
                                  post_recv.first.border->HasPrimalAttr(kPrimalAttrForwardUniqueId)};
    SendRecvControl(send, post_recv);
    auto prior = stage_ % INT64_TWO != 0 ? send_node : receive_node;
    auto post = stage_ % INT64_TWO != 0 ? receive_node : send_node;
    InsertSchedulerNode(prior, post, index);
  }
}

void SeqpipeScheduler::Reorder1f1bOverlap() {
  for (size_t index = 0; index < execute_order_.size() - 1; ++index) {
    auto prior_cell = GetBorderNode(kCell, index);
    auto post_recv = GetBorderNode(kReceive, index + 1);
    auto next_cell = GetBorderNode(kCell, index + 1);
    auto send = GetBorderNode(kSend, index);
    bool is_1b1f = execute_order_[index].is_bp && !execute_order_[index + 1].is_bp;
    if (bp_fp_inline_ && is_1b1f) {
      if (IsPrimitiveCNode(post_recv.first.border, prim::kPrimReceive)) {
        auto cell_inputs = prior_cell.second.border->inputs();
        for (const auto &cell_input : cell_inputs) {
          if (IsPrimitiveCNode(cell_input, prim::kPrimDepend)) {
            ControlOrder({cell_input->cast<CNodePtr>(), 0, 0}, post_recv.first, "input_recv_1f1b");
          }
        }
      }
      if (IsPrimitiveCNode(send.second.border, prim::kPrimSend)) {
        auto next_users = GetOutputNodesWithFilter(next_cell.first.border, [&](const AnfNodePtr &anode) {
          return IsPrimitiveCNode(anode, prim::kPrimTupleGetItem);
        });
        for (const auto &next_user : next_users) {
          if (IsPrimitiveCNode(next_user.first, prim::kPrimDepend)) {
            ControlOrder(send.second, {next_user.first->cast<CNodePtr>(), 0, 0}, "send_out_1f1b");
          }
        }
      }
      if (index > 0) {
        auto prior_prior_cell = GetBorderNode(kCell, index - 1);
        ControlOrder(prior_prior_cell.second, next_cell.first, "call_call_1f1b");
      }
      if (index < execute_order_.size() - 3) {
        auto next_next_cell = GetBorderNode(kCell, index + 2);
        ControlOrder(prior_cell.second, next_next_cell.first, "call_call_1f1b");
      }
    }
  }
}

void SeqpipeScheduler::Add1f1bAttr(const BorderPair &recv, const std::string &tag, size_t index_1f1b) {
  if (IsPrimitiveCNode(recv.second.border)) {
    recv.second.border->AddAttr(tag, MakeValue<size_t>(index_1f1b));
  }
  if (IsPrimitiveCNode(recv.first.border)) {
    recv.first.border->AddAttr(tag, MakeValue<size_t>(index_1f1b));
  }
}

void SeqpipeScheduler::Reorder() {
  ExtractDataStruct();
  ControlCleanAssigns();
  ComputePrefetchInfo();
  size_t index_1f1b = 0;
  for (size_t index = 0; index < execute_order_.size() - 1; ++index) {
    auto prior_cell = GetBorderNode(kCell, index);
    auto post_recv = GetBorderNode(kReceive, index + 1);
    auto next_cell = GetBorderNode(kCell, index + 1);
    bool is_1b1f = execute_order_[index].is_bp && !execute_order_[index + 1].is_bp;
    if (!(bp_fp_inline_ && is_1b1f)) {
      ControlOrder(prior_cell.second, next_cell.first, "call_call");
    } else {
      prior_cell.second.border->AddAttr(kCNodeAttr1f1bIndexBp, MakeValue<size_t>(index_1f1b));
      next_cell.first.border->AddAttr(kCNodeAttr1f1bIndexFp, MakeValue<size_t>(index_1f1b));
      auto prior_recv = GetBorderNode(kReceive, index);
      Add1f1bAttr(prior_recv, kCNodeAttr1f1bIndexRecv, index_1f1b);
      Add1f1bAttr(post_recv, kCNodeAttr1f1bIndexInterRecv, index_1f1b);
      ControlOrder(prior_cell.second, next_cell.first, kPrimalAttr1b1fCallCall);
      ++index_1f1b;
    }
    if (IsPrimitiveCNode(post_recv.first.border, prim::kPrimReceive)) {
      SchedulerNode receive_node = {"receive", post_recv.first.seq_chunk, post_recv.first.micro, post_recv.first.chunk,
                                    post_recv.first.border->HasPrimalAttr(kPrimalAttrForwardUniqueId)};
      if (scheduler_node_order_[index].count(receive_node) == 0) {
        scheduler_node_order_[index][receive_node] = 0;
      }
      if (!(bp_fp_inline_ && is_1b1f)) {
        ControlOrder(prior_cell.second, post_recv.first, "call_recv");
      }
    }
    // send to recv or recv to send
    auto send = GetBorderNode(kSend, index);
    ControlSendRecvOrder(send, post_recv, index);
    if (IsPrimitiveCNode(send.second.border, prim::kPrimSend)) {
      SchedulerNode send_node = {"send", send.second.seq_chunk, send.second.micro, send.second.chunk,
                                 send.second.border->HasPrimalAttr(kPrimalAttrForwardUniqueId)};
      if (scheduler_node_order_[index].count(send_node) == 0) {
        scheduler_node_order_[index][send_node] = 0;
      }
      if (!(bp_fp_inline_ && is_1b1f)) {
        ControlOrder(send.second, next_cell.first, "send_call");
      }
    }
    if (IsPrimitiveCNode(send.first.border, prim::kPrimSend) && !(bp_fp_inline_ && is_1b1f)) {
      ControlOrder(prior_cell.second, send.first, "call_send");
    }
  }
  auto last_cell = GetBorderNode(kCell, execute_order_.size() - 1);
  auto last_send = GetBorderNode(kSend, execute_order_.size() - 1);
  if (IsPrimitiveCNode(last_send.first.border, prim::kPrimSend)) {
    ControlOrder(last_cell.second, last_send.first, "call_send");
  }

  Reorder1f1bOverlap();
  SchedulerOrder();
  OptimizerShardCommReorder();
  ReorderShardedParam();
}

AbstractBasePtr SeqpipeScheduler::GenerateTupleAbstract(const std::vector<AnfNodePtr> &nodes) {
  AbstractBasePtr abs;
  if (nodes.size() == kSizeTwo) {
    auto cnode = nodes.back()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    abs = cnode->abstract();
  } else {
    AbstractBasePtrList abstract_list;
    abstract_list.resize(nodes.size() - 1);
    (void)std::transform(nodes.begin() + 1, nodes.end(), abstract_list.begin(), [](const AnfNodePtr &node) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      return cnode->abstract();
    });
    abs = std::make_shared<abstract::AbstractTuple>(abstract_list);
  }
  return abs;
}

void SeqpipeScheduler::SetCleanAssignsMicro() {
  auto add_micro = [&](std::vector<Border> &border_vector) {
    std::unordered_map<int64_t, int64_t> micro_map;
    for (Border &border : border_vector) {
      if (micro_map.count(border.chunk) == 0) {
        micro_map[border.chunk] = 0;
      } else {
        micro_map[border.chunk] += 1;
      }
      border.micro = micro_map[border.chunk];
    }
    if (micro_map.size() != LongToSize(chunk_num_)) {
      MS_LOG(ERROR) << "The chunk num of cleaning assign of kv_cache:" << micro_map.size()
                    << "is not equal to the real chunk num:" << chunk_num_;
      return false;
    }
    if (std::find_if(micro_map.begin(), micro_map.end(),
                     [&](const auto &iter) { return iter.second != micro_size_ - 1; }) != micro_map.end()) {
      MS_LOG(ERROR) << "The micro size of cleaning assign of kv_cache is not equal to the micro size:" << micro_size_;
      return false;
    }
    return true;
  };
  for (auto &param_borders : clean_mask_cache_assigns_) {
    if (!add_micro(param_borders.second)) {
      MS_LOG(EXCEPTION) << "Parameter:" << param_borders.first << " has wrong chunk num or micro size.";
    }
  }
  for (auto &param_borders : clean_seq_chunk_assigns_) {
    if (!add_micro(param_borders.second)) {
      MS_LOG(EXCEPTION) << "Parameter:" << param_borders.first << " has wrong chunk num or micro size.";
    }
  }
}

void SeqpipeScheduler::GetCleanAssigns() {
  auto all_nodes = TopoSort(root_->get_return(), SuccDeeperSimple);
  for (auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimAssign)) {
      continue;
    }
    auto assign_cnode = node->cast<CNodePtr>();
    auto assign_graph = assign_cnode->func_graph();
    if (assign_graph->has_flag((FUNC_GRAPH_FLAG_CELL_REUSE))) {
      continue;
    }
    auto assign_input = GetInputNodeWithFilter(assign_cnode->input(kIndex1), [&](const CNodePtr &cnode) {
      bool filter = IsPrimitiveCNode(cnode, prim::kPrimLoad) || IsPrimitiveCNode(cnode, prim::kPrimDepend);
      return std::make_pair(filter, 1);
    });
    if (!assign_input->isa<Parameter>()) {
      continue;
    }
    auto assign_param = assign_input->cast<ParameterPtr>();
    auto param_name = assign_param->name();
    if (param_name.find(kMaskCache) != std::string::npos) {
      clean_mask_cache_assigns_[param_name].push_back({assign_cnode, 0, 0});
      for (int64_t i = 1; i < chunk_num_; ++i) {
        auto new_assign_cnode = assign_graph->NewCNode(assign_cnode->inputs());
        new_assign_cnode->set_abstract(assign_cnode->abstract()->Clone());
        clean_mask_cache_assigns_[param_name].push_back({new_assign_cnode, i, 0});
      }
    }
    if (param_name.find(kSeqChunk) != std::string::npos) {
      clean_seq_chunk_assigns_[param_name].push_back({assign_cnode, 0, 0});
      for (int64_t i = 1; i < chunk_num_; ++i) {
        auto new_assign_cnode = assign_graph->NewCNode(assign_cnode->inputs());
        new_assign_cnode->set_abstract(assign_cnode->abstract()->Clone());
        clean_seq_chunk_assigns_[param_name].push_back({new_assign_cnode, i, 0});
      }
    }
  }
  SetCleanAssignsMicro();
}

void SeqpipeScheduler::ControlCleanAssigns() {
  std::unordered_map<int64_t, std::unordered_map<std::int64_t, std::vector<CNodePtr>>> chunk_assigns;
  std::unordered_map<int64_t, std::unordered_map<std::int64_t, std::vector<CNodePtr>>> chunk_calls;
  for (const auto &mask_cache : clean_mask_cache_assigns_) {
    for (const auto &cache_border : mask_cache.second) {
      chunk_assigns[cache_border.chunk][cache_border.micro].push_back(cache_border.border);
    }
  }
  for (const auto &seq_chunk : clean_seq_chunk_assigns_) {
    for (const auto &cache_border : seq_chunk.second) {
      chunk_assigns[cache_border.chunk][cache_border.micro].push_back(cache_border.border);
    }
  }
  for (const auto &call_cell : fwd_cell_) {
    chunk_calls[call_cell.chunk][call_cell.micro].push_back(call_cell.border);
  }

  for (size_t index = 0; index < fp_execute_order_.size(); ++index) {
    auto chunk = fp_execute_order_[index].chunk;
    auto micro = fp_execute_order_[index].micro;
    auto seq_chunk = fp_execute_order_[index].seq_chunk;
    // Clean before each chunk & micro.
    if (seq_chunk != 0) {
      continue;
    }
    // all_assign0->make_tuple->fwd_cell0s
    std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple->Clone())};
    auto assign_v = chunk_assigns[chunk][micro];
    if (assign_v.empty()) {
      continue;
    }
    std::vector<AbstractBasePtr> maketuple_abs_inputs;
    std::copy(assign_v.begin(), assign_v.end(), std::back_inserter(make_tuple_inputs));
    (void)std::transform(assign_v.begin(), assign_v.end(), std::back_inserter(maketuple_abs_inputs),
                         [](AnfNodePtr anf_node) { return anf_node->abstract()->Clone(); });
    auto func_graph = assign_v.front()->func_graph();
    auto make_tuple_cnode = func_graph->NewCNode(make_tuple_inputs);
    make_tuple_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(maketuple_abs_inputs));
    std::vector<AnfNodePtr> call_make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple->Clone())};
    std::vector<AbstractBasePtr> call_make_tuple_abs_inputs;
    for (const auto &call_node : chunk_calls[chunk][micro]) {
      call_make_tuple_inputs.push_back(call_node);
      call_make_tuple_abs_inputs.push_back(call_node->abstract()->Clone());
      ControlOrder({make_tuple_cnode, SizeToLong(chunk), SizeToLong(micro)},
                   {call_node, SizeToLong(chunk), SizeToLong(micro)}, "clean_seqpipe_control");
    }
    // find next micro & chunk
    auto next_index = index + LongToSize(seq_chunk_size_);
    if (next_index >= fp_execute_order_.size()) {
      continue;
    }
    auto n_chunk = fp_execute_order_[next_index].chunk;
    auto n_micro = fp_execute_order_[next_index].micro;
    auto call_make_tuple_cnode = func_graph->NewCNode(call_make_tuple_inputs);
    call_make_tuple_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(call_make_tuple_abs_inputs));
    // fwd_cell0s->make_tuple->all_assign1s
    for (const auto &assign_node : chunk_assigns[n_chunk][n_micro]) {
      ControlOrder({call_make_tuple_cnode, SizeToLong(chunk), SizeToLong(micro)},
                   {assign_node, SizeToLong(n_chunk), SizeToLong(n_micro)}, "clean_seqpipe_control");
    }
  }
}

void SeqpipeScheduler::SchedulerOrder() {
  if (stage_ != 0) {
    MS_LOG(INFO) << "(node, micro, chunk, seq_chunk, is_bp): (receive, 0, 0, 0, 0)";
  }
  for (size_t index = 0; index < execute_order_.size() - 1; ++index) {
    MS_LOG(INFO) << "(node, micro, chunk, seq_chunk, is_bp): (fp_bp, " << execute_order_[index].micro << ", "
                 << execute_order_[index].chunk << ", " << execute_order_[index].seq_chunk << ", "
                 << execute_order_[index].is_bp << ")";
    auto scheduler_nodes = scheduler_node_order_[index];
    std::vector<std::pair<SchedulerNode, size_t>> send_recv_nodes(scheduler_nodes.begin(), scheduler_nodes.end());
    std::sort(send_recv_nodes.begin(), send_recv_nodes.end(),
              [](const auto &a, const auto &b) { return a.second < b.second; });
    for (const auto &s_node : send_recv_nodes) {
      MS_LOG(INFO) << "(node, micro, chunk, seq_chunk, is_bp): (" << s_node.first.type << ", " << s_node.first.micro
                   << ", " << s_node.first.chunk << ", " << s_node.first.seq_chunk << ", " << s_node.first.is_bp << ")";
    }
  }
  MS_LOG(INFO) << "(node, micro, chunk, seq_chunk, is_bp): (fp_bp, " << micro_size_ - 1 << ", 0, 0, 1)";
  if (stage_ != 0) {
    MS_LOG(INFO) << "(node, micro, chunk, seq_chunk, is_bp): (send, " << micro_size_ - 1 << ", 0, 0, 1)";
  }
}

void SeqpipeScheduler::ReorderShardedParam() {
  if (!fwd_params_.empty()) {
    std::sort(fwd_params_.begin(), fwd_params_.end(), SortFuncInsideMicro);
    auto fwd_begin_info = execute_order_.front();
    auto prior = fwd_params_.back();
    auto last = sorted_fwd_begin_[fwd_begin_info.chunk][fwd_begin_info.micro][fwd_begin_info.seq_chunk];
    ControlOrder(prior, last.first);
  }
  if (!bwd_params_.empty()) {
    std::sort(bwd_params_.begin(), bwd_params_.end(), SortFuncInsideMicro);
    auto bwd_end_info = execute_order_.back();
    auto prior2 = sorted_bwd_end_[bwd_end_info.chunk][bwd_end_info.micro][bwd_end_info.seq_chunk];
    auto last2 = bwd_params_.front();
    ControlOrder(prior2.second, last2);
  }
}

void SeqpipeScheduler::OptimizerShardCommReorder() {
  auto enable_opt_shard = ParallelContext::GetInstance()->enable_parallel_optimizer();
  if (!enable_opt_shard) {
    return;
  }
  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  CompactSet<AnfNodePtr> tuple_set;
  for (int64_t chunk = 1; chunk < chunk_num_; ++chunk) {
    auto chunk_cells = sorted_fwd_cell_[chunk];
    for (const auto &row : chunk_cells) {
      for (const auto &border : row) {
        auto cnode = border.first.border;
        for (const auto &input : cnode->inputs()) {
          if (!IsPrimitiveCNode(input, prim::kPrimAllGather)) {
            continue;
          }
          tuple_set.insert(input);
        }
      }
    }
  }
  std::copy(tuple_set.begin(), tuple_set.end(), std::back_inserter(make_tuple_inputs));
  if (make_tuple_inputs.size() > 1) {
    auto make_tuple = root_->NewCNode(make_tuple_inputs);
    auto abs = GenerateTupleAbstract(make_tuple_inputs);
    make_tuple->set_abstract(abs);
    auto begin_node = sorted_fwd_begin_[0][0][0].first.border;
    if (begin_node->inputs().size() < kSizeTwo) {
      return;
    }
    std::vector<AnfNodePtr> depend_inputs = {NewValueNode(prim::kPrimDepend), begin_node->input(1), make_tuple};
    auto depend = root_->NewCNode(depend_inputs);
    depend->set_abstract(begin_node->input(1)->abstract());
    (void)manager_->SetEdge(begin_node, 1, depend);
  }
}
SchedulerRegisterAction PipelineSchedulerSeqpipe(parallel::kPipelineSeqpipe, [](const FuncGraphManagerPtr &manager,
                                                                                const FuncGraphPtr &root, int64_t stage,
                                                                                int64_t stage_num) {
  return std::make_shared<parallel::SeqpipeScheduler>(manager, root, stage, stage_num);
});
SchedulerRegisterAction PipelineSchedulerSeqvpp(parallel::kPipelineSeqvpp, [](const FuncGraphManagerPtr &manager,
                                                                              const FuncGraphPtr &root, int64_t stage,
                                                                              int64_t stage_num) {
  return std::make_shared<parallel::SeqvppScheduler>(manager, root, stage, stage_num);
});
SchedulerRegisterAction PipelineSchedulerSeqsmartvpp(parallel::kPipelineSeqsmartvpp,
                                                     [](const FuncGraphManagerPtr &manager, const FuncGraphPtr &root,
                                                        int64_t stage, int64_t stage_num) {
                                                       return std::make_shared<parallel::SeqsmartvppScheduler>(
                                                         manager, root, stage, stage_num);
                                                     });
}  // namespace parallel
}  // namespace mindspore
