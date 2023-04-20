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
#include "tools/graph_kernel/converter/conv_tuning_expander.h"
#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "backend/common/graph_kernel/core/graph_kernel_callback.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "tools/graph_kernel/converter/akg/akg_kernel_builder.h"
#include "utils/anf_utils.h"
#include "utils/file_utils.h"
#include "utils/hash_set.h"
#include "utils/ms_context.h"

namespace mindspore::graphkernel {
bool IsSameNumberList(const std::vector<int64_t> &vec, int64_t n) {
  return std::all_of(vec.begin(), vec.end(), [n](int64_t i) { return i == n; });
}

bool InvalidConvAttr(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &stride,
                     const std::vector<int64_t> &dilation) {
  constexpr int64_t one_kernel_size = 1;
  constexpr int64_t two_kernel_size = 2;
  constexpr int64_t winograd_kernel_size = 3;
  if ((IsSameNumberList(kernel_size, one_kernel_size) || IsSameNumberList(kernel_size, two_kernel_size) ||
       IsSameNumberList(kernel_size, winograd_kernel_size)) &&
      IsSameNumberList(stride, 1LL) && IsSameNumberList(dilation, 1LL)) {
    return true;
  }
  return false;
}

bool IsInvalidConv(const AnfNodePtr &node) {
  auto cb = Callback::Instance();
  auto input_shape = cb->GetInputShape(node, 0);
  if (input_shape.size() == 0) {
    return true;
  }
  auto prim = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(prim);
  const auto kernel_size = GetValue<std::vector<int64_t>>(prim->GetAttr("kernel_size"));
  const auto stride = GetValue<std::vector<int64_t>>(prim->GetAttr("stride"));
  const auto dilation = GetValue<std::vector<int64_t>>(prim->GetAttr("dilation"));
  if (InvalidConvAttr(kernel_size, stride, dilation)) {
    return true;
  }
  return false;
}

bool IsBlackListOp(const AnfNodePtr &node) {
  std::vector<PrimitivePtr> black_list = {prim::kPrimMatMulFusion};
  for (auto &prim : black_list) {
    if (IsPrimitiveCNode(node, prim)) {
      return true;
    }
    if (IsPrimitiveCNode(node, prim::kPrimConv2DFusion) && IsInvalidConv(node)) {
      return true;
    }
  }
  return false;
}

std::vector<int64_t> GenConvShape(const AnfNodePtr &conv_node) {
  auto cb = Callback::Instance();
  auto input_shape = cb->GetInputShape(conv_node, 0);
  auto prim = GetCNodePrimitive(conv_node);
  auto pads = GetValue<std::vector<int64_t>>(prim->GetAttr("pad_list"));
  constexpr size_t n_pos = 0;
  constexpr size_t h_pos = 1;
  constexpr size_t w_pos = 2;
  constexpr size_t c_pos = 3;
  input_shape[h_pos] += pads[n_pos] + pads[h_pos];
  input_shape[w_pos] += pads[w_pos] + pads[c_pos];
  return input_shape;
}

nlohmann::json GenTuneInfo(const AnfNodePtr &conv_node, const std::map<AnfNodePtr, bool> &former_conv_nodes,
                           const AnfNodePtrList &conv_list) {
  nlohmann::json node_info;
  auto prim = GetCNodePrimitive(conv_node);
  node_info["op_id"] = std::find(conv_list.begin(), conv_list.end(), conv_node) - conv_list.begin();
  node_info["op_type"] = "Conv2D";
  node_info["impl"] = "direct";
  node_info["origin_input_shape"] = GenConvShape(conv_node);
  node_info["dilation"] = GetValue<std::vector<int64_t>>(prim->GetAttr("dilation"));
  node_info["origin_format"] = "NHWC";
  node_info["group"] = GetValue<int64_t>(prim->GetAttr("group"));
  node_info["in_channel"] = GetValue<int64_t>(prim->GetAttr("in_channel"));
  node_info["kernel_size"] = GetValue<std::vector<int64_t>>(prim->GetAttr("kernel_size"));
  node_info["out_channel"] = GetValue<int64_t>(prim->GetAttr("out_channel"));
  node_info["stride"] = GetValue<std::vector<int64_t>>(prim->GetAttr("stride"));
  std::vector<nlohmann::json> prev_infos;
  if (former_conv_nodes.empty()) {
    nlohmann::json prev_info;
    prev_info["op_id"] = -1;
    prev_info["fixed_format"] = "Any";
    (void)prev_infos.emplace_back(prev_info);
  } else {
    for (auto iter : former_conv_nodes) {
      nlohmann::json prev_info;
      prev_info["op_id"] = std::find(conv_list.begin(), conv_list.end(), iter.first) - conv_list.begin();
      if (iter.second) {
        // has black list node
        prev_info["fixed_format"] = "NHWC";
      } else {
        prev_info["fixed_format"] = "Any";
      }
      (void)prev_infos.emplace_back(prev_info);
    }
  }
  node_info["pre_nodes"] = prev_infos;

  return node_info;
}

void TuneProcess(const std::string &json_file_name, const std::string &res_file_name, const std::string &akg_path) {
  std::ostringstream py_cmd;
  const auto &flags = GraphKernelFlags::GetInstance();
  py_cmd << kAddAkgPath;
  py_cmd << "import auto_tune\n";
  py_cmd << "auto_tune.tune_layout(\'" << json_file_name << "\', \'" << res_file_name << "\', "
         << flags.cpu_refer_thread_num << ")\n";
  std::string cmd = "python -c \"" + py_cmd.str() + "\"";
  MS_LOG(INFO) << "GraphKernel conv tuning content: \n" << cmd;
  auto ret = system(cmd.c_str());
  if (!WIFEXITED(ret)) {
    MS_LOG(ERROR) << "Tune process start fail! process content is as follows:\n" << cmd;
  }
  if (WEXITSTATUS(ret) != 0) {
    MS_LOG(ERROR) << "Failed to tune json: " << json_file_name;
  }
}

void SetTuneAttrs(const AnfNodePtrList &conv_list, const std::string &res_file) {
  std::ifstream f(res_file);
  if (!f.is_open()) {
    MS_LOG(WARNING) << "No conv tuning results!";
    return;
  }
  nlohmann::json tune_res;
  f >> tune_res;
  f.close();
  for (auto op_info : tune_res["graph"]) {
    auto prim = GetCNodePrimitive(conv_list[op_info["op_id"]]);
    prim->set_attr("tuned_src_format", MakeValue(std::string(op_info["src_format"])));
    prim->set_attr("tuned_dst_format", MakeValue(std::string(op_info["dst_format"])));
    prim->set_attr("tuned_dim", MakeValue(std::string(op_info["tuned_attrs"]["dim"])));
    prim->set_attr("akg_num_threads", MakeValue(int64_t(op_info["tuned_attrs"]["akg_num_threads"])));
  }
}

void TuneConvOps(const AnfNodePtrList &conv_list) {
  auto dir_path = FileUtils::CreateNotExistDirs(std::string("./conv_tune"));
  if (!dir_path.has_value()) {
    MS_LOG(ERROR) << "Failed to CreateNotExistDirs: ./conv_tune, start tuning failed";
    return;
  }
  nlohmann::json tune_info;
  std::vector<nlohmann::json> conv_infos;
  for (auto &conv_node : conv_list) {
    MS_EXCEPTION_IF_NULL(conv_node);
    mindspore::HashSet<AnfNodePtr> visited;
    std::function<void(AnfNodePtr)> dfs;
    std::map<AnfNodePtr, bool> former_conv_nodes;
    dfs = [&dfs, &former_conv_nodes, &visited](const AnfNodePtr &node) {
      (void)visited.insert(node);
      auto cnode = node->cast<CNodePtr>();
      if (cnode != nullptr) {
        auto inputs = cnode->inputs();
        bool has_black_node = false;
        for (size_t i = 1; i < inputs.size(); i++) {
          if (inputs[i]->cast<CNodePtr>() == nullptr || visited.count(inputs[i]) != 0) {
            continue;
          } else if (IsPrimitiveCNode(inputs[i], prim::kPrimConv2DFusion) && !IsInvalidConv(inputs[i])) {
            former_conv_nodes[inputs[i]] = has_black_node;
            has_black_node = false;
            continue;
          } else {
            if (IsBlackListOp(inputs[i])) {
              has_black_node = true;
            }
            dfs(inputs[i]);
          }
        }
      }
    };
    dfs(conv_node);
    (void)conv_infos.emplace_back(GenTuneInfo(conv_node, former_conv_nodes, conv_list));
  }
  tune_info["graph"] = conv_infos;
  tune_info["backend"] = "cpu";
  tune_info["feature"] = common::GetEnv("MS_CPU_FEATURE");
  std::string input_file = dir_path.value() + "/input.json";
  std::string output_file = dir_path.value() + "/output.json";
  std::string akg_path = dir_path.value() + "/akg_path.txt";
  std::ofstream fout(input_file, std::ios::trunc);
  fout << tune_info.dump() << std::endl;
  fout.close();
  TuneProcess(input_file, output_file, akg_path);
  SetTuneAttrs(conv_list, output_file);
}

std::vector<PrimitivePtr> ConvTuningExpander::InitOpList() {
  auto expand_only_list = GraphKernelFlags::GetInstance().enable_expand_ops_only;
  auto conv_expand_list = GraphKernelExpanderLite::ConvTuningExpanderOps();
  if (expand_only_list.empty()) {
    return conv_expand_list;
  }
  std::vector<PrimitivePtr> conv_only_list;
  for (auto conv_expand : conv_expand_list) {
    if (std::find(expand_only_list.begin(), expand_only_list.end(), conv_expand->name()) != expand_only_list.end()) {
      conv_only_list.emplace_back(conv_expand);
    }
  }
  return conv_only_list;
}

bool ConvTuningExpander::Run(const FuncGraphPtr &func_graph) {
  if (GraphKernelExpanderLite::DisableConvTuning()) {
    return false;
  }
  bool changed = false;
  auto valid_op_list = InitOpList();
  if (std::find(valid_op_list.begin(), valid_op_list.end(), prim::kPrimConv2DFusion) != valid_op_list.end()) {
    MS_EXCEPTION_IF_NULL(func_graph);
    MS_EXCEPTION_IF_NULL(func_graph->get_return());
    auto todos = TopoSort(func_graph->get_return());
    AnfNodePtrList conv_list;
    for (auto &node : todos) {
      if (IsPrimitiveCNode(node, prim::kPrimConv2DFusion) && !IsInvalidConv(node)) {
        (void)conv_list.emplace_back(node->cast<CNodePtr>());
        changed = true;
      }
    }
    TuneConvOps(conv_list);
  }
  changed = GraphKernelExpanderLite::Run(func_graph) || changed;
  return changed;
}
}  // namespace mindspore::graphkernel
