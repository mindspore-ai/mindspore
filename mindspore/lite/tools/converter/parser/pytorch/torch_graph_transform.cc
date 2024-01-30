/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <memory>
#include <string>
#include <vector>

#include "torch/csrc/jit/passes/freeze_module.h"
#include "torch/csrc/jit/passes/inliner.h"
#include "torch/csrc/jit/passes/remove_mutation.h"
#include "torch/csrc/jit/passes/normalize_ops.h"
#include "mindspore/core/utils/log_adapter.h"
#include "mindspore/lite/tools/converter/parser/pytorch/pytorch_node_parser.h"

namespace torch {
namespace jit {
void OutputsUnpack(Graph *graph) {
  std::function<void(Node * tuple, std::vector<Node *> &, std::vector<Value *> &)> flattenTuple =
    [&flattenTuple](Node *tuple, std::vector<Node *> &tuples, std::vector<Value *> &values) -> void {
    tuples.push_back(tuple);
    for (auto input : tuple->inputs()) {
      auto node = input->node();
      if (node->kind() == prim::TupleConstruct) {
        flattenTuple(node, tuples, values);
      } else {
        values.push_back(input);
      }
    }
  };
  for (size_t i = 0; i < graph->outputs().size(); i++) {
    auto node = graph->outputs()[i]->node();
    // unpack output
    switch (node->kind()) {
      case prim::TupleConstruct: {
        std::vector<Node *> tuples;
        std::vector<Value *> values;
        flattenTuple(node, tuples, values);
        for (auto realOutput : values) {
          graph->registerOutput(realOutput);
        }
        graph->eraseOutput(i);
        for (auto tuple : tuples) {
          if (!tuple->hasUses()) {
            tuple->destroy();
          }
        }
        break;
      }
      case prim::DictConstruct: {
        graph->registerOutput(node->input(1));
        graph->eraseOutput(i);
        node->destroy();
        break;
      }
      case prim::ListConstruct: {
        for (size_t j = 0; i < node->inputs().size(); j++) {
          graph->registerOutput(node->input(j));
        }
        graph->eraseOutput(i);
        node->destroy();
        break;
      }
      default: {
        MS_LOG(INFO) << "skip " << mindspore::lite::PytorchNodeParser::GetTorchNodeType(node);
        break;
      }
    }
  }
}

void FuseListUnpack(Block *block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    auto *node = *it;
    it++;

    for (Block *sub_block : node->blocks()) {
      FuseListUnpack(sub_block);
    }
    std::set<NodeKind> fusekind = {
      aten::split,
      aten::split_with_sizes,
      aten::split_with_sizes,
      aten::unsafe_split_with_sizes,
      aten::unbind,
      aten::chunk,
      aten::unsafe_chunk,
      aten::where,
    };
    if (fusekind.count(it->kind()) && it->outputs().size() == 1 && it->output()->uses().size() == 1) {
      const auto listunpack = it->output()->uses()[0].user;
      if (listunpack->kind() == prim::ListUnpack) {
        for (size_t i = 0; i < listunpack->outputs().size(); ++i) {
          auto new_output = it->addOutput();
          new_output->copyMetadata(listunpack->output(i));
        }
        listunpack->removeAllInputs();
        it->eraseOutput(0);
        listunpack->replaceAllUsesWith(*it);
        listunpack->destroy();
      }
    }
  }
}

/*
   Remove all ListConstruct op with only one input and not used by aten::cat, like below:
        %116 : Tensor?[] = prim::ListConstruct(%115)
        %alpha0.1 : Tensor = aten::index_put_(%alpha.1, %116, %x.1, %16)
   ListConstruct used by aten::cat will be reserved like below:
        %features.2 : Tensor[] = prim::ListConstruct(%input3.4)
        %concated_features.380 : Tensor = aten::cat(%features.2, %5)
   Attention: Running this pass after removeListAppend
 */
void RemoveListConstructOps(Block *block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end; ++it) {
    if (it->kind() == prim::ListConstruct && it->inputs().size() == 1) {
      bool remove = true;
      for (auto use : it->output()->uses()) {
        if (use.user->kind() == aten::cat) {
          remove = false;
          break;
        }
      }
      if (remove) {
        it->output()->replaceAllUsesWith(it->input(0));
        it->removeInput(0);
        it.destroyCurrent();
      }
    }
  }
}

// flatten tuple input and remove tuple unpack
bool FlattenInputsTuple(Graph *graph) {
  for (size_t i = 0; i < graph->inputs().size(); i++) {
    auto input_value = graph->inputs()[i];
    auto tuple = input_value->type()->cast<at::TupleType>();
    if (!tuple) {
      continue;
    }

    auto use_list = input_value->uses();
    if (use_list.size() != 1) {
      MS_LOG(ERROR) << "current pass only supports tuple input has only one user!";
      return false;
    }
    auto tuple_unpack = use_list[0].user;
    auto node_type = mindspore::lite::PytorchNodeParser::GetTorchNodeType(tuple_unpack);
    if (node_type != "TupleUnpack") {
      MS_LOG(ERROR) << "unsupported node user type of tuple: " << node_type;
      return false;
    }

    auto elements = tuple->elements();
    size_t idx = 0;
    for (auto &element : elements) {
      auto new_input = graph->addInput(tuple_unpack->output(idx)->debugName());
      new_input->setType(element);

      auto tuple_item = tuple_unpack->output(idx);
      auto item_use_list = tuple_item->uses();
      for (const auto &use : item_use_list) {
        use.user->replaceInputWith(tuple_item, new_input);
      }
      idx++;
    }
    tuple_unpack->destroy();
    graph->eraseInput(i);
  }
  return true;
}

std::shared_ptr<Graph> TorchGraphTransform(Module *module) {
  module->eval();                                 // eval to expand function call
  auto mod = torch::jit::freeze_module(*module);  // freeze module
  auto torch_graph = mod.get_method("forward").graph();
  if (torch_graph == nullptr) {
    return nullptr;
  }
  // parse submodules in graph
  torch::jit::Inline(*torch_graph);
  torch::jit::NormalizeOps(torch_graph);

  RemoveListConstructOps(torch_graph->block());
  FlattenInputsTuple(torch_graph.get());
  FuseListUnpack(torch_graph->block());

  OutputsUnpack(torch_graph.get());
  return torch_graph;
}
}  // namespace jit
}  // namespace torch
