/**

    Copyright 2023 Huawei Technologies Co., Ltd
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    */

#include <set>
#include <vector>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include "pipeline/jit/pi/graph_capture/code_generator.h"
#include "pipeline/jit/pi/graph_capture/side_effect.h"

namespace mindspore {
namespace pijit {

void SideEffect::CleanSideEffects(int new_bci) {
  for (auto item : GetSideEffectNodes()) {
    if (item->bci() >= new_bci) {
      GetSideEffectNodes().erase(std::remove(GetSideEffectNodes().begin(), GetSideEffectNodes().end(), item),
                                 GetSideEffectNodes().end());
    }
  }
}

void SideEffect::ConvertReplaceList() {
  for (auto item : replace_map) {
    replace_list.push_back(item.first);
  }
}

std::vector<ValueNode *> SideEffect::CollectSideEffectAliveNodes() const {
  std::vector<ValueNode *> alive_nodes;
  for (auto item : GetSideEffectNodes()) {
    if (item->GetOpcode() == BUILD_LIST) {
      alive_nodes.push_back(item);
      if (GetReplaceMap().size() != 0) {
        for (auto &replace_map : GetReplaceMap()) {
          if (item == replace_map.first) {
            alive_nodes.push_back(replace_map.second);
          }
        }
      }
    } else if (item->GetOpcode() == CALL_FUNCTION) {
      if (item->getInputs().size() != 0) {
        for (auto input_item : item->getInputs()) {
          if (input_item->GetOpcode() == CALL_FUNCTION) {
            continue;
          }
          alive_nodes.push_back(input_item);
        }
      }
    }
  }
  if (GetReplaceList().size() != 0) {
    auto replace_list = GetReplaceList();
    alive_nodes.insert(alive_nodes.end(), replace_list.begin(), replace_list.end());
  }

  for (auto item : GetGlobalList()) {
    if (item.getNode() != nullptr) {
      alive_nodes.push_back(item.getNode());
    }
  }
  return alive_nodes;
}

void SideEffect::Merge(const std::unique_ptr<SideEffect> &sub_side_effect) {
  for (auto item : sub_side_effect->GetSideEffectNodes()) {
    SetSideEffectNode(item);
  }
  for (auto item : sub_side_effect->GetReplaceMap()) {
    SetReplaceMap(item.first, item.second);
  }
  sub_side_effect->ConvertReplaceList();
  for (auto item : sub_side_effect->GetReplaceList()) {
    SetReplaceList(item);
  }

  for (auto item : sub_side_effect->GetGlobalList()) {
    SetGlobalList(item);
  }
}

void SideEffect::RestoreSideEffect(CodeGenerator *code_gen) const {
  for (auto &item : GetSideEffectNodes()) {
    if (item->GetOpcode() == BUILD_LIST) {
      if (GetReplaceMap().size() != 0) {
        for (auto &replace_map : GetReplaceMap()) {
          std::cout << replace_map.first->ToString() << std::endl;
          std::cout << replace_map.second->ToString() << std::endl;

          if (item == replace_map.first) {
            code_gen->LoadValue(replace_map.second);
          }
        }
      } else {
        code_gen->NewInstr(LOAD_FAST, 0);
      }

      code_gen->LoadValue(item);
      code_gen->NewInstr(LOAD_CONST, 0);
      code_gen->GetCode().co_code.back()->set_cnst(py::none());
      code_gen->NewInstr(LOAD_CONST, 0);
      code_gen->GetCode().co_code.back()->set_cnst(py::none());
      code_gen->NewInstr(BUILD_SLICE, 0);
      code_gen->NewInstr(STORE_SUBSCR, 0);
    } else if (item->GetOpcode() == CALL_FUNCTION) {
      for (auto input : item->getInputs()) {
        if (input->GetOpcode() == CALL_FUNCTION) {
          continue;
        }
        code_gen->LoadValue(input);
      }
      code_gen->NewInstr(item->GetOpcode(), item->GetOparg());
    }
  }

  for (auto const &item : GetGlobalList()) {
    if (item.getNode() != nullptr) {
      code_gen->LoadValue(item.getNode());
      code_gen->GetCode().co_code.back()->set_name(item.getName());
      code_gen->NewInstr(STORE_GLOBAL, 0);
    } else {
      code_gen->NewInstr(DELETE_GLOBAL, 0);
    }
  }
}

}  // namespace pijit
}  // namespace mindspore
