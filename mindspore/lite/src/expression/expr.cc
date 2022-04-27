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

#include <string>
#include <iostream>
#include "src/expression/expr.h"
#include "src/expression/node.h"

namespace mindspore {
namespace lite {
std::string EXPR::name() { return node_->name(); }
void EXPR::Travers(std::function<bool(EXPR *e, EXPR *itr)> cb) {
  if (!visited) {
    visited = true;
    for (auto &itr : params_) {
      if (cb(this, itr)) {
        itr->Travers(cb);
      }
    }
  }
}

void EXPR::Replace(EXPR **old, EXPR **n, std::vector<Node *> *to_delete) {
  if (!visited) {
    visited = true;
    for (auto &itr : params_)
      if (itr == *old) {
        to_delete->push_back(itr->node());
        itr = *n;
      }
    for (auto &itr : params_) itr->Replace(old, n, to_delete);
  }
}

void EXPR::Replace(const std::vector<EXPR *> &vec, std::vector<EXPR *> *old, std::vector<EXPR *> *n) {
  std::vector<Node *> to_delete;
  for (auto &e : vec) {
    for (std::size_t i = 0; i < old->size(); i++) {
      e->Replace(&old->at(i), &n->at(i), &to_delete);
    }
  }
  for (auto &itr : to_delete) delete itr;
  for (auto e : vec) e->Clear();
}

void EXPR::Clear() {
  EXPR *item = this;
  if (visited == false) return;
  visited = false;
  while (item->params_.size() == 1) {
    item = item->params_.front();
    if (item->visited == false) return;
    item->visited = false;
  }
  for (auto &itr : item->params_) itr->Clear();
}

void EXPR::Clear(std::vector<EXPR *> vec) {
  for (auto e : vec) e->Clear();
}

void EXPR::CreateOutputMap(std::vector<EXPR *> vec, std::map<EXPR *, std::list<EXPR *>> *outmap) {
  for (auto e : vec) {
    e->Travers([&](EXPR *e, EXPR *itr) {
      (*outmap)[itr].push_back(e);
      return true;
    });
  }
  Clear(vec);
}

void EXPR::PrintDot(std::vector<EXPR *> vec) {
  std::cout << "digraph \"expr\" { " << std::endl;
  for (auto e : vec) {
    e->Travers([](EXPR *e, EXPR *itr) {
      std::cout << "\"" << itr->node_->name() << "\"->"
                << "\"" << e->node_->name() << "\"" << std::endl;
      return true;
    });
  }
  std::cout << "}" << std::endl;
  Clear(vec);
}
}  // namespace lite
}  // namespace mindspore
