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

#include "mindspore/core/symbolic_shape/math_info.h"
#include "mindspore/core/symbolic_shape/symbol.h"

namespace mindspore {
namespace symshape {
IntSymbolPtr MathInfo::ToIntSymbol() const { return int_symbol_->as_sptr<IntSymbol>(); }

void MathInfo::UpdateExprRoot() const {
  if (relation_expr_.s == nullptr) {
    // "this" is the root node
    return;
  }
  if (relation_expr_.s->math_info_.relation_expr_.s == nullptr) {
    // "this" is second layer node, do not need to update
    return;
  }
  // "this" is third or lower layer node, link it to the root node.
  auto new_parent = relation_expr_.s->math_info_.root();
  const auto &old_parent_expr = relation_expr_.s->math_info_.relation_expr_;
  relation_expr_.b = relation_expr_.a * old_parent_expr.b + relation_expr_.b;
  relation_expr_.a = relation_expr_.a * old_parent_expr.a;
  relation_expr_.s = new_parent;
}

void MathInfo::SetEqual(const IntSymbolPtr &other) {
  // this:  s2 = s1 * a + b  -->  s1 = (s2 - b) / a  -->  s1 = (s2 / a) - (b / a)
  // other: s4 = s3 * c + d  -->  s3 = (s4 - d) / c  -->  s3 = (s4 / c) - (d / c)
  // set "s2 == s4", that's to set "s3 = (s2 / c) - (d / c)" or "s1 = (s4 / a) - (b / a)"
  MS_EXCEPTION_IF_NULL(other);
  if (this == &other->math_info_) {
    return;
  }
  auto s1 = root();
  auto s3 = other->math_info_.root();
  if (s1 == s3) {
    // the two symbols are already in the same tree.
    return;
  }
  auto s2 = ToIntSymbol();
  auto s4 = other;
  if (s1->math_info_.math_info_id_ < s3->math_info_.math_info_id_) {
    MS_LOG(DEBUG) << "Set " << s4->ToString() << " equals to " << s2->ToString();
    if (s4->math_info_.relation_expr_.s == nullptr) {
      s4->math_info_.relation_expr_ = {s2, kFrac1, kFrac0};
    } else {
      s3->math_info_.relation_expr_ = {s2, s4->math_info_.relation_expr_.a.rec(),
                                       -(s4->math_info_.relation_expr_.b / s4->math_info_.relation_expr_.a)};
      s3->math_info_.UpdateExprRoot();
    }
  } else {  // s3->math_info_id_ < s1->math_info_id_
    MS_LOG(DEBUG) << "Set " << s2->ToString() << " equals to " << s4->ToString();
    if (relation_expr_.s == nullptr) {
      relation_expr_ = {s4, kFrac1, kFrac0};
    } else {
      s1->math_info_.relation_expr_ = {s4, s2->math_info_.relation_expr_.a.rec(),
                                       -(s2->math_info_.relation_expr_.b / s2->math_info_.relation_expr_.a)};
      s1->math_info_.UpdateExprRoot();
    }
  }
}

bool MathInfo::MathLess(const MathInfo &other) const {
  if (range_max() < other.range_min()) {
    return true;
  }
  if (range_min() > other.range_max()) {
    return false;
  }
  return (this->root() == other.root()) && (this->relation_expr_ < other.relation_expr_);
}
bool MathInfo::MathLessEqual(const MathInfo &other) const {
  if (range_max() <= other.range_min()) {
    return true;
  }
  if (range_min() > other.range_max()) {
    return false;
  }
  return (this->root() == other.root()) && (this->relation_expr_ <= other.relation_expr_);
}

void MathInfo::Dump(std::ostringstream &oss) const {
  oss << "<[";
  oss << (range_min() == -kINF ? "-inf" : range_min() == kINF ? "inf" : std::to_string(range_min())) << ",";
  oss << (range_max() == -kINF ? "-inf" : range_max() == kINF ? "inf" : std::to_string(range_max())) << "]";
  if (relation_expr_.s != nullptr) {
    oss << "=" << relation_expr_.s->ToRawString();
    if (relation_expr_.a != 1) {
      if (relation_expr_.a.x() == 1) {
        oss << "/" << relation_expr_.a.y();
      } else {
        oss << "*" << relation_expr_.a;
      }
    }
    if (relation_expr_.b != 0) {
      if (relation_expr_.b.x() > 0) {
        oss << "+";
      }
      oss << relation_expr_.b;
    }
  }
  oss << ">";
}
}  // namespace symshape
}  // namespace mindspore
