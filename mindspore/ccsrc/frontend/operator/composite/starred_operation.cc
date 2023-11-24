/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#include "frontend/operator/composite/starred_operation.h"
#include <algorithm>
#include <vector>
#include <utility>

#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/array_ops.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
using mindspore::abstract::AbstractBase;
using mindspore::abstract::AbstractScalar;
using mindspore::abstract::AbstractSequence;
using mindspore::abstract::AbstractSequencePtr;
using mindspore::abstract::AbstractTensor;
using mindspore::abstract::AbstractTuple;

// x = (1, 2, 3, 4)
// a, *b, c = x    // targets(a, *b, c) = assign(x)
// a = 1, *b = [2, 3], c = 4
// convert:
// StarredGetItem(sequence, position_in_target, targets_num)
// *b: StarredGetItem(x, 1, 3)
// output: *b = makelist(getitem(x, 1), getitem(x, 2))
FuncGraphPtr StarredGetItem::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  // Check inputs
  constexpr size_t starred_getitem_args_size = 3;
  constexpr size_t sequence_index = 0;
  constexpr size_t position_in_target_index = 1;
  constexpr size_t targets_num_index = 2;

  if (args_abs_list.size() != starred_getitem_args_size) {
    MS_LOG(EXCEPTION) << "For 'StarredGetItem', the number of input should be " << starred_getitem_args_size
                      << ", but got " << args_abs_list.size();
  }

  auto first_input__abs = args_abs_list[sequence_index];
  MS_EXCEPTION_IF_NULL(first_input__abs);
  if (!first_input__abs->isa<AbstractSequence>()) {
    MS_LOG(EXCEPTION) << "The first input of StarredGetItem operation must be sequence, but got "
                      << first_input__abs->ToString();
  }
  auto seq_abs = first_input__abs->cast<AbstractSequencePtr>();
  const auto &elements = seq_abs->elements();
  size_t elements_size = elements.size();

  auto pos_abs = args_abs_list[position_in_target_index];
  MS_EXCEPTION_IF_NULL(pos_abs);
  int64_t position_in_target = GetValue<int64_t>(pos_abs->GetValueTrack());

  auto targets_num_abs = args_abs_list[targets_num_index];
  MS_EXCEPTION_IF_NULL(targets_num_abs);
  int64_t targets_num = GetValue<int64_t>(targets_num_abs->GetValueTrack());

  FuncGraphPtr ret_graph = std::make_shared<FuncGraph>();
  ret_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);

  std::vector<AnfNodePtr> make_list_inputs;
  make_list_inputs.push_back(NewValueNode(prim::kPrimMakeList));
  int64_t list_input_num = elements_size - (targets_num - 1);
  auto assign_node = ret_graph->add_parameter();

  for (int64_t index = 0; index < list_input_num; ++index) {
    auto get_item_prim = NewValueNode(prim::kPrimTupleGetItem);
    std::vector<AnfNodePtr> get_item_inputs{get_item_prim, assign_node};
    auto index_value = NewValueNode(static_cast<int64_t>(position_in_target + index));
    get_item_inputs.push_back(index_value);
    auto get_item = ret_graph->NewCNodeInOrder(get_item_inputs);
    make_list_inputs.push_back(get_item);
  }

  for (size_t idx = 0; idx < args_abs_list.size() - 1; idx++) {
    (void)ret_graph->add_parameter();
  }

  auto list_out = ret_graph->NewCNodeInOrder(make_list_inputs);
  ret_graph->set_output(list_out);
  return ret_graph;
}

// x = [1, 2, 3, 4]
// a = *x,    // targets(a) = assign(*x,)
// a = (1, 2, 3, 4)
// convert:
// StarredUnpackMerge(StarredUnpack(sequence))
// StarredUnpackMerge(((1, 2, 3, 4), )
// StarredUnpackMerge(tuple_getitem(x, 0), ...)
FuncGraphPtr StarredUnpack::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  // Check inputs
  constexpr size_t starred_unpack_args_size = 1;
  constexpr size_t sequence_index = 0;
  if (args_abs_list.size() != starred_unpack_args_size) {
    MS_LOG(EXCEPTION) << "For 'StarredUnpack', the number of input should be " << starred_unpack_args_size
                      << ", but got " << args_abs_list.size();
  }
  auto &unpack_arg = args_abs_list[sequence_index];
  MS_EXCEPTION_IF_NULL(unpack_arg);
  FuncGraphPtr ret_graph = std::make_shared<FuncGraph>();
  ret_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  if (unpack_arg->isa<AbstractScalar>()) {
    auto arg_scalar = dyn_cast_ptr<AbstractScalar>(unpack_arg);
    const auto &arg_value = arg_scalar->GetValueTrack();
    if (arg_value->isa<StringImm>()) {
      auto str = arg_value->cast_ptr<StringImm>();
      MS_EXCEPTION_IF_NULL(str);
      std::string str_value = str->value();
      AbstractBasePtrList ptr_list;
      for (size_t index = 0; index < str_value.size(); ++index) {
        std::stringstream stream;
        stream << str_value[index];
        string index_str = stream.str();
        auto index_abs = std::make_shared<AbstractScalar>(static_cast<std::string>(index_str));
        ptr_list.push_back(index_abs);
      }
      auto tuple_abs = std::make_shared<abstract::AbstractTuple>(ptr_list);
      auto unpack_node = ret_graph->add_parameter();
      unpack_node->set_abstract(tuple_abs);
      ret_graph->set_output(unpack_node);
      return ret_graph;
    }
  } else if (unpack_arg->isa<AbstractSequence>()) {
    auto seq = args_abs_list[0]->cast<AbstractSequencePtr>();
    const auto &elements = seq->elements();
    auto tuple_abs = std::make_shared<abstract::AbstractTuple>(elements);
    auto unpack_node = ret_graph->add_parameter();
    unpack_node->set_abstract(tuple_abs);
    ret_graph->set_output(unpack_node);
    return ret_graph;
  } else if (unpack_arg->isa<AbstractTensor>()) {
    auto input = ret_graph->add_parameter();
    auto prim = prim::kPrimUnstack;
    auto unstack_node = ret_graph->NewCNodeInOrder({NewValueNode(prim), input});
    prim->set_attr(kAttrAxis, MakeValue(static_cast<int64_t>(0)));
    ret_graph->set_output(unstack_node);
    return ret_graph;
  }
  MS_LOG(INTERNAL_EXCEPTION) << "The object is not iterable, " << unpack_arg->ToString();
}

std::pair<std::vector<int64_t>, int64_t> StarredUnpackMerge::GetStarredUnpackMergeFlags(
  const AbstractBasePtrList &args_abs_list) {
  constexpr size_t args_size = 3;
  constexpr size_t flags_num = 2;
  size_t starred_flags_index = args_abs_list.size() - 2;
  size_t is_tuple_index = args_abs_list.size() - 1;

  if (args_abs_list.size() < args_size) {
    MS_LOG(EXCEPTION) << "For 'StarredUnpackMerge', the number of input should be " << args_size
                      << " at least, but got " << args_abs_list.size();
  }
  if (!args_abs_list[starred_flags_index]->isa<AbstractSequence>()) {
    MS_LOG(EXCEPTION) << "For 'StarredUnpackMerge', the starred_flags input should be sequence, but got "
                      << args_abs_list[starred_flags_index]->ToString();
  }
  if (!args_abs_list[is_tuple_index]->isa<AbstractScalar>()) {
    MS_LOG(EXCEPTION) << "For 'StarredUnpackMerge', the is_tuple input should be scalar, but got "
                      << args_abs_list[is_tuple_index]->ToString();
  }

  auto abs_seq = args_abs_list[starred_flags_index]->cast<AbstractSequencePtr>();
  const auto &elements = abs_seq->elements();
  std::vector<int64_t> starred_flags(elements.size(), 0);
  for (size_t index = 0; index < elements.size(); ++index) {
    auto ele = elements[index];
    auto ele_value = ele->GetValueTrack();
    auto val = GetValue<int64_t>(ele_value);
    starred_flags[index] = val;
  }
  int64_t is_tuple = GetValue<int64_t>(args_abs_list[is_tuple_index]->GetValueTrack());
  size_t sequence_input_size = args_abs_list.size() - flags_num;
  if (sequence_input_size != elements.size()) {
    MS_LOG(EXCEPTION) << "For 'StarredUnpackMerge', the input is wrong, please check.";
  }
  return {starred_flags, is_tuple};
}

// a = *[1, 2], (3, 4)
// convert:
// StarredUnpackMerge(assign_node1, assign_node2, starred_flags_node, is_tuple)
// StarredUnpackMerge(StarredUnpack(*[1, 2]), (3, 4), (1, 0), 1) --> (1, 2, (3, 4))
// a: (1, 2, (3, 4))
FuncGraphPtr StarredUnpackMerge::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  // Check inputs, and get flags info.
  auto [starred_flags, is_tuple] = GetStarredUnpackMergeFlags(args_abs_list);

  FuncGraphPtr ret_graph = std::make_shared<FuncGraph>();
  ret_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  std::vector<AnfNodePtr> new_inputs;
  if (is_tuple == 1) {
    new_inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
  } else if (is_tuple == 0) {
    new_inputs.push_back(NewValueNode(prim::kPrimMakeList));
  }
  constexpr size_t unpack_flags_num = 2;
  for (size_t index = 0; index < args_abs_list.size() - unpack_flags_num; ++index) {
    auto &unpack_arg = args_abs_list[index];
    MS_EXCEPTION_IF_NULL(unpack_arg);
    int64_t is_starred = starred_flags[index];
    auto input = ret_graph->add_parameter();
    if (!is_starred) {
      new_inputs.push_back(input);
    } else {
      // starred must be sequence.
      if (!unpack_arg->isa<AbstractSequence>()) {
        MS_LOG(EXCEPTION) << "The starred unpack merge input must be sequence, but got " << unpack_arg->ToString();
      }
      auto unpack_abs_seq = unpack_arg->cast<AbstractSequencePtr>();
      const auto &elements = unpack_abs_seq->elements();
      size_t unpack_size = elements.size();
      for (size_t ele_index = 0; ele_index < unpack_size; ++ele_index) {
        std::vector<AnfNodePtr> get_item_inputs{NewValueNode(prim::kPrimTupleGetItem), input};
        get_item_inputs.push_back(NewValueNode(static_cast<int64_t>(ele_index)));
        auto get_item = ret_graph->NewCNodeInOrder(get_item_inputs);
        new_inputs.push_back(get_item);
      }
    }
  }
  for (size_t index = 0; index < unpack_flags_num; ++index) {
    (void)ret_graph->add_parameter();
  }
  auto new_node = ret_graph->NewCNodeInOrder(new_inputs);
  ret_graph->set_output(new_node);
  return ret_graph;
}
}  // namespace prim
}  // namespace mindspore
