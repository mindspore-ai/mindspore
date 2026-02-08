/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/parallel_optimizer/opt_param_mgr.h"
#include <string>
#include <vector>
#include <functional>
#include <map>
#include <memory>
#include "frontend/parallel/ops_info/operator_info.h"
#include "include/utils/parallel_context.h"
#include "ir/dtype/type_id.h"

namespace mindspore {
namespace parallel {
class OptParamMgrImpl : public OptParamMgr {
 public:
  explicit OptParamMgrImpl(const FuncGraphPtr &root) : root_(root) {}
  virtual ~OptParamMgrImpl() = default;
  std::string ShardOptGroup(const AnfNodePtr &parameter, TensorLayout *const tensor_layout,
                            const OperatorInfoPtr &distribute_operator) const override {
    if (!SplitParam(parameter)) {
      return "";
    }

    Status ret = tensor_layout->GenerateOptShardSliceShape();
    if (ret != Status::SUCCESS) {
      MS_LOG(WARNING) << parameter->ToString() << "'s distributed shape " << tensor_layout->slice_shape().ToString()
                      << " does not satisfy the conditions, thus not apply optimizer shard.";
      return "";
    }
    // get the shard tensor slice shape if the weight is repeated on devices
    // and the shape of the first dimension could be divided
    // apply parallel optimizer on parameters
    // create communication group for allgather operator
    std::string opt_shard_group;
    std::vector<Group> dev_group;
    MS_LOG(INFO) << "Creating shard group for param: " << parameter->ToString()
                 << ", shape: " << parameter->Shape()->ToString();
    if (distribute_operator->CreateGroupForOptShard(tensor_layout, &dev_group) == Status::SUCCESS &&
        !dev_group.empty()) {
      opt_shard_group = dev_group[0].name();
      MS_LOG(INFO) << "create group success.";
    } else {
      MS_LOG(WARNING) << "create opt shard group for the parameter " << parameter->ToString() << " failed.";
    }
    return opt_shard_group;
  }

 private:
  size_t ComputeShapeSize(const AnfNodePtr &parameter) const {
    auto param_shape = parameter->Shape();
    MS_EXCEPTION_IF_NULL(param_shape);
    auto shape_ptr = param_shape->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_ptr);
    ShapeVector shape(shape_ptr->shape());
    size_t total_size = std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
    return total_size;
  }

  // unit: B
  size_t ComputeMemorySize(const AnfNodePtr &parameter) const {
    // key, value: typeid, bytes
    const std::map<TypeId, size_t> dtype_size_map = {
      {kNumberTypeBool, sizeof(bool)},       {kNumberTypeInt4, sizeof(int8_t)},
      {kNumberTypeInt8, sizeof(int8_t)},     {kNumberTypeInt16, sizeof(int16_t)},
      {kNumberTypeInt32, sizeof(int32_t)},   {kNumberTypeInt64, sizeof(int64_t)},
      {kNumberTypeFloat16, sizeof(float16)}, {kNumberTypeFloat32, sizeof(float)},
      {kNumberTypeFloat64, sizeof(double)},  {kNumberTypeUInt8, sizeof(uint8_t)},
      {kNumberTypeUInt16, sizeof(uint16_t)}, {kNumberTypeUInt32, sizeof(uint32_t)},
      {kNumberTypeUInt64, sizeof(uint64_t)}, {kNumberTypeBFloat16, sizeof(bfloat16)}};

    size_t shape_size = ComputeShapeSize(parameter);
    auto tensor_type = parameter->Type()->cast<mindspore::TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    TypeId type_id = tensor_type->element()->type_id();
    if (dtype_size_map.find(type_id) == dtype_size_map.end()) {
      MS_LOG_WITH_NODE(EXCEPTION, parameter) << "unsupported type of parameter: " << parameter->DebugString();
    }
    size_t type_size = dtype_size_map.find(type_id)->second;
    return shape_size * type_size;
  }

  int64_t GetThresholdFromUsrInput() const {
    return ParallelContext::GetInstance()->get_parallel_optimizer_threshold();
  }

  bool SplitParam(const AnfNodePtr &parameter) const {
    if (!ParallelContext::GetInstance()->enable_parallel_optimizer()) {
      MS_LOG(INFO) << "Parallel optimizer: feature is not enabled. Skipped.";
      return false;
    }

    auto param_ptr = parameter->cast<ParameterPtr>();
    if ((param_ptr == nullptr) || (!param_ptr->has_default())) {
      MS_LOG(INFO) << "Parallel optimizer: " << parameter->ToString() << " is not a parameter.";
      return false;
    }

    if (param_ptr->param_info() != nullptr && !param_ptr->param_info()->parallel_optimizer()) {
      MS_LOG(INFO) << "Parallel optimizer: " << parameter->ToString() << " is manually set skipped.";
      return false;
    }

    size_t param_split_threshold = DEFAULT_VAL * KB_SIZE;
    int64_t user_define_threshold = GetThresholdFromUsrInput();
    if (user_define_threshold != -1) {
      MS_LOG(INFO) << "Parallel optimizer: use user-define threshold = " << user_define_threshold << "KB.";
      param_split_threshold = user_define_threshold * KB_SIZE;
    } else {
      MS_LOG(INFO) << "Parallel optimizer: use DEFAULT threshold = " << DEFAULT_VAL << "KB.";
    }

    size_t param_size = ComputeMemorySize(parameter);
    MS_LOG(INFO) << "Parallel optimizer: " << parameter->ToString() << " size = " << param_size << "B";
    if (param_size < param_split_threshold) {
      MS_LOG(INFO) << "Parallel optimizer: the size of " << parameter->ToString() << "(" << param_size
                   << "KB) is smaller than the threshold(" << param_split_threshold << "B). Skipped.";
      MS_EXCEPTION_IF_NULL(param_ptr->param_info());
      param_ptr->param_info()->set_parallel_optimizer(false);
      return false;
    }
    return true;
  }

  FuncGraphPtr root_;
  size_t DEFAULT_VAL = 64;  // unit: KB
  size_t KB_SIZE = 1024;
};

std::unique_ptr<OptParamMgr> createOptParamMgr(const FuncGraphPtr &root) {
  return std::make_unique<OptParamMgrImpl>(root);
}
}  // namespace parallel
}  // namespace mindspore
