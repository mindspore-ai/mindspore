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

#include "pybind_api/ops/generator_impl.h"
#include <random>
#include <memory>
#include <cstring>
#include <string>
#include "ir/tensor.h"
#include "include/utils/tensor_py.h"
#include "ops/op_def_utils.h"
#include "ops_utils/op_constants.h"
#include "include/runtime/pipeline/pipeline.h"
#include "infer/ops_func_impl/generator.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "pybind_api/ops/ops_api.h"
#include "include/securec.h"
#include "pybind11/pybind11.h"

namespace mindspore {
namespace {
void TensorCheck(const tensor::TensorPtr &tensor, const TypeId type_id, const size_t data_size,
                 const std::string &tensor_name) {
  MS_ASSERT_TRUE(tensor != nullptr) << "Invalid tensor input: " << tensor_name;
  const auto tensor_type = static_cast<TypeId>(tensor->data_type_c());
  MS_ASSERT_TRUE(tensor_type == type_id) << tensor_name << " data type must be " << TypeIdToString(type_id)
                                         << ", but got " << TypeIdToString(tensor_type);
  const auto tensor_data_size = tensor->DataSize();
  MS_ASSERT_TRUE(tensor_data_size == data_size)
    << tensor_name << " data size must be " << data_size << ", but got " << tensor_data_size;
}
}  // namespace
GeneratorImpl::GeneratorImpl(const py::handle &seed_param, const py::handle &offset_param) {
  auto seed_tensor = tensor::ConvertToTensor(seed_param);
  auto offset_tensor = tensor::ConvertToTensor(offset_param);
  TensorCheck(seed_tensor, ParamTypeId, 1, "Seed");
  TensorCheck(offset_tensor, ParamTypeId, 1, "Offset");
  seed_ = seed_tensor->cpu();
  offset_ = offset_tensor->cpu();

  seed_data_ = static_cast<param_type *>(seed_->data_c());
  offset_data_ = static_cast<param_type *>(offset_->data_c());
  MS_ASSERT_TRUE(seed_data_ != nullptr) << "Failed to get data from seed tensor, please check the input.";
  MS_ASSERT_TRUE(offset_data_ != nullptr) << "Failed to get data from offset tensor, please check the input.";
}

py::object GeneratorImpl::set_state(const py::handle &state) {
  auto new_state_tensor = tensor::ConvertToTensor(state);
  TensorCheck(new_state_tensor, StateTypeId, sizeof(param_type) * kInputNum2, "State");
  new_state_tensor = new_state_tensor->cpu();
  auto new_state_data = static_cast<state_type *>(new_state_tensor->data_c());
  {
    GilReleaseWithCheck gil_release;
    runtime::Pipeline::Get().frontend_stage()->Wait();
    runtime::Pipeline::Get().backend_stage()->Wait();
    auto ret = memcpy_s(seed_data_, sizeof(param_type), new_state_data, sizeof(param_type));
    if (ret != EOK) {
      MS_EXCEPTION(ValueError) << "Failed to copy seed data, memcpy_s error. Error no: " << ret;
    }
    ret = memcpy_s(offset_data_, sizeof(param_type), new_state_data + sizeof(param_type), sizeof(param_type));
    if (ret != EOK) {
      MS_EXCEPTION(ValueError) << "Failed to copy offset data, memcpy_s error. Error no: " << ret;
    }
  }
  return py::none();
}

py::object GeneratorImpl::get_state() const {
  const ShapeVector state_shape = {sizeof(param_type) * 2};
  auto state = std::make_shared<tensor::Tensor>(StateTypeId, state_shape)->cpu();
  auto state_data = static_cast<state_type *>(state->data_c());
  {
    GilReleaseWithCheck gil_release;
    runtime::Pipeline::Get().frontend_stage()->Wait();
    runtime::Pipeline::Get().backend_stage()->Wait();
    auto ret = memcpy_s(state_data, sizeof(param_type), seed_data_, sizeof(param_type));
    if (ret != EOK) {
      MS_EXCEPTION(ValueError) << "Failed to copy seed data to state, memcpy_s error. Error no: " << ret;
    }
    ret = memcpy_s(state_data + sizeof(param_type), sizeof(param_type), offset_data_, sizeof(param_type));
    if (ret != EOK) {
      MS_EXCEPTION(ValueError) << "Failed to copy offset data to state, memcpy_s error. Error no: " << ret;
    }
  }
  return py::make_tuple(py::none(), py::none(), tensor::PackTensorToPyObject(state));
}

py::object GeneratorImpl::seed() const {
  auto seed = static_cast<param_type>(std::random_device()());
  *seed_data_ = seed;
  *offset_data_ = 0;
  return py::make_tuple(tensor::PackTensorToPyObject(seed_));
}

py::object GeneratorImpl::step(const py::handle &step_py) {
  auto step_tensor = tensor::ConvertToTensor(step_py);
  TensorCheck(step_tensor, ParamTypeId, 1, "Step");
  auto step_val = *static_cast<param_type *>(step_tensor->cpu()->data_c());
  auto old_offset = std::make_shared<tensor::Tensor>(ParamTypeId, ShapeVector({1}))->cpu();
  auto old_offset_data = static_cast<param_type *>(old_offset->data_c());
  *old_offset_data = *offset_data_;
  *offset_data_ = *offset_data_ + step_val;
  return py::make_tuple(tensor::PackTensorToPyObject(seed_), tensor::PackTensorToPyObject(old_offset), py::none());
}

py::object GeneratorImpl::manual_seed(const py::handle &seed) {
  auto seed_tensor = tensor::ConvertToTensor(seed);
  TensorCheck(seed_tensor, ParamTypeId, 1, "Seed");
  auto seed_val = *static_cast<param_type *>(seed_tensor->cpu()->data_c());
  {
    GilReleaseWithCheck gil_release;
    runtime::Pipeline::Get().frontend_stage()->Wait();
    runtime::Pipeline::Get().backend_stage()->Wait();
    *seed_data_ = seed_val;
    *offset_data_ = 0;
  }
  return py::none();
}

py::object GeneratorImpl::initial_seed() const { return py::make_tuple(tensor::PackTensorToPyObject(seed_)); }

py::object GeneratorImpl::operator()(const py::handle &cmd, const py::tuple &inputs) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto cmd_val = py::cast<cmd_type>(cmd);
  MS_ASSERT_TRUE(cmd_val > _START && cmd_val < _END) << "Unknown Generator cmd: " << cmd_val;

  switch (cmd_val) {
    case STEP: {
      MS_ASSERT_TRUE(inputs.size() == kInputNum3) << "STEP expects 3 inputs: seed, offset, step.";
      return step(inputs[kIndex2]);
    }
    case SEED: {
      MS_ASSERT_TRUE(inputs.size() == kInputNum2) << "SEED expects 2 inputs: seed, offset.";
      return seed();
    }
    case GET_STATE: {
      MS_ASSERT_TRUE(inputs.size() == kInputNum2) << "GET_STATE expects 2 inputs: seed, offset.";
      return get_state();
    }
    case SET_STATE: {
      MS_ASSERT_TRUE(inputs.size() == kInputNum3) << "SET_STATE expects 3 inputs: seed, offset, state.";
      return set_state(inputs[kIndex2]);
    }
    case MANUAL_SEED: {
      MS_ASSERT_TRUE(inputs.size() == kInputNum3) << "MANUAL_SEED expects 3 inputs: seed, offset, new_seed.";
      return manual_seed(inputs[kIndex2]);
    }
    case INITIAL_SEED: {
      MS_ASSERT_TRUE(inputs.size() == kInputNum2) << "INITIAL_SEED expects 2 inputs: seed, offset.";
      return initial_seed();
    }
    default:
      MS_EXCEPTION(ValueError) << "Unknown Generator cmd: " << cmd_val;
  }
}

void RegGeneratorImpl(const py::module *m) {
  (void)py::class_<GeneratorImpl, std::shared_ptr<GeneratorImpl>>(*m, "GeneratorImpl")
    .def(py::init<const py::handle &, const py::handle &>(), "Constructor")
    .def("__call__", &GeneratorImpl::operator(), "Call with cmd and inputs");
}
}  // namespace mindspore
