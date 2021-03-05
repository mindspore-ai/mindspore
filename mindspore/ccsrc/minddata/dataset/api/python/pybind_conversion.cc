/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/api/python/pybind_conversion.h"

namespace mindspore {
namespace dataset {
float toFloat(const py::handle &handle) { return py::reinterpret_borrow<py::float_>(handle); }

int toInt(const py::handle &handle) { return py::reinterpret_borrow<py::int_>(handle); }

int64_t toInt64(const py::handle &handle) { return py::reinterpret_borrow<py::int_>(handle); }

bool toBool(const py::handle &handle) { return py::reinterpret_borrow<py::bool_>(handle); }

std::string toString(const py::handle &handle) { return py::reinterpret_borrow<py::str>(handle); }

std::set<std::string> toStringSet(const py::list list) {
  std::set<std::string> set;
  if (!list.empty()) {
    for (auto l : list) {
      if (!l.is_none()) {
        (void)set.insert(py::str(l));
      }
    }
  }
  return set;
}

std::map<std::string, int32_t> toStringMap(const py::dict dict) {
  std::map<std::string, int32_t> map;
  if (!dict.empty()) {
    for (auto p : dict) {
      (void)map.emplace(toString(p.first), toInt(p.second));
    }
  }
  return map;
}

std::vector<std::string> toStringVector(const py::list list) {
  std::vector<std::string> vector;
  if (!list.empty()) {
    for (auto l : list) {
      if (l.is_none())
        vector.emplace_back("");
      else
        vector.push_back(py::str(l));
    }
  }
  return vector;
}

std::vector<pid_t> toIntVector(const py::list input_list) {
  std::vector<pid_t> vector;
  if (!input_list.empty()) {
    std::transform(input_list.begin(), input_list.end(), std::back_inserter(vector),
                   [&](const py::handle &handle) { return static_cast<pid_t>(toInt(handle)); });
  }
  return vector;
}

std::unordered_map<int32_t, std::vector<pid_t>> toIntMap(const py::dict input_dict) {
  std::unordered_map<int32_t, std::vector<pid_t>> map;
  if (!input_dict.empty()) {
    for (auto p : input_dict) {
      (void)map.emplace(toInt(p.first), toIntVector(py::reinterpret_borrow<py::list>(p.second)));
    }
  }
  return map;
}

std::pair<int64_t, int64_t> toIntPair(const py::tuple tuple) {
  std::pair<int64_t, int64_t> pair;
  if (!tuple.empty()) {
    pair = std::make_pair(toInt64((tuple)[0]), toInt64((tuple)[1]));
  }
  return pair;
}

std::vector<std::pair<int, int>> toPairVector(const py::list list) {
  std::vector<std::pair<int, int>> vector;
  if (list) {
    for (auto data : list) {
      auto l = data.cast<py::tuple>();
      if (l[1].is_none())
        vector.emplace_back(toInt64(l[0]), 0);
      else
        vector.emplace_back(toInt64(l[0]), toInt64(l[1]));
    }
  }
  return vector;
}

std::vector<std::shared_ptr<TensorOperation>> toTensorOperations(py::list operations) {
  std::vector<std::shared_ptr<TensorOperation>> vector;
  if (!operations.empty()) {
    for (auto op : operations) {
      std::shared_ptr<TensorOp> tensor_op;
      if (py::isinstance<TensorOp>(op)) {
        tensor_op = op.cast<std::shared_ptr<TensorOp>>();
        vector.push_back(std::make_shared<transforms::PreBuiltOperation>(tensor_op));
      } else if (py::isinstance<py::function>(op)) {
        tensor_op = std::make_shared<PyFuncOp>(op.cast<py::function>());
        vector.push_back(std::make_shared<transforms::PreBuiltOperation>(tensor_op));
      } else {
        if (py::isinstance<TensorOperation>(op)) {
          vector.push_back(op.cast<std::shared_ptr<TensorOperation>>());
        } else {
          THROW_IF_ERROR([]() {
            RETURN_STATUS_UNEXPECTED(
              "Error: tensor_op is not recognised (not TensorOp, TensorOperation and not pyfunc).");
          }());
        }
      }
    }
  }
  return vector;
}

std::shared_ptr<TensorOperation> toTensorOperation(py::handle operation) {
  std::shared_ptr<TensorOperation> op;
  std::shared_ptr<TensorOp> tensor_op;
  if (py::isinstance<TensorOperation>(operation)) {
    op = operation.cast<std::shared_ptr<TensorOperation>>();
  } else if (py::isinstance<TensorOp>(operation)) {
    tensor_op = operation.cast<std::shared_ptr<TensorOp>>();
    op = std::make_shared<transforms::PreBuiltOperation>(tensor_op);
  } else {
    THROW_IF_ERROR(
      []() { RETURN_STATUS_UNEXPECTED("Error: input operation is not a tensor_op or TensorOperation."); }());
  }
  return op;
}

std::vector<std::shared_ptr<DatasetNode>> toDatasetNode(std::shared_ptr<DatasetNode> self, py::list datasets) {
  std::vector<std::shared_ptr<DatasetNode>> vector;
  vector.push_back(self);
  if (datasets) {
    for (auto ds : *datasets) {
      if (py::isinstance<DatasetNode>(ds)) {
        vector.push_back(ds.cast<std::shared_ptr<DatasetNode>>());
      } else {
        THROW_IF_ERROR(
          []() { RETURN_STATUS_UNEXPECTED("Error: datasets is not recognised (not a DatasetNode instance)."); }());
      }
    }
  }
  return vector;
}

std::shared_ptr<SamplerObj> toSamplerObj(py::handle py_sampler, bool isMindDataset) {
  if (py_sampler.is_none()) {
    return nullptr;
  }
  if (py_sampler) {
    std::shared_ptr<SamplerObj> sampler_obj;
    if (!isMindDataset) {
      auto parse = py::reinterpret_borrow<py::object>(py_sampler).attr("parse");
      sampler_obj = parse().cast<std::shared_ptr<SamplerObj>>();
    } else {
      // Mindrecord Sampler
      std::shared_ptr<mindrecord::ShardOperator> sampler;
      auto parse = py::reinterpret_borrow<py::object>(py_sampler).attr("parse_for_minddataset");
      sampler = parse().cast<std::shared_ptr<mindrecord::ShardOperator>>();
      sampler_obj = std::make_shared<PreBuiltSamplerObj>(std::move(sampler));
    }
    return sampler_obj;
  } else {
    THROW_IF_ERROR([]() { RETURN_STATUS_UNEXPECTED("Error: sampler input is not SamplerRT."); }());
  }
  return nullptr;
}

// Here we take in a python object, that holds a reference to a C++ object
std::shared_ptr<DatasetCache> toDatasetCache(std::shared_ptr<CacheClient> cc) {
  if (cc) {
    std::shared_ptr<DatasetCache> built_cache;
    built_cache = std::make_shared<PreBuiltDatasetCache>(std::move(cc));
    return built_cache;
  } else {
    // don't need to check here as cache is not enabled.
    return nullptr;
  }
}

ShuffleMode toShuffleMode(const int32_t shuffle) {
  if (shuffle == 0) return ShuffleMode::kFalse;
  if (shuffle == 1) return ShuffleMode::kFiles;
  if (shuffle == 2) return ShuffleMode::kGlobal;
  return ShuffleMode();
}

std::vector<std::shared_ptr<CsvBase>> toCSVBase(py::list csv_bases) {
  std::vector<std::shared_ptr<CsvBase>> vector;
  if (csv_bases) {
    for (auto base : *csv_bases) {
      if (py::isinstance<py::int_>(base)) {
        vector.push_back(std::make_shared<CsvRecord<int>>(CsvType::INT, toInt(base)));
      } else if (py::isinstance<py::float_>(base)) {
        vector.push_back(std::make_shared<CsvRecord<float>>(CsvType::FLOAT, toFloat(base)));
      } else if (py::isinstance<py::str>(base)) {
        vector.push_back(std::make_shared<CsvRecord<std::string>>(CsvType::STRING, toString(base)));
      } else {
        THROW_IF_ERROR([]() { RETURN_STATUS_UNEXPECTED("Error: each default value must be int, float, or string"); }());
      }
    }
  }
  return vector;
}

Status ToJson(const py::handle &padded_sample, nlohmann::json *const padded_sample_json,
              std::map<std::string, std::string> *sample_bytes) {
  for (const py::handle &key : padded_sample) {
    if (py::isinstance<py::bytes>(padded_sample[key])) {
      (*sample_bytes)[py::str(key).cast<std::string>()] = padded_sample[key].cast<std::string>();
      // py::str(key) enter here will loss its key name, so we create an unuse key for it in json, to pass ValidateParam
      (*padded_sample_json)[py::str(key).cast<std::string>()] = nlohmann::json::object();
    } else {
      nlohmann::json obj_json;
      if (padded_sample[key].is_none()) {
        obj_json = nullptr;
      } else if (py::isinstance<py::int_>(padded_sample[key])) {
        obj_json = padded_sample[key].cast<int64_t>();
      } else if (py::isinstance<py::float_>(padded_sample[key])) {
        obj_json = padded_sample[key].cast<double>();
      } else if (py::isinstance<py::str>(padded_sample[key])) {
        obj_json = padded_sample[key].cast<std::string>();  // also catch py::bytes
      } else {
        MS_LOG(ERROR) << "Python object convert to json failed: " << py::cast<std::string>(padded_sample[key]);
        RETURN_STATUS_SYNTAX_ERROR("Python object convert to json failed");
      }
      (*padded_sample_json)[py::str(key).cast<std::string>()] = obj_json;
    }
  }
  return Status::OK();
}

Status toPadInfo(py::dict value, std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> *pad_info) {
  for (auto p : value) {
    if (!p.second.is_none()) {
      auto tp = py::reinterpret_borrow<py::tuple>(p.second);
      CHECK_FAIL_RETURN_UNEXPECTED(tp.size() == 2, "tuple in pad_info must be (list,int) or (list,float)");
      TensorShape shape = tp[0].is_none() ? TensorShape::CreateUnknownRankShape() : TensorShape(tp[0]);
      std::shared_ptr<Tensor> pad_val = nullptr;
      if (py::isinstance<py::str>(tp[1])) {
        std::string pad_val_string = tp[1].is_none() ? "" : toString(tp[1]);
        CHECK_FAIL_RETURN_UNEXPECTED(
          Tensor::CreateFromVector(std::vector<std::string>{pad_val_string}, TensorShape::CreateScalar(), &pad_val),
          "Cannot create pad_value Tensor");
      } else {
        float pad_val_float = tp[1].is_none() ? 0 : toFloat(tp[1]);
        CHECK_FAIL_RETURN_UNEXPECTED(
          Tensor::CreateEmpty(TensorShape::CreateScalar(), DataType(DataType::DE_FLOAT32), &pad_val),
          "Cannot create pad_value Tensor");
        pad_val->SetItemAt<float>({}, pad_val_float);
      }
      (void)pad_info->insert({toString(p.first), {shape, pad_val}});
    } else {  // tuple is None
      (void)pad_info->insert({toString(p.first), {TensorShape({}), nullptr}});
    }
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> toPyFuncOp(py::object func, DataType::Type data_type) {
  std::shared_ptr<TensorOp> py_func;
  if (!func.is_none()) {
    py::function py_function = func.cast<py::function>();
    py_func = std::make_shared<PyFuncOp>(py_function, data_type);
  } else {
    py_func = nullptr;
  }
  return py_func;
}

py::list shapesToListOfShape(std::vector<TensorShape> shapes) {
  py::list shape_list;
  for (const auto &shape : shapes) {
    shape_list.append(shape.AsVector());
  }
  return shape_list;
}

py::list typesToListOfType(std::vector<DataType> types) {
  py::list type_list;
  for (const auto &type : types) {
    type_list.append(type.AsNumpyType());
  }
  return type_list;
}
}  // namespace dataset
}  // namespace mindspore
