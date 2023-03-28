/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_MODEL_LOADER_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_MODEL_LOADER_H_

#include <memory>

#include "extendrt/mindir_loader/abstract_base_model.h"

namespace mindspore::infer {
class ModelLoader {
 public:
  virtual AbstractBaseModel *ImportModel(const char *model_buf, size_t size, bool take_buf) = 0;

 protected:
  virtual int InitModelBuffer(AbstractBaseModel *model, const char *model_buf, size_t size, bool take_buf);
};

class ModelLoaderRegistry {
 public:
  ModelLoaderRegistry();
  virtual ~ModelLoaderRegistry();

  static ModelLoaderRegistry *GetInstance();

  void RegModelLoader(mindspore::ModelType model_type, std::function<std::shared_ptr<ModelLoader>()> creator) {
    model_loader_map_[model_type] = creator;
  }

  std::shared_ptr<ModelLoader> GetModelLoader(mindspore::ModelType model_type) {
    auto it = model_loader_map_.find(model_type);
    if (it == model_loader_map_.end()) {
      return nullptr;
    }
    return it->second();
  }

 private:
  mindspore::HashMap<mindspore::ModelType, std::function<std::shared_ptr<ModelLoader>()>> model_loader_map_;
};

class ModelLoaderRegistrar {
 public:
  ModelLoaderRegistrar(const mindspore::ModelType &model_type, std::function<std::shared_ptr<ModelLoader>()> creator) {
    ModelLoaderRegistry::GetInstance()->RegModelLoader(model_type, creator);
  }
  ~ModelLoaderRegistrar() = default;
};

#define REG_MODEL_LOADER(model_type, model_loader_creator) \
  static ModelLoaderRegistrar g_##model_type##model_loader##ModelLoader(model_type, model_loader_creator);
}  // namespace mindspore::infer

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_MODEL_LOADER_H_
