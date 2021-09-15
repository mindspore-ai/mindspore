/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "SSDResnet50Fpn.h"
#include "MxBase/Log/Log.h"

namespace {
    const uint32_t CLASS_NUM = 81;
}

int main(int argc, char *argv[]) {
    if (argc <= 2) {
        LogWarn << "Please input image path, such as './ssd_resnet50_fpn ssd_resnet50.om test.jpg'.";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.classNum = CLASS_NUM;
    initParam.labelPath = "../models/coco.names";

    initParam.iou_thresh = 0.6;
    initParam.score_thresh = 0.2;
    initParam.checkTensor = true;

    initParam.modelPath = argv[1];
    auto ssdResnet50Fpn = std::make_shared<SSDResnet50Fpn>();
    APP_ERROR ret = ssdResnet50Fpn->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "SSDResnet50Fpn init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgPath = argv[2];
    ret = ssdResnet50Fpn->Process(imgPath);
    if (ret != APP_ERR_OK) {
        LogError << "SSDResnet50Fpn process failed, ret=" << ret << ".";
        ssdResnet50Fpn->DeInit();
        return ret;
    }
    ssdResnet50Fpn->DeInit();
    return APP_ERR_OK;
}
