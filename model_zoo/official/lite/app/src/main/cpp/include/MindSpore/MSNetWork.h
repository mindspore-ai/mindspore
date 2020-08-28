// * Copyright (c) Huawei Technologies Co., Ltd. 2018-2019. All rights reserved.

#ifndef MSNETWORK_H
#define MSNETWORK_H

#include <cstdio>
#include <algorithm>
#include <fstream>
#include <functional>
#include <sstream>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <utility>

#include <context.h>
#include <lite_session.h>
#include <model.h>
#include <errorcode.h>

using namespace mindspore;

struct ImgDims {
    int channel = 0;
    int width = 0;
    int height = 0;
};

/*struct SessIterm {
    std::shared_ptr<mindspore::session::LiteSession> sess = nullptr;
};*/



class MSNetWork {
public:
    MSNetWork();
    ~MSNetWork();

    void CreateSessionMS(char* modelBuffer, size_t bufferLen, mindspore::lite::Context* ctx);
    int ReleaseNets(void);
    mindspore::session::LiteSession *session;
    mindspore::lite::Model *model;

private:
    //std::map<std::string, SessIterm> sess;
};

#endif
