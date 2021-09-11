/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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

package com.mindspore.flclient;

import java.util.concurrent.TimeoutException;

/**
 * Define basic communication interface.
 *
 * @since 2021-06-30
 */
public interface IFLCommunication {
    /**
     * Sets the timeout interval for communication on the device.
     *
     * @param timeout the timeout interval for communication on the device.
     * @throws TimeoutException catch TimeoutException.
     */
    void setTimeOut(int timeout) throws TimeoutException;

    /**
     * Synchronization request function.
     *
     * @param url the URL for device-sever interaction set by user.
     * @param msg the message need to be sent to server.
     * @return the response message.
     * @throws Exception catch Exception.
     */
    byte[] syncRequest(String url, byte[] msg) throws Exception;

    /**
     * Asynchronous request function.
     *
     * @param url      the URL for device-sever interaction set by user.
     * @param msg      the message need to be sent to server.
     * @param callBack the call back object.
     * @throws Exception catch Exception.
     */
    void asyncRequest(String url, byte[] msg, IAsyncCallBack callBack) throws Exception;
}