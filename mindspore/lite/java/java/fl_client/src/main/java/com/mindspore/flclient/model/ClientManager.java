/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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

package com.mindspore.flclient.model;

import com.mindspore.flclient.Common;

import java.util.HashMap;
import java.util.logging.Logger;

public class ClientManager {
    private static final HashMap<String, Client> clientMaps = new HashMap<>();

    private static final Logger logger = Logger.getLogger(ClientManager.class.toString());

    /**
     * Register client.
     *
     * @param client client.
     */
    public static void registerClient(Client client) {
        if(client == null) {
            logger.severe(Common.addTag("client cannot be null"));
        }
        clientMaps.put(client.getClass().getName(), client);
    }

    /**
     * Get client object.
     *
     * @param name clent class name.
     * @return client object.
     */
    public static Client getClient(String name) {
        return clientMaps.getOrDefault(name, null);
    }
}
