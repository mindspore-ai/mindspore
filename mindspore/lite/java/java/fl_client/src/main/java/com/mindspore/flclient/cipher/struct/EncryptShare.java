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

package com.mindspore.flclient.cipher.struct;

import com.mindspore.flclient.Common;

import java.util.logging.Logger;

/**
 * class used for encrypt shares of secret
 *
 * @since 2021-8-27
 */
public class EncryptShare {
    private static final Logger LOGGER = Logger.getLogger(DecryptShareSecrets.class.toString());
    private String flID;
    private NewArray<byte[]> share;

    /**
     * get client's flID
     *
     * @return flID of this client
     */
    public String getFlID() {
        if (flID == null || flID.isEmpty()) {
            LOGGER.severe(Common.addTag("[DecryptShareSecrets] the parameter of <flID> is null, please set it before " +
                    "use"));
            throw new IllegalArgumentException();
        }
        return flID;
    }

    /**
     * set client's flID
     *
     * @param flID hash value used for identify client
     */
    public void setFlID(String flID) {
        this.flID = flID;
    }

    /**
     * get secret share
     *
     * @return secret share
     */
    public NewArray<byte[]> getShare() {
        return share;
    }

    /**
     * set secret share
     *
     * @param share secret share
     */
    public void setShare(NewArray<byte[]> share) {
        this.share = share;
    }
}
