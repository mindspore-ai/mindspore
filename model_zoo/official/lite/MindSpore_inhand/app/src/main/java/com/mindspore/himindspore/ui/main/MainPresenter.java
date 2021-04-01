/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.mindspore.himindspore.ui.main;

import android.content.res.TypedArray;
import android.util.Log;

import com.mindspore.common.utils.Utils;
import com.mindspore.himindspore.R;
import com.mindspore.himindspore.base.BasePresenter;
import com.mindspore.himindspore.bean.TabEntity;
import com.mindspore.himindspore.net.FileDownLoadObserver;
import com.mindspore.himindspore.net.RetrofitHelper;
import com.mindspore.himindspore.net.UpdateInfoBean;
import com.mindspore.himindspore.ui.view.MSTabEntity;

import java.io.File;
import java.util.ArrayList;

import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;


public class MainPresenter extends BasePresenter<MainActivity> implements MainContract.Presenter {

    private static final String TAG = "MainPresenter";

    private MainContract.View mView;
    private RetrofitHelper retrofitHelper;

    public MainPresenter(MainContract.View androidView) {
        this.mView = androidView;
        retrofitHelper = new RetrofitHelper();
    }


    @Override
    public ArrayList<MSTabEntity> getTabEntity() {
        ArrayList<MSTabEntity> mTabEntities = new ArrayList<>();
        TypedArray mIconUnSelectIds = Utils.getApp().getResources().obtainTypedArray(R.array.main_tab_un_select);
        TypedArray mIconSelectIds = Utils.getApp().getResources().obtainTypedArray(R.array.main_tab_select);
        String[] mainTitles = Utils.getApp().getResources().getStringArray(R.array.main_tab_title);
        for (int i = 0; i < mainTitles.length; i++) {
            int unSelectId = mIconUnSelectIds.getResourceId(i, R.drawable.experience_uncheck);
            int selectId = mIconSelectIds.getResourceId(i, R.drawable.experience_checked);
            mTabEntities.add(new TabEntity(mainTitles[i], selectId, unSelectId));
        }
        mIconUnSelectIds.recycle();
        mIconSelectIds.recycle();
        return mTabEntities;
    }

    @Override
    public void getUpdateInfo() {
        retrofitHelper.getUpdateInfo().enqueue(new Callback<UpdateInfoBean>() {
            @Override
            public void onResponse(Call<UpdateInfoBean> call, Response<UpdateInfoBean> response) {
                Log.i(TAG, "onResponse" + response.toString());
                mView.showUpdateResult(response.body());
            }

            @Override
            public void onFailure(Call<UpdateInfoBean> call, Throwable t) {
                Log.e(TAG, "onFailure>>>" + t.toString());
                mView.showFail(call.toString());
            }
        });
    }

    @Override
    public void downloadApk(final String destDir, final String fileName, final FileDownLoadObserver<File> fileDownLoadObserver) {
        retrofitHelper.downlaodApk()
                .subscribeOn(Schedulers.io())
                .observeOn(Schedulers.io())
                .observeOn(Schedulers.computation())
                .map(responseBody -> fileDownLoadObserver.saveFile(responseBody, destDir, fileName))
                .observeOn(AndroidSchedulers.mainThread())
                .subscribe(fileDownLoadObserver);
    }
}
