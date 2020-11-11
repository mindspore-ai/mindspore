package com.mindspore.himindspore.net;

import io.reactivex.Observable;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.http.GET;
import retrofit2.http.Streaming;

public interface RetrofitService {

    @GET("version.json")
    Call<UpdateInfoBean> getUpdateInfo();

    @Streaming
    @GET("himindsporedemo.apk")
    Observable<ResponseBody> downloadApk();
}
