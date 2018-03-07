mymnist_client.py等文件为mnist的客户端，连接的tensorflow serving地址信息在源码中
运行方式
/root/mnist-client为mnist-client下载到本地的目录，启动容器
docker run -it -v /root/mnist-client:/ts mycat/tensorflowclient bash
cd /ts
python mymnist_client.py
输出图片example3.bmp的识别结果