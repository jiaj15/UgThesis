# UgThesis

## 中期报告 Due date : 23th, April
### To do list

#### 0401：

测试了conflict graph, weight 用 traffic volume(veh/step) 更新 ， 效果疑似不好


需要：

1. 找一个benchmark

    不是很想用fixed-time(---), vehicle-actuated 可以
    
    后续要用可变流量，fixed time比较没有意义
    
2. average waiting time 替代 volume

    实验结果没有太大差别
    hybrid
    volume: loop
    
3. penalty

    还没弄, 先找一个比较基准
    没有PENALTY的情况下，T=25（当前流量
    pWE = 1. / 9
    pEW = 1. / 10
    pNS = 1. / 9
    pSN = 1. / 10
    pWN = 1. / 30
    pNE = 1. / 30
    pES = 1. / 30
    pSW = 1. / 30）比较好，但是加了PENALTY不一定

4. T的取值和流量有关
    
    考虑一下用可变流量（例如二次/分段函数/随机分布）
    
    现在的流量是固定的，所以当T=20/25和vehilce actuated 相近，结果相似
    
    当流量发生改变时候，如果方差较大，T的最佳取值也会发生改变，但基本上最优情况= benchmark
5. 

#### 0404 

1. 把volume的计算方式改成detector,重测

    发现冲突图的结构建模做错了TAT！！！
 
    推翻重来，结果非常非常差劲

2. all red and yellow time 

    没啥影响，事实证明是因为切换过频繁或者建模错了

3. T很小的时候测试，加入penalty似乎感觉上会好一点

    加入简单的线性惩罚函数后，T的取值相对影响好了点，但是仍然是20-25的效果比较好

4. report: introduction + process + graph + numerical experi

5. 改动流量,然后测试不同的WEIGHT METHOD

#### 0410

建了个简单的CNN

    
### 数据记录

0403-作废：

####固定流量
（  pWE = 1. / 9
    pEW = 1. / 10
    pNS = 1. / 9
    pSN = 1. / 10
    pWN = 1. / 30
    pNE = 1. / 30
    pES = 1. / 30
    pSW = 1. / 30）：
    
| cycle(T) | 5    | 10  | 15  |20   |25     |30     |35     |40     | vechicle actuate|
| :---     | :--- | :---| :---|:--- |:---   |:---   |:---   |:---   |:---           |
| sumsteps | 4537 | 4230| 3561|3351 |3358   |3376   |3376   |3396   |3384           |
| meanwait | 26.38| 4.28| 1.87|1.87 |1.87   |1.87   |1.87   |1.87   |1.87           |
    



 #### 半变化流量 
 
 流量= sigma* np.random.randn + average
 
 sigma=0.1 
 
     
| cycle(T) |20   |25     |30     | vechicle actuate|
| :---     |:--- |:---   |:---   |:---           |
| sumsteps |4335 |4130   |3684   |3504           |
| meanwait |4.47 |19.39  |3.12   |3.12           |
| meanwait |2.97 |2.97   |2.97   |2.97           |
  
 
 
 
 ### state estimation
 #### 网络结构： 两层卷积+两层全链接
 
 
 
 网络输入 v1： 考虑PENETRATION RATE的速度和位置矩阵
 
 网络输出 v1： 全局的速度和位置矩阵
 
 初步训练用的数据集比较小，大概50-300个，SGD optim LR=0.2 loss大概会从0.15减小到0.05
 
 几个问题：0.05的loss是否足够，输出对于图结构的更新是否有用
 
 &darr;
 
 网络输入 v2： 考虑PENETRATION RATE的速度和位置矩阵 
 
 网络输出 v2： 12个方向的vehicle position(12,1 or 25)和WAITING TIME(12,1 or 25): (12,25,2)
 
 训练
 
 训练集： 0.7， 流量固定， 800 training set
 
    采集的方法：
    for penetration rate in 0.7+0.1rand
        generate new route file
        sumo
        collect 50 screenshots randomly
    output: 10*50 training sets
 测试集：200
 loss 0.03
 
 
 
### sumo learning notes
```

netconvert --node-files my_nod.nod.xml --edge-files my_edge.edg.xml -t my_type.type.xml -o my_net.net.xml

netconvert --osm-files ny.osm -o test.net.xml

python randomTrips.py -n test.net.xml -r test.rou.xml -e 50 -l

sumo -c myConfig.sumocfg --fcd-output.signals sumoTrace.xml

traceExporter.py --penetration 0.1 --fcd-input sumotrace.xml --ns2-mobilityoutput ns2mobility.tcl

```

/usr/share/sumo/tools/xml

`$ python xml2csv.py /home/jing/PycharmProjects/sumo_demo/UgThesis/SUMO/OUTPUT/v5.xml`