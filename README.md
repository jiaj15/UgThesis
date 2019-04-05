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

1.  把volume的计算方式改成detector,重测

2. all red and yellow time 

3. T很小的时候测试，加入penalty似乎感觉上会好一点

4. report: introduction + process + graph + numerical experi

5. 改动流量,然后测试不同的WEIGHT METHOD
    
### 数据记录

0403：

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
  
    
### sumo learning notes
```

netconvert --node-files my_nod.nod.xml --edge-files my_edge.edg.xml -t my_type.type.xml -o my_net.net.xml

netconvert --osm-files ny.osm -o test.net.xml

python randomTrips.py -n test.net.xml -r test.rou.xml -e 50 -l

sumo -c myConfig.sumocfg --fcd-output.signals sumoTrace.xml

traceExporter.py --penetration 0.1 --fcd-input sumotrace.xml --ns2-mobilityoutput ns2mobility.tcl

```