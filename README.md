# UgThesis

## Due date : 15th, April

0401：

测试了conflict graph, weight 用 traffic volume(veh/step) 更新 ， 效果疑似不好


需要：

1. 找一个benchmark

    不是很想用fixed time, vehicle-actuated + loop可以
    
    后续要用可变流量，fixed time比较没有意义
    
2. average waiting time 替代 volume

    实验结果没有太大差别
    
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
    

### sumo learning notes
```

netconvert --node-files my_nod.nod.xml --edge-files my_edge.edg.xml -t my_type.type.xml -o my_net.net.xml

netconvert --osm-files ny.osm -o test.net.xml

python randomTrips.py -n test.net.xml -r test.rou.xml -e 50 -l

sumo -c myConfig.sumocfg --fcd-output.signals sumoTrace.xml

traceExporter.py --penetration 0.1 --fcd-input sumotrace.xml --ns2-mobilityoutput ns2mobility.tcl

```