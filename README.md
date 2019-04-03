# UgThesis

## Due date : 15th, April

0401：

测试了conflict graph, weight 用 traffic volume(veh/step) 更新 ， 效果疑似不好


需要：

1. 找一个benchmark
2. average waiting time 替代 volume
3. penalty

### sumo learning notes
'
netconvert --node-files my_nod.nod.xml --edge-files my_edge.edg.xml -t my_type.type.xml -o my_net.net.xml

netconvert --osm-files ny.osm -o test.net.xml

python randomTrips.py -n test.net.xml -r test.rou.xml -e 50 -l

sumo -c myConfig.sumocfg --fcd-output.signals sumoTrace.xml

traceExporter.py --penetration 0.1 --fcd-input sumotrace.xml --ns2-mobilityoutput ns2mobility.tcl

'
