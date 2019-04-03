import matplotlib.pyplot as plt
import pandas as pd
import os


def SaveResults(results,cycle,standard):
    volume=[]
    waiting=[]
    quelength=[]
    steps=[]
    step=0


    for index in range(len(results)):
        step=index*cycle
        steps.append(step)
        volume_step=0
        waiting_step=0
        quelength_step=0

        for klane in results[index]:
            volume_step += klane.traffic_volume
            waiting_step+= klane.AverageWaitingTime * klane.traffic_volume

            quelength_step+=klane.queuelength
        if volume_step !=0:
            waiting.append(waiting_step/volume_step)
        else:
            waiting.append(0)
        volume.append(volume_step)
        quelength.append(quelength_step)

    data={'steps':steps,'volume':volume, 'queue length':quelength, 'Average waiting time':waiting}
    results_df=pd.DataFrame(data)

    filename = 'data/output/'+str(cycle)+'-'+str(standard)+'.csv'

    results_df.to_csv(filename, mode="w+")
    return results_df
def PlotResults():
    label_list=[]
    mean_dict={}

    dic={'v':'volume','w':'weight'}
    path='/home/jing/PycharmProjects/sumo_demo/UgThesis/SUMO/data/output'
    names=os.listdir(path)
    flag =True
    for csv_name in names:
        data_f=pd.read_csv(path+'/'+csv_name)
        waiting= data_f['Average waiting time']

        steps=data_f['steps']
        # waitplot=pd.DataFrame(waiting,index=data_f.index)
        # steps= data_f.index
        label= os.path.splitext(csv_name)[0]
        labels=label.split('-',1)
        label="cycle = "+labels[0]+' '+dic[labels[1]]
        label_list.append(label)
        if flag:
            data={'steps':steps,label:waiting}
            df=pd.DataFrame(data)
            flag=False
        else:
            df[label]=waiting
        mean_dict[labels[0]]=data_f['Average waiting time'].mean()


        # data_f.plot(x='steps',y='Average waiting time')
        # plt.figure()
        # data_f.plot()
        # print csv_name
    df.plot(x='steps')

    plt.legend(label_list)
    plt.show()

    print mean_dict
    lists=sorted(mean_dict.items())
    x, y = zip(*lists) # unpack a list of pairs into two tuples

    plt.plot(x, y)
    plt.show()


    # plt.plot(df.mean)


def csvPlot():
    cycle = ['5', '10', '15', '20', '25', '30', '35', '40']
    stand = ['v']
    path = '/home/jing/PycharmProjects/sumo_demo/UgThesis/SUMO/mean/csv'
    names = os.listdir(path)
    flag = True
    for csv_name in names:
        data_f = pd.read_csv(path + '/' + csv_name)
        steps = data_f['step_time']
        waiting = data_f['step_meanWaitingTime']
        label = os.path.splitext(csv_name)[0]
        labels = label.split('v', 1)
        labela = "cycle = " + labels[1]
        if labela in cycle:
            if flag:
                data = {'steps': steps, labela: waiting}
                df = pd.DataFrame(data)
                flag = False
            else:
                df[labela] = waiting
