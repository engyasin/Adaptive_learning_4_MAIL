
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d


def get_vec_mean(arr,size=2):
    return np.mean(arr.reshape(-1,arr.shape[0]//size),axis=0).flatten().repeat(size,axis=0)

def get_errors(line,step=-1):
    out_ = [step]
    #step,R,STD,ADE,FDE = -1,-1,-1,-1,-1
    new_line = line[line.find('R='):].replace('=',' ').replace(',',' ').split(' ')[1::3]

    out_.extend([float(x) for x in new_line])
    return out_

def rev_accum(arr):
    reversed_mean = arr[::-1].cumsum()/(np.arange(arr.shape[0]+1)[1:])
    return arr#reversed_mean#arr

def read_data(data_size=16,test_mode=0):
    step_f = 5
    if data_size in [8,64,256]:
        step_f = 16

    results = {'Random':[],'GAIL':[],'BC':[], 'Expert':[-14,2.3,0.0,0.0]}

    with open(f'warm_single_gail_ds_{data_size}_agents_3_test_{test_mode}.txt') as f:

        data = f.readlines()
        
        # expert is -14 (average)
        if 'GAIL' in data[1]:
            data = data[1:]
        else:
            #random and bc
            results['Random'] = get_errors(data[1])
            for j in range(5):
                results['BC'].append(get_errors(data[j+2],step=j+1))
            data = data[7:]

        for i in range(len(data)):
            results['GAIL'].append(get_errors(data[i],step=i*step_f))

    return results


def plot_results(results, data_size=16):

    # results has 3 datasets [full, single, group]
    # Note: BC, Random, Expert in a table

    full = np.array(results[0]['GAIL'])
    single = np.array(results[1]['GAIL'])
    group = np.array(results[2]['GAIL'])



    #if you want regular ADE/FDE/R just pass the original in rev_accum
    env_type = 'adv'
    scale_std  = 5

    fig = plt.figure(figsize=(15,3))
    plt.subplot(1,3,1)
    #plt.title('ADE')
    N = int(len(full[:,0])*0.45)#30%
    
    ade_full = np.mean(full[-N:,3])
    plt.plot(full[:,0],rev_accum(full[:,3]),'-',color='green',label=f'Full-ADE ')
    #plt.fill_between(full[:,0],full[:,3]-full[:,2]/scale_std,full[:,3]+full[:,2]/scale_std,color='gray',alpha=0.2)
    ade_single = np.mean(single[-N:,3])
    plt.plot(single[:,0],rev_accum(single[:,3]),'-',color='blue',label=f'Single-ADE ')
    #plt.fill_between(single[:,0],single[:,3]-single[:,2]/scale_std,single[:,3]+single[:,2]/scale_std,color='blue',alpha=0.2)
    ade_group = np.mean(group[-N:,3])
    plt.plot(group[:,0] ,rev_accum(group[:,3]),'-',color='red',label=f'Group-ADE ')
    #plt.fill_between(group[:,0],group[:,3]-group[:,2]/scale_std,group[:,3]+group[:,2]/scale_std,color='red',alpha=0.2)

    #plt.plot(group[:,0] ,np.ones_like(group[:,0])*results[0]['BC'][-1][3],'-',color='black',label=f'BC-ADE')


    plt.xticks([1, 400, 800])
    plt.legend()
    plt.xlabel('Number of evaluation steps counted in the average')
    plt.ylabel('Mean ADE')
    #plt.xlim([0,880])
    plt.grid()

    plt.subplot(1,3,2)


    fde_full = np.mean(full[-N:,4])
    plt.plot(full[:,0],rev_accum(full[:,4]),'-',color='green',label=f'Full-FDE  ')
    #plt.fill_between(full[:,0],full[:,3]-full[:,2]/scale_std,full[:,3]+full[:,2]/scale_std,color='gray',alpha=0.2)
    fde_single = np.mean(single[-N:,4])
    plt.plot(single[:,0],rev_accum(single[:,4]),'-',color='blue',label=f'Single-FDE  ')
    #plt.fill_between(single[:,0],single[:,3]-single[:,2]/scale_std,single[:,3]+single[:,2]/scale_std,color='blue',alpha=0.2)
    fde_group = np.mean(group[-N:,4])
    plt.plot(group[:,0] ,rev_accum(group[:,4]),'-',color='red',label=f'Group-FDE  ')
    #plt.fill_between(group[:,0],group[:,3]-group[:,2]/scale_std,group[:,3]+group[:,2]/scale_std,color='red',alpha=0.2)
    #plt.plot(group[:,0] ,np.ones_like(group[:,0])*results[0]['BC'][-1][4],'-',color='black',label=f'BC-FDE')

    plt.xticks([0, 400, 800])
    plt.legend()
    plt.xlabel('Number of evaluation steps counted in the average')
    plt.ylabel('Mean FDE')
    #plt.xlim([0,880])
    plt.grid()


    plt.subplot(1,3,3)

    r_full = np.mean(full[-N:,1])
    plt.plot(full[:,0],rev_accum(full[:,1]),'-',color='green',label=f'Full-R  ')
    #plt.fill_between(full[:,0],full[:,1]-full[:,2]/scale_std,full[:,1]+full[:,2]/scale_std,color='green',alpha=0.2)
    r_single = np.mean(single[-N:,1])
    plt.plot(single[:,0],rev_accum(single[:,1]),'-',color='blue',label=f'Single-R ')
    #plt.fill_between(single[:,0],single[:,1]-single[:,2]/scale_std,single[:,1]+single[:,2]/scale_std,color='blue',alpha=0.2)
    r_group = np.mean(group[-N:,1])
    plt.plot(group[:,0] ,rev_accum(group[:,1]),'-',color='red',label=f'Group-R ')
    #plt.fill_between(group[:,0],group[:,1]-group[:,2]/scale_std,group[:,1]+group[:,2]/scale_std,color='red',alpha=0.2)
    #plt.plot(group[:,0] ,np.ones_like(group[:,0])*results[0]['BC'][-1][1],'-',color='black',label=f'BC-R')
    plt.xticks([0, 400, 800])
    plt.legend()
    plt.xlabel('Number of evaluation steps counted in the average')
    plt.ylabel('Mean Reward')
    #plt.xlim([0,880])
    plt.grid()

    fig.tight_layout(pad=0.7)
    
    plt.savefig(f'erros_plot_r_{data_size}_{env_type}.pdf')
    plt.show()

    return np.array([[ade_full,ade_single,ade_group],[fde_full,fde_single,fde_group]])
    

def plot_results__(results, data_size=16):

    # results has 3 datasets [full, single, group]
    # Note: BC, Random, Expert in a table

    full = np.array(results[0]['GAIL'])
    single = np.array(results[1]['GAIL'])
    group = np.array(results[2]['GAIL'])

    #if you want regular ADE/FDE/R just pass the original in rev_accum
    env_type = 'spread'
    scale_std  = 5

    fig = plt.figure(figsize=(5,4))
    plt.subplot(1,1,1)
    #plt.title('ADE')
    N = int(len(full[:,0])*0.45)#30%
    
    ade_full = np.mean(full[-N:,3])
    plt.plot(full[:,0],(full[:,3]),'-',color='green',label=f'Full-ADE ',linewidth=3)
    #plt.fill_between(full[:,0],full[:,3]-full[:,2]/scale_std,full[:,3]+full[:,2]/scale_std,color='gray',alpha=0.2)
    ade_single = np.mean(single[-N:,3])
    plt.plot(single[:,0],(single[:,3]),'-',color='blue',label=f'Single-ADE ')
    #plt.fill_between(single[:,0],single[:,3]-single[:,2]/scale_std,single[:,3]+single[:,2]/scale_std,color='blue',alpha=0.2)
    ade_group = np.mean(group[-N:,3])
    plt.plot(group[:,0] ,(group[:,3]),'-',color='red',label=f'Group-ADE ')
    #plt.fill_between(group[:,0],group[:,3]-group[:,2]/scale_std,group[:,3]+group[:,2]/scale_std,color='red',alpha=0.2)

    #plt.plot(group[:,0] ,np.ones_like(group[:,0])*results[0]['BC'][-1][3],'-',color='black',label=f'BC-ADE')


    plt.xticks([1, 400, 800])
    plt.legend()
    plt.xlabel('Number of evaluation steps counted in the average')
    plt.ylabel('Mean ADE')
    #plt.xlim([0,880])
    plt.grid()
    plt.show()


def plot_results_(results, data_size=16):

    # results has 3 datasets [full, single, group]
    # Note: BC, Random, Expert in a table

    full = np.array(results[0]['GAIL'])
    single = np.array(results[1]['GAIL'])
    group = np.array(results[2]['GAIL'])

    #if you want regular ADE/FDE/R just pass the original in rev_accum
    env_type = 'spread'
    scale_std  = 5

    smallest_size = min([len(full[:,0]),len(single[:,0]),len(group[:,0])])
    full = full[:smallest_size]
    single = single[:smallest_size]
    group = group[:smallest_size]
    print(smallest_size)
    fig = plt.figure(figsize=(8,7))
    plt.subplot(2,1,1)
    #plt.title('ADE')
    N = int(len(full[:,0])*0.3)#30%
    
    ade_full = np.mean(full[-N:,3])
    plt.plot(full[:,0],rev_accum(full[:,3]),'-',color='green',label=f'Full-ADE ',linewidth=4)
    #plt.fill_between(full[:,0],full[:,3]-full[:,2]/scale_std,full[:,3]+full[:,2]/scale_std,color='gray',alpha=0.2)
    ade_single = np.mean(single[-N:,3])
    plt.plot(single[:,0],rev_accum(single[:,3]),'-',color='blue',label=f'Single-ADE ',linewidth=2)
    #plt.fill_between(single[:,0],single[:,3]-single[:,2]/scale_std,single[:,3]+single[:,2]/scale_std,color='blue',alpha=0.2)
    ade_group = np.mean(group[-N:,3])
    plt.plot(group[:,0] ,rev_accum(group[:,3]),'-',color='red',label=f'Group-ADE ')
    #plt.fill_between(group[:,0],group[:,3]-group[:,2]/scale_std,group[:,3]+group[:,2]/scale_std,color='red',alpha=0.2)

    #plt.plot(group[:,0] ,np.ones_like(group[:,0])*results[0]['BC'][-1][3],'-',color='black',label=f'BC-ADE')


    plt.xticks([1, smallest_size*3, smallest_size*5])
    plt.legend()
    plt.xlabel('Training Steps',fontsize=14)
    plt.ylabel('Average Displacement Error')
    #plt.xlim([0,880])
    plt.grid()

    plt.subplot(2,1,2)


    fde_full = np.mean(full[-N:,4])
    plt.plot(full[:,0],rev_accum(full[:,4]),'-',color='green',label=f'Full-FDE ',linewidth=4)
    #plt.fill_between(full[:,0],full[:,3]-full[:,2]/scale_std,full[:,3]+full[:,2]/scale_std,color='gray',alpha=0.2)
    fde_single = np.mean(single[-N:,4])
    plt.plot(single[:,0],rev_accum(single[:,4]),'-',color='blue',label=f'Single-FDE  ',linewidth=2)
    #plt.fill_between(single[:,0],single[:,3]-single[:,2]/scale_std,single[:,3]+single[:,2]/scale_std,color='blue',alpha=0.2)
    fde_group = np.mean(group[-N:,4])
    plt.plot(group[:,0] ,rev_accum(group[:,4]),'-',color='red',label=f'Group-FDE ')
    #plt.fill_between(group[:,0],group[:,3]-group[:,2]/scale_std,group[:,3]+group[:,2]/scale_std,color='red',alpha=0.2)
    #plt.plot(group[:,0] ,np.ones_like(group[:,0])*results[0]['BC'][-1][4],'-',color='black',label=f'BC-FDE')

    plt.xticks([1, smallest_size*3, smallest_size*5])
    plt.legend()
    plt.xlabel('Training Steps',fontsize=14)
    plt.ylabel('Final Displacement Error')
    #plt.xlim([0,880])
    plt.grid()

    if False:
        plt.subplot(3,1,3)

        r_full = np.mean(full[-N:,1])
        plt.plot(full[:,0],rev_accum(full[:,1]),'-',color='green',label=f'Full-R {r_full} ')
        #plt.fill_between(full[:,0],full[:,1]-full[:,2]/scale_std,full[:,1]+full[:,2]/scale_std,color='green',alpha=0.2)
        r_single = np.mean(single[-N:,1])
        plt.plot(single[:,0],rev_accum(single[:,1]),'-',color='blue',label=f'Single-R {r_single}')
        #plt.fill_between(single[:,0],single[:,1]-single[:,2]/scale_std,single[:,1]+single[:,2]/scale_std,color='blue',alpha=0.2)
        r_group = np.mean(group[-N:,1])
        plt.plot(group[:,0] ,rev_accum(group[:,1]),'-',color='red',label=f'Group-R {r_group}')
        #plt.fill_between(group[:,0],group[:,1]-group[:,2]/scale_std,group[:,1]+group[:,2]/scale_std,color='red',alpha=0.2)
        #plt.plot(group[:,0] ,np.ones_like(group[:,0])*results[0]['BC'][-1][1],'-',color='black',label=f'BC-R')
        plt.xticks([0, 400, 800])
        plt.legend()
        plt.xlabel('Training steps counted in the average')
        plt.ylabel('Mean Reward')
        #plt.xlim([0,880])
        plt.grid()

    fig.tight_layout(pad=0.7)
    
    #plt.savefig(f'erros_plot_r_{data_size}_{env_type}.pdf')
    plt.show()

    return np.array([[ade_full,ade_single,ade_group],[fde_full,fde_single,fde_group]])


if __name__ == '__main__':

    
    Ade_Fde = []
    #ds = 256
    for ds in [10,80,160]:#,16,256,1024]:
        results = []
        for i in range(3):
            results.append(read_data(data_size=ds,test_mode=i))
        print(f'ds: {ds}')
        print(plot_results_(results,data_size=ds))
        #errors = plot_results(results,data_size=ds)
        #Ade_Fde.append(errors)
    #print(np.array(Ade_Fde).mean(axis=0))
    #breakpoint()
    #plot_accumelativ(results)



