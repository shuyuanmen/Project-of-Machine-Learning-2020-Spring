import configparser
import pickle
import numpy as np
import sys
import json
import matplotlib.pyplot as plt
from utils import get_SVR_input
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error

def MAPE(true,pred):
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)*100

def test():
    speed_x, speed_y,time, speed_scaler = get_SVR_input('test','speed')
    flow_x, flow_y,time, flow_scaler = get_SVR_input('test','flow')

    # Load the prediction pipeline.
    # with open('pickle/svr_speed_model.pickle', "rb") as model_file:
    #     speed_model = pickle.load(model_file)
    #
    # with open('pickle/svr_flow_model.pickle', "rb") as model_file:
    #     flow_model = pickle.load(model_file)
    with open('pickle/svr_speed_model.pickle', "rb") as model_file:
        speed_model = pickle.load(model_file)

    with open('pickle/svr_flow_model.pickle', "rb") as model_file:
        flow_model = pickle.load(model_file)

    with open('grid_search/pickle/svr_speed_model.pickle', "rb") as model_file:
        plain_speed_model = pickle.load(model_file)

    with open('grid_search/pickle/svr_flow_model.pickle', "rb") as model_file:
        plain_flow_model = pickle.load(model_file)

    # Calculate test score(coefficient of determination).
    speed_score = speed_model.score(speed_x, speed_y)
    flow_score = flow_model.score(flow_x, flow_y)

    print("Test score(coefficient of determination) :",speed_score,flow_score)

    predictVelocity = []
    predictFlow = []
    predictVelocityPlain = []
    predictFlowPlain = []


    for i in speed_x:
        predictVelocity.append(speed_model.predict([i]).tolist()[0])
        predictVelocityPlain.append(plain_speed_model.predict([i]).tolist()[0])

    for i in flow_x:
        predictFlow.append(flow_model.predict([i]).tolist()[0])
        predictFlowPlain.append(plain_flow_model.predict([i]).tolist()[0])

    print ('PSO')
    print ('\tspeed:')
    print ('\t\tmae: ',mean_absolute_error(speed_y,predictVelocity))
    print ('\t\tr2 score: ',r2_score(speed_y,predictVelocity))
    print ('\t\tmse: ',mean_squared_error(speed_y,predictVelocity))
    print ('\t\tmape: ',MAPE(speed_y,predictVelocity))
    print ('\tflow:')
    print ('\t\tmae: ',mean_absolute_error(flow_y,predictFlow))
    print ('\t\tr2 score: ',r2_score(flow_y,predictFlow))
    print ('\t\tmse: ',mean_squared_error(flow_y,predictFlow))
    print ('\t\tmape: ',MAPE(flow_y,predictFlow))

    print ('Grid Search')
    print ('\tspeed:')
    print ('\t\tmae: ',mean_absolute_error(speed_y,predictVelocityPlain))
    print ('\t\tr2 score: ',r2_score(speed_y,predictVelocityPlain))
    print ('\t\tmse: ',mean_squared_error(speed_y,predictVelocityPlain))
    print ('\t\tmape: ',MAPE(speed_y,predictVelocity))
    print ('\tflow:')
    print ('\t\tmae: ',mean_absolute_error(flow_y,predictFlowPlain))
    print ('\t\tr2 score: ',r2_score(flow_y,predictFlowPlain))
    print ('\t\tmse: ',mean_squared_error(flow_y,predictFlowPlain))
    print ('\t\tmape: ',MAPE(flow_y,predictFlowPlain))

    x = range(len(speed_x))

    last=-1

    # plt.rcParams['font.family'] = ['Ping Hei']
    # plt.rcParams['font.size'] = 15
    # plt.rcParams['axes.unicode_minus'] = False
    # font = FontProperties(fname='/Users/YogaLucas/Library/Fonts/SF-Pro-Text-Regular.otf',size=12)

    fig=plt.figure(dpi=200)#figsize=(15,9.375))
    ax1=fig.add_subplot(2,1,1)
    ax1.plot(x[:last],speed_y[:last],color='#3AB8E8',label="True",linestyle='-',lw=1.5)
    ax1.plot(x[:last],predictVelocity[:last],color='#EE6B72',label= "Optimized",linestyle='-',lw=1.5)
    ax1.plot(x[:last],predictVelocityPlain[:last],color='#9C9184',label= "Raw",linestyle='-',lw=1.5)

    ax1.set_xlabel("time/min")#,fontproperties=font)
    ax1.set_ylabel("speed")#,fontproperties=font)
    # ax.set_ylim(-1,1)
    # ax1.set_title('PSO-SVR 预测结果')
    ax1.legend(loc="best",framealpha=0.5)

    ax3=fig.add_subplot(2,1,2)
    ax3.plot(x[:last],flow_y[:last],color='#3AB8E8',label="True",linestyle='-',lw=1.5)
    ax3.plot(x[:last],predictFlow[:last],color='#EE6B72',label= "Optimized",linestyle='-',lw=1.5)
    ax3.plot(x[:last],predictFlowPlain[:last],color='#9C9184',label= "Raw",linestyle='-',lw=1.5)

    ax3.set_xlabel("time/min")#,fontproperties=font)
    ax3.set_ylabel("volume")#,fontproperties=font)
    # ax.set_ylim(-1,1)
    # ax3.set_title('PSO-SVR 预测结果')

    ax3.legend(loc="best",framealpha=0.5)



    # fig.suptitle('\nSVR-PSO预测结果')
    plt.tight_layout()
    # plt.subplots_adjust(left=0.08,hspace =0.5)
    # plt.show()
    plt.savefig('pic/res.png')




if __name__ == "__main__":
    test()
