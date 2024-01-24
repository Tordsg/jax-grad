import jax
import json
import numpy as np
import random
import jax.numpy as jnp
from controller import PIDController, NeuralController
from plant import Bathtub, Cournot
import matplotlib.pyplot as plt


def runEpochs(settings, controller, params, plant):
    mseList = []
    paramList = []
    for i in range(settings['epochs']):
        print('debug',
              settings['controller'],
              settings['plant'],
                settings['target'],
                settings['timesteps'],
                settings['epochs'],
                settings['lr']
              )
        gradfunc = jax.value_and_grad(runOneEpoch, argnums=0)
        mse, grad = gradfunc(params, controller, plant, settings)
        params = jax.tree_map(lambda p, g: p - settings['lr'] * g, params, grad)
        print('Epoch: ', i, ' MSE: ', mse)
        paramList.append(params)
        mseList.append(mse)
    if(settings['controller'] == 'PID'):
        plotPID(mseList, paramList)
    else:
        plotMSE(mseList)

def runOneEpoch(params, controller, plant, settings):
    controller.reset()
    plant.reset()
    target = settings['target']
    signal = 0
    errors = 0
    for i in range(settings['timesteps']):
        output = plant.update(signal)
        error = target - output
        signal = controller.predict(error, params)
        errors += (jnp.square(target-output))
    return errors/settings['timesteps']

#Plot
def plotMSE(x):

    plt.plot(x, label='MSE')
    plt.title('Mean Squared Error (MSE) over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show() 
def plotPID(x, y):
    #both the mse and the parameters
    
    plt.plot(x, label='MSE')
    plt.plot(y, label='Parameters')
    plt.title('Mean Squared Error (MSE) and Parameters over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
    


if __name__ == "__main__":
    try:
        config = json.load(open('config.json'))['config'][3]
    except:
        print('No config file found or config file is not valid.')
        exit()
    print('config loaded')
    
    controllerData = config['controller']
    if(controllerData['type'] == 'PID'):
        controller = PIDController()
        params = jnp.array([random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)])
    elif(controllerData['type'] == 'Neural'):
        controller = NeuralController(controllerData['activation'])
        params = controller.gen_params(controllerData['hidden_layers'], controllerData['initRange'])
        
    plantData = config['plant']
    print(plantData)
    if(plantData['type'] == 'Bathtub'):
        plant = Bathtub(plantData['crossTub'], plantData['initHeight'], plantData['crossDrain'], plantData['noise'])
    elif(plantData['type'] == 'Cournot'):
        plant = Cournot(plantData['pMAX'], plantData['cM'], plantData['noise'])
    settings = {
        'controller': controllerData['type'], 
        'plant': config['plant']['type'],
        'target': config['target'],
        'timesteps': config['timesteps'],
        'epochs': config['epochs'],
        'lr': config['lr'],
               }

    runEpochs(settings, controller, params, plant)