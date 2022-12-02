import model
import torch
import numpy as np
import os 
from flask import Flask
from flask import request
import ast
import json
app = Flask(__name__)

net = model.MultilayerPerceptron(num_features=model.num_features,
                             num_classes=model.num_classes,
                            num_hidden_1=model.num_hidden_1,
                            num_hidden_2=model.num_hidden_2,
                            num_hidden_3=model.num_hidden_3,
                            num_hidden_4=model.num_hidden_4)
net.to(model.device)
net.load_state_dict(torch.load(os.path.join('Save_network', 'model'),map_location=model.device))
@app.route('/predict', methods=['POST'])
def predict():
    print(request.data)
    # data = torch.from_numpy(np.array([json_array],dtype=np.float32))
    resultat = ""
    # _, probas = net(test)
    
    # classLabel = torch.argmax(probas[0]).item()
    # probabilite = max(probas[0][0],probas[0][1]).item()
    # resultat = "{class: "+str(classLabel)+", proba:"+str(probabilite)+"}"
    return resultat


if __name__ == '__main__':
    app.run()