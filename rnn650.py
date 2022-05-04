# -*- coding: utf-8 -*-
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

import random

pathp = '/Users/Malakhi/Downloads/VehicleMotionPrediction-main/VehicleMotionPrediction-main/data/final_data.pkl'

pathff = '/Users/Malakhi/Downloads/VehicleMotionPrediction-main/VehicleMotionPrediction-main/data/final_datap4.pkl'

pathcsv = '/Users/Malakhi/Downloads/VehicleMotionPrediction-main/VehicleMotionPrediction-main/data/final_dataCSV.csv'

#data = pd.read_pickle(pathp)

print(pd.__version__)
data = pickle.load(open(pathp, "rb"))

#with open(pathp, "rb") as fh:
#  data = pickle.load(fh)

"""with open(pathp, "rb") as fh:
    pickle.dump(d, pathff, protocol=pickle.HIGHEST_PROTOCOL)"""
    
pickle.dump(data, open(pathff, "wb"))

df = pd.DataFrame(data)
#df.to_csv(pathcsv)
  
#data.to_pickle(pathff)
print(pickle.format_version)

print("done")

init = 0
ilen = 100#len(data['input'])-1
x = data['input'][init][:,0]
for i in range(init+1,init+1+ilen):
    x = np.append(x,data['input'][i][:,0])
x = x.flatten()
y = data['input'][init][:,1]
for i in range(init+1,init+1+ilen):
    y = np.append(y,data['input'][i][:,1])
y = y.flatten()
lam = data['input'][init][:,2]
for i in range(init+1,init+1+ilen):
    lam = np.append(lam,data['input'][i][:,2])
lam = lam.flatten()
print(data['input'][init])

print("")



init = 0
ilen = 100#len(data['target'])-1
x2 = data['target'][init][:,0]
for i in range(init+1,init+1+ilen):
    x2 = np.append(x2,data['target'][i][:,0])
x2 = x2.flatten()
y2 = data['target'][init][:,1]
for i in range(init+1,init+1+ilen):
    y2 = np.append(y2,data['target'][i][:,1])
y2 = y2.flatten()
lam2 = data['target'][init][:,2]
for i in range(init+1,init+1+ilen):
    lam2 = np.append(lam2,data['target'][i][:,2])
lam2 = lam2.flatten()
print(data['target'][init])

print("x", x)
print("y", y)




#get the length of the data
number_of_characters = len(data["input"])

def get_seq(i,lim):
  start = i# - 25#random.randint(0,number_of_characters - 25)
  if start < 0:
    start = 0

  seq = []

  #togo = abs(start - i)
  #for x in range(0,togo):
  #  seq.append(data[start+x])
  seq = data["input"][i]

  return seq

print('Number of characters in text file :', number_of_characters)
#print(len(seq))

all_inputs = range(len(data["input"]))
all_inputs_list = list(all_inputs)

all_states = data["input"][0]
for i in range(1,len(data["input"])):
    all_states = np.vstack((all_states,data["input"][i]))
    
all_targets = all_states
all_targets[1:] = all_states[:-1]
all_targets[-1] = all_states[0]

print(all_targets)

plt.scatter(all_targets[0:8,0], all_targets[0:8,1], s = 1, color="blue")
plt.scatter(all_states[0:8,0], all_states[0:8,1], s = 0.1, color="red")
plt.ylabel('some numbers')
plt.xlim([-100, 50])
plt.ylim([-50, 100])
plt.show()

all_chars = range(len(all_states))#set(data)
print("okay", all_states)

m = len(all_chars)
n_letters = m
all_letters = list(all_chars)

print(data[0:100])
print(m)
print(all_chars)

print(len(set("okay lets go")))
print(set("okay lets go"))

print(data['input'][0])

print(data['input'][1])

print(data['input'][0][5:])

outti = data['input'][0][5:]
outti = np.vstack((outti,data['input'][0][:5]))

print("outti ", outti)

all_chars_list = list(all_chars)

split = math.floor(len(all_chars_list) * .8)
plt.scatter(x,y, s = 0.1, color="red")
plt.scatter(x2,y2, s = 0.1, color="blue")
plt.ylabel('some numbers')
plt.xlim([-100, 50])
plt.ylim([-50, 100])
plt.show()

import torch

argscuda = False#True
if torch.cuda.is_available():
    if not argscuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if argscuda else "cpu")


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    #print("ll", letter)
    out = next((i for i, x in enumerate(all_chars_list) if x == letter), 0)
    #print(out)
    return out#all_chars.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print("ltot", letterToTensor(1))

print("lineto", lineToTensor([1,2,3,5,8]).size())


import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        #print(self.hidden_size)
        return torch.zeros(1, self.hidden_size)
n_hidden = 1
rnn = RNN(n_letters, n_hidden, n_letters)

input = letterToTensor(1)
hidden = torch.zeros(1, n_hidden)#, device=torch.device('cuda'))

output, next_hidden = rnn(input, hidden)

input = lineToTensor([3,2,1])
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output)

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_chars_list[category_i], category_i

print(categoryFromOutput(output))

mylist = [6,9,3,6,40,25]
def lookallindicesof(char, list):
    indexlist = []
    for item in enumerate(list):
        if char in item:
            indexlist.append(item[0])
    return indexlist

indexlist = lookallindicesof(6,mylist)
print(indexlist)

import random

def randomChoice(l):
    return l[random.randint(0, len(l)-1)]

def randomTrainingExample():
    #print("cat")
    #split = math.floor(len(all_chars_list) * .8)
    category = randomChoice(all_inputs_list)#[:split])
    ind_c = -1
    """
    chooselist = lookallindicesof(category,data)
    for ele in chooselist:
      if ele < 25:
        chooselist.remove(ele)
    """
    chooselist = [category]
    #print("ind")
    ind_c = randomChoice(chooselist)
    line = get_seq(ind_c,number_of_characters)
    category_tensor = torch.tensor([all_chars_list.index(category)], dtype=torch.long)
    #print("li ", line)
    line = np.arange(ind_c-10,ind_c,1)#range(ind_c,ind_c+8)
    line_tensor = lineToTensor(line)#letterToTensor(ind_c) #lineToTensor(line)
    #print(ind_c, line_tensor)
    #x = 1/0
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)

print("DONE")

import torch.optim as optim

criterion = nn.CrossEntropyLoss()

learning_rate = 0.01 # If you set this too high, it might explode. If too low, it might not learn

optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        #print(line_tensor[i])
        output, hidden = rnn(line_tensor[i], hidden)
    #output, hidden = rnn(line_tensor, hidden)


    loss = criterion(output, category_tensor)
    loss.backward()
    
    optimizer.step()

    #rnn.step()
    # Add parameters' gradients to their values, multiplied by learning rate
    #for p in rnn.parameters():
    #    p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

import time
import math

n_iters = 10000
print_every = 500
plot_every = 100



# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        print(guess, category)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
        #plt.scatter(line[:,0], line[:,1], s = 1, color="pink")
        #plt.scatter(data["target"][guess][:,0], data["target"][guess][:,1], s = 1, color="yellow")
        plt.scatter(all_states[line[0]:line[-1],0], all_states[line[0]:line[-1],1], s = 5, color="black")
        plt.scatter(all_targets[guess:guess+80,0], all_targets[guess:guess+80,1], s = 1, color="red")
        plt.xlim([-100, 50])
        plt.ylim([-50, 100])
        plt.show()

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)
plt.show()

def evaluate(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    #for i in range(line_tensor.size()[0]):
    #    print(line_tensor[i])
    #    output, hidden = rnn(line_tensor[i], hidden)
    #output, hidden = rnn(line_tensor, hidden)
    #rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        #print(line_tensor[i])
        output, hidden = rnn(line_tensor[i], hidden)

    #rnn.step()
    # Add parameters' gradients to their values, multiplied by learning rate
    #for p in rnn.parameters():
    #    p.data.add_(p.grad.data, alpha=-learning_rate)

    return output

def randomTestExample():
    #print("cat")
    #split = math.floor(len(all_chars_list) * .8)
    category = randomChoice(all_inputs_list)#[split:])
    #if category == all_chars_list[-1]:
    #    category = all_chars_list[-2]
    ind_c = -1
    """
    chooselist = lookallindicesof(category,data)
    for ele in chooselist:
      if ele < 25:
        chooselist.remove(ele)
    """
    chooselist = [category]
    #print("ind")
    ind_c = randomChoice(chooselist)
    outti = data['input'][ind_c][5:]
    outti = np.vstack((outti,data['input'][ind_c+1][:5]))
    outti = data["target"][ind_c][25:33]
    line = outti#get_seq(ind_c,number_of_characters)
    category_tensor = torch.tensor([all_chars_list.index(category)], dtype=torch.long)
    #print(line)
    line = np.arange(ind_c-10,ind_c,1)#line = np.arange(ind_c+25,ind_c+25+8,1)##range(ind_c+25,ind_c+33)
    line_tensor = lineToTensor(line)#letterToTensor(ind_c) #lineToTensor(line)
    #print(ind_c, line_tensor)
    #x = 1/0
    return category, line, category_tensor, line_tensor

for i in range(25):
    guess_list = []
    category, line, category_tensor, line_tensor = randomTestExample()
    line_origin = line
    for i in range(80):
        output = evaluate(category_tensor, line_tensor)
        guess, guess_i = categoryFromOutput(output)
        guess_list.append(guess)
        line = list(line)
        line.append(guess)
        line = line[1:]
        line = np.array(line)
        line_tensor = lineToTensor(line)
        print(line)
    print(guess, category)
    #plt.scatter(data["target"][category][:,0], data["target"][category][:,1], s = 1, color="black")
    #plt.scatter(data["target"][guess][:,0], data["target"][guess][:,1], s = 1, color="blue")
    #plt.scatter(data["input"][category][:,0], data["input"][category][:,1], s = 1.5, color="yellow")
    #plt.scatter(line[:,0], line[:,1], s = 0.1, color="red")
    print(line)
    print(guess_list)
    line_out = all_states[line_origin[0]]
    targ_out = all_targets[guess_list[0]]
    for i in range(1,len(line)):
        line_out = np.vstack((line_out,all_states[line_origin[i]]))
    for i in range(1,len(guess_list)):
        targ_out = np.vstack((targ_out,all_targets[guess_list[i]]))
    #print(all_states[line[0]:line[-1],0])
    print(line_out)
    print(targ_out)
    plt.scatter(line_out[:,0], line_out[:,1], s = 5, color="pink")
    plt.scatter(targ_out[:,0], targ_out[:,1], s = 1, color="yellow")
    plt.xlim([-100, 50])
    plt.ylim([-50, 100])
    plt.show()
