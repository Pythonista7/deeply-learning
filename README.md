# Deeply Learning

After i wrote [Backprop = a chain of VJPs !](https://ashwinmirskar.substack.com/p/backprop-a-chain-of-vjps) I was really eager to get started with playing around with these ideas and quickly level up my understanding of both the conceptual building blocks as well as the modern tools we use for ML ( or popularly known as AI). And this repo is my manifestation of the journey to "deeply learn" about some of the most important concepts and the tools which allow me to tinker with the concepts themself.


## Step 1: A Feed Forward Network 
The classic start, trying to move from as "first principle" as possible. Not just "Building backprop with only numpy" , that would miss the point. This was about actually understand why a network does `WX +b` , what does "learning" look like in the most nascent form , understanding the core pitfalls of when networks just don't work , putting "stochasticity" into action, implementing some tweaks which go a long way in improving a paramterized model's performance(data Normalisation Regularization) , all that in numpy is whats inside [1-FPP](https://github.com/Pythonista7/deeply-learning/blob/main/1-FFP.ipynb)

> Colab: [Notebook](https://colab.research.google.com/drive/1OGTIQ6tRQFYuApJ9SvfSJiPw9nZbWxVD?usp=sharing)

<img width="716" height="559" alt="image" src="https://github.com/user-attachments/assets/a48e1110-3230-4537-ab3b-1a781d58d612" />

## Step 2: Torch+AutoGradCNN
It would obviously be foolish to continue to use `np` for everything since the goal is to not just learn things conceptually but also become proficient with the tooling, hence torch. 
I start off from [this article](http://yann.lecun.com/exdb/publis/pdf/lecun-iscas-10.pdf) by the OG himself and we will try to use only the "[autodiff](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)" feature of torch.
This strikes the perfect balance between understand nuances of a architecture and all the bells and whistles it comes with while building on top of what we already know for sure.

The notebook kicks off with building a naive conv op to build my first CNN layer and train it. Then when i do train it , i see that the GPU never gets used and the training is ridiculously slow!
Guess what? GPU's HATE LOOPS! To solve this we look at how torch can help us with thinks like "unfold" and take our first steps into thinking about "vectorizing" ops A.K.A , thinking about GPU friendly matrix ops instead of naively coding in torch.

> Colab: [Notebook](https://colab.research.google.com/drive/15MP1EmlyL5so86WzF11I04MQKAfbs5j1?usp=sharing)

<img width="306" height="360" alt="image" src="https://github.com/user-attachments/assets/8c60d436-e817-4f16-b7e6-bfcb45e5db9b" />

## Step 3: Batch Norm'ing
Tuning myself to look beyond model and architecture and focus a bit on other controllables like data and behaviour of model activations , how grads and params change and whats the way to get the best out of them.
Takes me through a good read of the [paper and my notes of the same here](https://rlist.ashwinms.com/garden/2f46ca50-7fbe-4f96-a560-d3a2d21a4b73) and an exploration into visualizing internals of a model.

> Colab: [Notebook](https://colab.research.google.com/drive/1EjOft4Vr_7Pguq5tPkA8T7DqM33AtwSc#scrollTo=iJS5eVsQvPFM)

<img width="439" height="295" alt="image" src="https://github.com/user-attachments/assets/ac187487-d457-4ab5-b611-65fee02b06db" />


## Step 4: RNN+BPTT
This topic is something that eluded me for a while, but i think i finally get what Karapathy senpai told this in his (must read for rnn) [blog](https://karpathy.github.io/2015/05/21/rnn-effectiveness/).
<img width="1002" height="200" alt="image" src="https://github.com/user-attachments/assets/2e41d571-8565-4025-b0d8-6e2572ff4f93" /> 

I read through this amazing [article](https://arxiv.org/pdf/1912.05911) ( + [my notes here](https://rlist.ashwinms.com/garden/83167c4a-3ac8-4d84-9e90-c51c2f59a086)) on RNNs and used it as the guide
to my code.

The notebook mainly contains implementing an RNN doing BPTT using just auto-grad, understanding how "parallel" RNN training can get and building a small RNN classifier which reads "names" char by char and tries to tell us which language the name is from.
I try my auto-grad only mode, get super slow model training and then go on to truly appreciate the CPP torch provides and endup using their API. Along the way I explore `grad clipping`, `NLLLoss`, playing around the mental model of different configs RNNs can operate in i.e: 1:1 , 1:many , many:1 , many:many and slowly build the mechanics to 
understand sequential models and auto-regressive behaviour leading up to transformers.

Colab: [Notebook](https://colab.research.google.com/drive/1xONyE0W3ipjii-_YRB_IF0thgHEtaFyq?usp=sharing)

<img width="423" height="295" alt="image" src="https://github.com/user-attachments/assets/6e5a070c-8830-4fcf-81fc-4acc98f6bd82" />



