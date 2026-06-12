# Pytorch-Four-In-A-Row-Ai-Training-Learning
Just me trying to learn how to build AIs and machine learning in python using the pytorch library. I built the game by hand and then used mostly used Claude to generate a lot of the code to generate the code for the AI itself. Then, I'll study everything it uses.


I started out by trying to do claude based DQL (Deep Q-Learning). Due to several reason, it failed. I tried to fix it several times, but at most I got a very bad model that was unplayable.

Still, winning was not the point, the point was learning more about machine learning and how to implement it. I learned several very valuable lessons, like what a minmaxing algorhythm or a DQL is. I also learned how to work with tensors a lot better and that there are other non-linear forms of neural networks, like conv2d.

After I got Claude to generate me a minmaxing algorhythm, I tested it and it worked. But, unfortunately, it wasn't really machine learning, therefore, not what I was looking for.

Later, I got the idea to make the neural network learn based on the output of the algorythm, later I was informed it is called "imitation learning". How it is going, we'll see later. Or at least, I will.

The original attempt with a minmaxing with length of 5 failed. It took over a second and a half per game, and would take several hours to complete, something I wasn't willing to spend. So I lowered the training depth to 3. We'll see how it goes.

