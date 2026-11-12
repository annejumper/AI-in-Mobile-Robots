# Voice-over script: The Imp-anne-etrable Safe

The movie (`cnn_explainer.mp4`) is 2:03 long and has no sound. Each section below covers one scene and gives the time when it starts. The words are a starting point, so say them your own way. Each section is sized for a normal speaking pace (about 2.5 words a second). If you run long, slow the movie down instead of rushing: change the seconds in `SCENES` at the bottom of `make_movie.py` and re-run it.

Everything on screen comes from the real trained network: the filters, feature maps and probabilities are what it actually computes for one of your board pictures of 3.

---

**0:00 Title**
> This is the Imp-anne-etrable Safe. To open it, you show it a code with your fingers. So how does a computer count fingers?

**0:05 The camera**
> A webcam looks down into the box. The network only ever sees what's inside the red square. That square gets shrunk to 64 by 64 pixels, with three colours each: red, green and blue. That's about twelve thousand numbers, and that's all the network gets.

**0:14 A picture is just numbers**
> Zoom into a fingertip and every pixel is just a number from 0 to 255. Bright skin is a big number, and the shadow between fingers is a smaller one. The network never sees a hand, only a grid of numbers.

**0:22 One neuron**
> The building block is the same neuron I used to fit the V data. Multiply each input by a weight, add them up, add a bias, then apply ReLU: keep it if it's positive, and make it zero if it's negative. The V-fit had two parameters, m and b. This network has about ninety-nine thousand.

**0:31 Convolution**
> A convolution is a neuron that slides. Here's a small 5 by 5 filter. At every spot in the picture, it multiplies the patch under it by its weights and adds them up. Where the picture matches the filter, the output lights up. This one learned to find the edges of fingers. Nobody programmed it; it learned that from my training pictures.

**0:43 Sixteen filters**
> The first layer has sixteen filters, so it redraws the picture sixteen different ways. Some see edges, some see bright skin, and some see the background.

**0:49 ReLU**
> Then ReLU. Positive values stay, and negative values become zero. Without this step, stacking layers would collapse into one big multiply-and-add, and the network couldn't learn anything complicated.

**0:57 Max pooling**
> Max pooling shrinks the map. In every two-by-two square, keep only the biggest number. The map gets half as wide and half as tall, and a finger that moves a pixel or two still gives the same answer.

**1:08 Going deeper**
> Then the whole thing repeats: convolution, batch norm, ReLU, pooling, four times. Each layer looks at the maps from the layer before, so it goes from edges to parts of fingers to whole fingers. The pictures get smaller, but there are more of them: 128 maps at the end.

**1:19 Global average pooling**
> Global average pooling turns each of those 128 maps into a single number: its average. Now it doesn't matter where the hand is in the square, only how much of each pattern the network found. That's exactly what you want for counting.

**1:28 The decision**
> One last layer of neurons turns those 128 numbers into six scores, one each for zero to five fingers. Softmax turns the scores into probabilities that add up to one. Here it says three fingers, and it's almost completely sure.

**1:37 How it learned**
> How did it learn all those weights? The same way as the V-fit. Show it a picture, let it guess, and measure how wrong it was with the loss. Backpropagation works out how much each weight was to blame, and gradient descent nudges every weight a little bit. I did this with over five thousand pictures, sixty times each, and the loss went down.

**1:48 The safe**
> On the safe, I press the button and show each digit. The LED matrix shows what the network sees, and I press again to enter it. If all three digits match the code, I get a check mark and the servo opens. Wrong code, and it stays locked.

**1:58 End**
> And that's the Imp-anne-etrable Safe: a CNN built from scratch, running on an Arduino UNO Q.

---

## Recording and adding your voice

1. Open `cnn_explainer.mp4` and get ready to read along.
2. Record your voice in **QuickTime** (File, then New Audio Recording) or **Zoom** (record locally, which gives an `.m4a`), and start the video at the same moment. Pausing between scenes is fine; you can trim later.
3. Put the two together using either option:
   - **iMovie:** drop in the video and the audio, then slide the audio until it lines up.
   - **One command:** from the repo folder (change `voice.m4a` to your file's name):
     ```
     ffmpeg -i movie/cnn_explainer.mp4 -i voice.m4a -c:v copy -c:a aac -shortest movie/cnn_explainer_voiced.mp4
     ```
