# Voice-over script: The Imp-anne-etrable Safe

The movie (`cnn_explainer.mp4`) is 2:03 long and has no sound. Each section below covers one scene and gives the time when it starts. The words are a starting point, so say them your own way. Each section is sized for a normal speaking pace (about 2.5 words a second). If you run long, slow the movie down instead of rushing: change the seconds in `SCENES` at the bottom of `make_movie.py` and re-run it.

Everything on screen comes from the real trained network: the filters, feature maps and probabilities are what it actually computes for one of your board pictures of 3.

---

**0:00 Title**
> This is the Imp-anne-etrable Safe. It is a lockbox opened by a 3 digit code shown with your fingers.

**0:05 The camera**
> A webcam looks down onto the top of the safe. The network only ever sees what's inside the red square, which gets shrunk to 64 by 64 pixels, with three colors each: red, green and blue.

**0:14 A picture is just numbers**
> Zoom into a fingertip and every pixel is just  a number from 0 to 255. Bright skin is a big number, and the shadow between fingers is a smaller one. The network never sees a hand, only a grid of numbers.

**0:22 One neuron**
> The building block is a neuron. Multiply each input by a weight, add them up, add a bias, then apply ReLU: keep it if it's positive, and make it zero if it's negative. This network has about ninety-nine thousand weights.

**0:31 Convolution**
> A convolution is a neuron that slides. Here's a small 5 by 5 filter. At every pixel, it multiplies the patch under it by its weights and adds them up. Where the picture matches the filter, the output lights up. This way, it can find edges of fingers from my training pictures.

**0:43 Sixteen filters**
> The first layer has sixteen filters to identify different features like edges, bright skin, and the background. It redraws the image to account for these.

**0:49 ReLU**
ReLU sets negative values to zero, so its graph bends at zero instead of being a straight line. Without that bend, all the layers would simplify into one, and the network couldn't build up complex shapes.

**0:57 Max pooling**
> Max pooling shrinks the map, keeping only the biggest number in every two-by-two square. This halves the maps height an width and a finger that moves a pixel or two still gives the same answer.

**1:08 Going deeper**
> This whole process then repeats four times: convolution, batch norm, ReLU, and pooling. Each layer looks at the maps from the layer before, so it goes from edges to parts of fingers to whole fingers. The pictures get smaller, but there are more of them with 128 maps at the end.

**1:19 Global average pooling**
> Global average pooling turns each of those 128 maps into a single number: its average. Now it doesn't matter where the hand is in the square, only how much of each pattern the network found.

**1:28 The decision**
> A final layer and softmax turn those numbers into probabilities for zero to five fingers. Here it says three fingers, and it's almost completely sure.

**1:37 How it learned**
> It learned these weights by guessing on a picture and measuring how wrong it was with the loss. Backpropagation determines how much each weight caused the error, and gradient descent nudges each slightly. I did this with over five thousand pictures, sixty times each, and the loss went down.

**1:48 The safe**
On the safe, I press the button and show each digit. The LED matrix shows what the network sees, and I press to enter it. The right code gets a check mark and opens the servo; a wrong one stays locked.

**1:58 End**
> And that's the Imp-anne-etrable Safe: a CNN running on an Arduino UNO Q.

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
