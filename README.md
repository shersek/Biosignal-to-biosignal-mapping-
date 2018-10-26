# Biosignal-to-biosignal-mapping-

Code for learning a mapping between a number of source biosignals and a target biosignal. The mapping is represented using a neural network (using the UNet architecture commonly encountered in computer vision). This method can be used, for example to map biosignals acquired from a wearable device to a more informative biosignal that cannot be acquired using wearable sensors. The neural network architecture is implemented in PyTorch.

Examples from some of the earliest models I trained while we were still collecting training data (will update videos once models are finalized):
(Red: target signal, blue: estimated signal, green: source signals (accelerometer x,y,z))

[![](http://img.youtube.com/vi/ARkq2VKUGz8/0.jpg)](http://www.youtube.com/watch?v=ARkq2VKUGz8 "")

[![](http://img.youtube.com/vi/Yx6Dj-PEz4U/0.jpg)](http://www.youtube.com/watch?v=Yx6Dj-PEz4U "")

