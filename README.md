# MIPT lectures processing

The processing is done in order to crop high-resolution video to the region where the speaker is currently located. As the speaker can move, we introduce the delay option - this option controls the frequency of different crops switchings. The algorithm is based on YOLO network predictions, which is necessary for speaker location.
