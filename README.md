# faster-rcnn-resnet50

## project overview
This project develops an object detection system using the Faster R-CNN framework with a ResNet-50 backbone, specifically trained to detect "smoke" in various environments. This model is crucial for applications in safety and surveillance, particularly in scenarios where early smoke detection can prevent fire disasters and enhance response times.

### Model Configuration

Faster R-CNN with ResNet-50 Backbone: The model employs the Faster R-CNN architecture, which is renowned for its effectiveness in object detection tasks due to its integrated approach of feature extraction (via ResNet-50) and region proposal network (RPN).

Custom Classifier Head: The pre-trained Faster R-CNN model is adapted by modifying the classifier head to recognize two classes: "smoke" and "background." The adaptation involves changing the number of input features in the last classification layer to accommodate these two specific classes.

Backbone Freezing: The ResNet-50 backbone's parameters are frozen during training to leverage the rich feature representations learned from large-scale image datasets, thereby focusing the training efforts on the region proposal and classification components of the network.

#### Practical Applications
Fire Safety and Early Detection: The model can be integrated into surveillance systems within buildings, forests, or other infrastructures to provide early warnings of fire, significantly reducing response times and potential damages.

Environmental Monitoring: Useful in monitoring environmental pollution, specifically air quality and the presence of smoke or smog, which can inform health and safety decisions in urban planning.

#### Using the requirements.txt:
pip install -r requirements.txt
