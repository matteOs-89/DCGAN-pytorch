
# Generative Adversarial Network 
### GANs WAS CREATED BY Ian Goodfella (2014)



Generative Adversarial Networks (GAN) are generative frameworks that produce new data with close similarities to its training data. This is achieved by training the model to learn the training data's distribution, storing it in its latent space and then creating new data from that distribution. 

The GANs uses two models to achieve this, these are:

•	The Generator – whose job is to distribute fake data similar to the real data in efforts to "trick" the Discriminator (see below); and
•	The Discriminator – which looks at the produced data and predicts if the data provided by the Generator is real or fake.

Repeating this process numerous of times allows the Discriminator to get better at predicting real from fakes, which also pushes the Generator to get better in producing better fakes to the point where the Discriminator is no longer as certain.

There are various types of GAN techniques such as WGAN, CircleGAN, however for our tutorial today we will be focusing on Deep Convolutional GAN (DCGAN).

DCGAN is a method where we use convolutional networks to create images.


## Specification

<img width="491" alt="Screenshot 2022-12-31 at 12 06 46" src="https://user-images.githubusercontent.com/111536571/210149868-150df46e-1cd9-4bce-9dc3-816edff1b32f.png">




For this tutorial we build a 512 sized Generator and Discriminator as image below shows.
DIAGRAM FROM - https://arxiv.org/pdf/1810.03764.pdf                                    


## EVALUATION


#### SAMPLES OF DATA USED FOR TRAINING

<img width="456" alt="Screenshot 2022-12-31 at 12 48 59" src="https://user-images.githubusercontent.com/111536571/210149495-ee051624-feca-4c75-86da-067285c1a55a.png">


The dataset used for this tutorial was downloaded from https://www.kaggle.com/code/husnakhan/animal-faces/data.



## GENERATED IMAGES DURING TRAINING

#### AFTER 2 EPOCHS                                                                     


<img width="560" alt="Screenshot 2022-12-31 at 09 59 51" src="https://user-images.githubusercontent.com/111536571/210149608-04d64202-cf2b-4c88-8652-bcb9f300a9e9.png">

#### EPOCH 11
<img width="556" alt="Screenshot 2022-12-31 at 12 49 58" src="https://user-images.githubusercontent.com/111536571/210149631-feef30fd-46a8-4058-96cf-a471ec980a79.png">

  
#### EPOCH 20

<img width="556" alt="Screenshot 2022-12-31 at 12 50 21" src="https://user-images.githubusercontent.com/111536571/210149650-c91da981-6ea8-446d-813a-ddc02253bd1a.png">


#### EPOCH 26

<img width="556" alt="Screenshot 2022-12-31 at 12 50 39" src="https://user-images.githubusercontent.com/111536571/210149665-c144684d-c6f7-4046-a3f6-18a05b271b71.png">


#### EPOCH 29


<img width="556" alt="Screenshot 2022-12-31 at 12 50 56" src="https://user-images.githubusercontent.com/111536571/210149682-b795c1a5-f92a-448a-9253-9131c8d2db48.png">


NOTE – Training was stopped after 29 epoch which took almost 2 hours to train.
During training we observed that Generator continued to improve during training epochs.
Perhaps if trained for longer the generated images would further improve.

Another way we could improve performance is by completing more training on the Discriminator over the Generator, which will encourage the Generator to create better images to trick the Discriminator.

Author - Eseosa Jesuorobo
