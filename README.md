# GAN-behavior

GAN行为信息生成

## 生成器

输入comment.train.txt，生成器需要根据comment.train.txt为每行数据生成类似于behavior.train.txt的行为信息。

## 判别器

判别器的输入为生成器生成的行为信息+behavior.train.txt（样本1、生成样本）、comment.train.txt+behavior.train.txt（样本2、正样本）、neg_train.txt+behavior.train.txt（样本3、负样本），判别器需要判别出哪个行为信息是生成器生成的、哪个是真实的，再以此指导生成器训练，具体来说是判断样本1为假、样本2为真、样本3为假。

## 功能

我的基于文本的GAN模型包括generator.py、discriminator.py、train.py、coldstart.py。generator.py的输入comment.train.txt，生成器需要根据comment.train.txt为每行数据生成类似于behavior.train.txt的行为信息,判别器需要判别出哪个行为信息是生成器生成的、哪个是真实的，再以此指导生成器训练。train.py最后需要保存最佳模型，打印模型训练报告。coldstart.py使用最佳的模型对给定的comment信息生成行为信息并保存到一个txt或者csv文件中。
