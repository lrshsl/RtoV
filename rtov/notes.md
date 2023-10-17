
# 21. 09.

After adding the accuracy to the plots, I have realized that the loss is decreasing rapidly, but the accuracy stayed approximately the same.

The results were sometimes pretty accurate, but after rerunning the program with the same parameters the model could learn something completely different, where it e. g. predicted almost all shapes as lines or rectangles

What I changed:
    1. Tried to change the metrics to accuracy. Was more complicated than in tensorflow, so I plan to do it later.
    2. Added a dropout layer -> didn't observe an effect on accuracy
    3. Added weight decay, so that single neurons don't get to powerful
    4. Batch normalization layers after the ReLU layers


After adding batch normalization, the loss changed drastically:

This was a typical progression of the loss before adding batch normalization layers:
```
Epoch 1/7, Loss: 27.713708877563477
Epoch 2/7, Loss: 1.5287768840789795
Epoch 3/7, Loss: 1.1809443235397339
Epoch 4/7, Loss: 1.16764497756958
Epoch 5/7, Loss: 1.11421537399292
Epoch 6/7, Loss: 1.110487461090088
Epoch 7/7, Loss: 1.1122630834579468
```

As one can see, the loss had rapidly drastically decreased, but than plateaued and eventually stagnated completely. This wasn't perfect, but at least continuously improving.

This is what happened with batch normalization:
```
Epoch 1/7, Loss: 1.0988205671310425
Epoch 2/7, Loss: 1144.3743896484375
Epoch 3/7, Loss: 99.31055450439453
Epoch 4/7, Loss: 55.350547790527344
Epoch 5/7, Loss: 69.87451934814453
Epoch 6/7, Loss: 74.32080841064453
Epoch 7/7, Loss: 57.944793701171875
```
It had reached a great loss value after the first epoch, but skyrocketed in the second one, to decrease again in the epochs 3 through 7.
One explanation would be, that the model had learned the data from epoch 1 by heart, without really understanding the features it ought to look out for, and thus performed horribly on a new set of data. The problem is, that the data should always be completely random, and no two images the same in the program that I wrote, so I still don't really have an explanation for that.


Decreasing the learning rate had helped much:
```
# Learning rate = 0.005
Epoch 1/7, Loss: 1.1150662899017334
Epoch 2/7, Loss: 91357.3203125
Epoch 3/7, Loss: 188652.84375
Epoch 4/7, Loss: 20371.61328125
Epoch 5/7, Loss: 5901.9912109375
Epoch 6/7, Loss: 4764.1708984375
Epoch 7/7, Loss: 2742.408203125

```


```
# Learning rate = 0.00005
Epoch 1/7, Loss: 1.0995022058486938
Epoch 2/7, Loss: 59.42584991455078
Epoch 3/7, Loss: 27.856739044189453
Epoch 4/7, Loss: 13.690399169921875
Epoch 5/7, Loss: 10.326250076293945
Epoch 6/7, Loss: 7.497406959533691
Epoch 7/7, Loss: 8.900046348571777
```

And this happened when I reduced the number of images per epoch from 80 to 20:

```
Epoch 1/7, Loss: 1.0998528003692627
Epoch 2/7, Loss: 1.1976195573806763
Epoch 3/7, Loss: 1.2382482290267944
Epoch 4/7, Loss: 1.1982051134109497
Epoch 5/7, Loss: 1.210884928703308
Epoch 6/7, Loss: 1.133998155593872
Epoch 7/7, Loss: 1.171489953994751
```







