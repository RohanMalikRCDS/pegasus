import pandas as pd
import tensorflow as tf
import os

# NOTE: run this AFTER downloading the main pretrained model and fine-tuned reddit_tifu model and other setup
# this script creates the TFRecord that stores the phrase we want to be annotating
# as well as an auxiliary file that the module needs to recognize the fine tuned checkpoints

inp = """so normally i can deal with taking care of my mom who has dementia, she does some annoying things but for the most part it's manageable. today it was different. i was going to sleep at night, with her in the other room trying to do the same, and just as i was falling asleep i hear a sound from the room.
my mother has wet herself in the bed and does not even realize that she is sleeping in it. when i tell her to get up and go to the bathroom so i can change her diaper, she does not even care and continuously asks me to repeat myself over and over.
by this point i am extremely annoyed, borderline angry, and i try to prevent myself from lashing out at her. this kind of thing is super frustrating and i love my mom but after some point i can't take it anymore.
"""

input_dict = dict(
    inputs=[inp],
    targets=["mom wet the bed and didn't care, normally fine with caregiving duties but this pushed it past the line."]
    )

save_path = "pegasus/data/testdata/test_pattern_1.tfrecord"
data = pd.DataFrame(input_dict)
with tf.io.TFRecordWriter(save_path) as writer:
    for row in data.values:
        inputs, targets = row[:-1], row[-1]
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "inputs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[inputs[0].encode('utf-8')])),
                    "targets": tf.train.Feature(bytes_list=tf.train.BytesList(value=[targets.encode('utf-8')])),
                }
            )
        )
        writer.write(example.SerializeToString())

# other preparation
with open("../ckpt/pegasus_ckpt/reddit_tifu/checkpoint", mode='w') as ckptFile:
    ckptFile.write('model_checkpoint_path: "model.ckpt-8000"\nall_model_checkpoint_paths: "model.ckpt-8000"')
