"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import tensorflow as tf
import numpy as np
import scipy.misc

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Logger(object):
    def __init__(self, log_dir, suffix=None):
        self.writer = tf.summary.create_file_writer(log_dir, filename_suffix=suffix)

    def scalar_summary(self, tag, value, step):
        value = value.cpu()
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        with self.writer.as_default():
          tf.summary.scalar(tag, data=value, step=step)
        # self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):

        # images = images.cpu()
        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            # scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            # img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
            #                            height=img.shape[0],
            #                            width=img.shape[1])
            # Create a Summary value
            # img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))
            with self.writer.as_default():
                tf.summary.image('%s/%d' % (tag, i), [img], step=step)

        # Create and write Summary
        # summary = tf.Summary(value=img_summaries)
        # self.writer.add_summary(summary, step)
        # tf.summary.image()
        self.writer.flush()

    def video_summary(self, tag, videos, step):

        sh = list(videos.shape)
        sh[-1] = 1
        # videos = videos.cpu()

        separator = np.zeros(sh, dtype=videos.dtype)
        videos = np.concatenate([videos, separator], axis=-1)

        img_summaries = []
        for i, vid in enumerate(videos):
            # Concat a video
            try:
                s = StringIO()
            except:
                s = BytesIO()

            v = vid.transpose(1, 2, 3, 0)
            v = [np.squeeze(f) for f in np.split(v, v.shape[0], axis=0)]
            img = np.concatenate(v, axis=1)[:, :-1, :]

            # scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            # img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
            #                            height=img.shape[0],
            #                            width=img.shape[1])
            # Create a Summary value
            # img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))
            with self.writer.as_default():
                tf.summary.image('%s/%d' % (tag, i), [img], step=step)

        # Create and write Summary
        # summary = tf.Summary(value=img_summaries)
        # self.writer.add_summary(summary, step)
        self.writer.flush()
