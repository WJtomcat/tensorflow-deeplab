import collections
import six
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_enum('image_format', 'png', ['jpg', 'jpeg', 'png'],
                         'Image format')

tf.app.flags.DEFINE_enum('label_format', 'png', ['png'],
                         'Segmentation label format')


class ImageReader(object):

    def __init__(self, image_format='jpeg', channels=3):

        with tf.Graph().as_default():
            self._decode_data = tf.placeholder(dtype=tf.string)
            self._image_format = image_format
            self._session = tf.Session()
            if self._image_format in ('jpeg', 'jpg'):
                self._decode = tf.image.decode_jpeg(self._decode_data,
                                                    channels=channels)
            elif self._image_format == 'png':
                self._decode = tf.image.decode_png(self._decode_data,
                                                   channels=channels)

    def read_image_dims(self, image_data):
        image = self.decode_image(image_data)
        return image.shape[:2]

    def decode_image(self, image_data):
        image = self._session.run(self._decode,
                                  feed_dict={self._decode_data: image_data})
        if len(image.shape) != 3 or image.shape[2] not in (1, 3):
            raise ValueError('The image channels not supported.')

        return image
