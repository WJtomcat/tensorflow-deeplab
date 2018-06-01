
import tensorflow as tf
from deeplab.core import feature_extractor

slim = tf.contrib.slim

_LOGITS_SCOPE_NAME = 'logits'
_MERGED_LOGITS_SCOPE = 'merged_logits'
_IMAGE_POOLING_SCOPE = 'image_pooling'
_ASPP_SCOPE = 'aspp'
_CONCAT_PROJECTION_SCOPE = 'concat_projection'
_DECODER_SCOPE = 'decoder'

def get_merged_logits_scope():
    pass

def get_extra_layer_scopes(last_layers_contain_logits_only=False):
    pass

def predict_labels_multi_scale(images,
                               model_options,
                               eval_scales=(1.0,),
                               add_flipped_images=False):
    """
    Predicts segmentation labels.

    Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    eval_scales: The scales to resize images for evaluation.
    add_flipped_images: Add flipped images for evaluation or not.

    Returns:
    A dictionary with keys specifying the output_type (e.g., semantic
    prediction) and values storing Tensors representing predictions (argmax
    over channels). Each prediction has size [batch, height, width]
    """
    pass


def predict_labels(images, model_options, image_pyramid=None):
    """
    Predicts segmentation labels.

    Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    add_flipped_images: Add flipped images for evaluation or not.

    Returns:
    A dictionary with keys specifying the output_type (e.g., semantic
    prediction) and values storing Tensors representing predictions (argmax
    over channels). Each prediction has size [batch, height, width]
    """
    pass

def scale_dimension(dim, scale):
    pass

def multi_scale_logits(images,
                       model_options,
                       image_pyramid,
                       weight_decay=0.0001,
                       is_training=False,
                       fine_tune_batch_norm=False):
    """
    Gets the logits for multi_scale inputs.

    Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    image_pyramid: Input image scales for mutli-scale feature extraction.

    weight_decay: The weight decay for model variables.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

    Returns:
    outputs_to_scales_to_logits: A map of maps from output_type (e.g.,
    semantic prediction) to a dictionary of multi-scale logits names to logits.
    For each output_type, the dictionary has keys which correspond to the
    scales and values which correspond to the logits.
    For example, if `scales` equals [1.0, 1.5], then the keys would
    include 'merged_logits', 'logits_1.00' and 'logits_1.50'.
    """

    pass

def _extract_features(images,
                      model_options,
                      weight_decay=0.0001,
                      reuse=None,
                      is_training=False,
                      fine_tune_batch_norm=False):
    """
    Extracts features by the particular model_variant.

    """
    return

def _get_logits(images,
                model_options,
                weight_decay=0.0001,
                reuse=None,
                is_training=False,
                fine_tune_batch_norm=False):
    pass

def refine_by_decoder(features,
                      end_points,
                      decoder_height,
                      decoder_width,
                      decoder_use_separable_conv=False,
                      model_variant=None,
                      weight_decay=0.0001,
                      reuse=None,
                      is_training=False,
                      fine_tune_batch_norm=False):
    pass

def _get_branch_logits(features,
                       num_classes,
                       atrous_rates=None,
                       aspp_with_batch_norm=False,
                       kernel_size=1,
                       weight_decay=0.0001,
                       reuse=None,
                       scope_suffix=''):
    pass

def _split_separable_conv2d(inputs,
                            filters,
                            rate=1,
                            weight_decay=0.00004,
                            depthwise_weights_initializer_stddev=0.33,
                            pointwise_weights_initializer_stddev=0.06,
                            scope=None):
    pass
