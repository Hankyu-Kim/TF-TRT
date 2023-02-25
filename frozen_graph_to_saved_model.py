# Convert ProtoBuf model to saved_model, format for TF Serving
# https://cloud.google.com/ai-platform/prediction/docs/exporting-savedmodel-for-prediction
import shutil
import tensorflow.compat.v1 as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

# TF Serving supports run different versions of same model. So we put current model to '1' folder.
# export_dir = './1'
export_dir = './models/native_saved_model'
graph_pb = 'frozen_inference_graph.pb'

# Clear out folder
shutil.rmtree(export_dir, ignore_errors=True)

builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

with tf.io.gfile.GFile(graph_pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

sigs = {}

with tf.Session(graph=tf.Graph()) as sess:
    # Prepare input and outputs of model
    tf.import_graph_def(graph_def, name="")
    g = tf.get_default_graph()
    image_tensor = g.get_tensor_by_name("image_tensor:0")
    num_detections = g.get_tensor_by_name("num_detections:0")
    detection_scores = g.get_tensor_by_name("detection_scores:0")
    detection_boxes = g.get_tensor_by_name("detection_boxes:0")
    detection_classes = g.get_tensor_by_name("detection_classes:0")

    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.saved_model.signature_def_utils.predict_signature_def(
            {"input_image": image_tensor},
            {"num_detections": num_detections,
             "detection_scores": detection_scores,
             "detection_boxes": detection_boxes,
             "detection_classes": detection_classes})

    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map=sigs)

builder.save()
