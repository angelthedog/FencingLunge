import os
import pandas as pd
import json
from google.cloud import videointelligence_v1 as videointelligence
from google.protobuf.json_format import MessageToDict

# set key credentials file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'airy-machine-411304-cc05ea533265.json'


######################################################
# Start Program
######################################################
# Your GCS video path and output json path
gcs_uri = "gs://lunge-videos/1705346165857.mp4"
output_uri = "gs://lunge-videos/data.json"

client = videointelligence.VideoIntelligenceServiceClient()

# Configure the request
config = videointelligence.types.PersonDetectionConfig(
    include_bounding_boxes=True,
    include_attributes=False,
    include_pose_landmarks=True,
)
context = videointelligence.types.VideoContext(person_detection_config=config)

# Start the asynchronous request
operation = client.annotate_video(
    request={
        "features": [videointelligence.Feature.PERSON_DETECTION],
        "input_uri": gcs_uri,
        "output_uri": output_uri,
        "video_context": context,
    }
)

result = operation.result(timeout=300)