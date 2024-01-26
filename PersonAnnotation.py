from datetime import timedelta
from typing import Optional, Sequence, cast
from google.cloud import videointelligence_v1 as vi
import os
import pandas as pd

# set key credentials file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'airy-machine-411304-cc05ea533265.json'

from google.cloud import videointelligence_v1 as videointelligence


def detect_person(gcs_uri):
    """Detects people in a video."""

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
            "video_context": context,
        }
    )

    print("Processing video for person detection annotations.\n")
    result = operation.result(timeout=300)
    print("Finished processing.\n")

    return result



def analyzePerson(person):
    frames = []
    for track in person.tracks:
        # Convert timestamps to seconds
        for ts_obj in track.timestamped_objects:
            time_offset = ts_obj.time_offset
            timestamp = time_offset.seconds + time_offset.microseconds / 1e6
            frame= {'timestamp' : timestamp}
            for landmark in ts_obj.landmarks:
                frame[landmark.name + '_x'] = landmark.point.x
                # Subtract y value from 1 because positions are calculated
                # from the top left corner
                frame[landmark.name + '_y'] = 1 - landmark.point.y
            frames.append(frame)
    
    frames = sorted(frames, key=lambda x: x['timestamp'])
    return frames
    

######################################################
# Start Program
######################################################

# Read the list of files from list.csv
file_list_df = pd.read_csv('list.csv')

for index, row in file_list_df.iterrows():
    annotations_df = pd.DataFrame()
    video_uri = "gs://lunge-videos/" + row['file_name']
    print("File: "+ video_uri)
    operation = detect_person(video_uri)
    people_annotations = operation.annotation_results[0].person_detection_annotations

    for annotation in people_annotations:
        frames = analyzePerson(annotation)
        df = pd.DataFrame(frames)
        annotations_df = pd.concat([annotations_df, df], ignore_index=True)

    # Sort the DataFrame by timestamp and drop unnecessary columns
    annotations_df = annotations_df.sort_values('timestamp', ascending=True)
    # List of columns to potentially drop
    columns_to_drop = ['nose_x', 'nose_y', 'right_eye_x', 'right_eye_y', 'left_eye_x', 'left_eye_y', 'right_ear_x', 'right_ear_y', 'left_ear_x', 'left_ear_y']

    # Try to drop the columns, and if a KeyError occurs, ignore it
    for column in columns_to_drop:
        try:
            annotations_df = annotations_df.drop(column, axis=1)
        except KeyError:
            pass

    # Modify user_id and score as needed (assuming you have this information in list.csv)
    annotations_df['user_id'] = row['user_id']
    annotations_df['score'] = row['score']

    # Save the combined DataFrame to data.csv
    annotations_df.to_csv('data.csv', mode='a', header=False, index=False)


""" video_uri = "gs://lunge-videos/V1 - 1705215994744.mp4"
operation = detect_person(video_uri)
people_annotations = operation.annotation_results[0].person_detection_annotations

annotationsDf = pd.DataFrame()

for annotation in people_annotations:
    frames = analyzePerson(annotation)
    df = pd.DataFrame(frames)
    annotationsDf = pd.concat([annotationsDf, df], ignore_index= True)
    
annotationsDf = annotationsDf.sort_values('timestamp', ascending=True)
annotationsDf = annotationsDf.drop(['nose_x', 'nose_y', 'right_eye_x', 'right_eye_y', 'left_eye_x', 'left_eye_y', 'right_ear_x', 'right_ear_y', 'left_ear_x', 'left_ear_y', ], axis=1)
annotationsDf['user_id'] = 1
annotationsDf['score'] = 9.0
print(annotationsDf.head())
annotationsDf.to_csv('data.csv', mode='a', header=False, index=False) """
