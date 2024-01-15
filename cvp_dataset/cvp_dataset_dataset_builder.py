from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from PIL import Image

camview_to_id = {
    '24259877_right': 0,
    '20521388_left': 1,
}

SIZE=(256, 256)
def resize_image(image):
    img = Image.fromarray(image).resize(SIZE, Image.Resampling.LANCZOS)
    return np.array(img)
class CVPDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image_0': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'image_1': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot state, consists of [7x robot joint angles, '
                                '2x gripper position, 1x door opening angle].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'episode_id': tfds.features.Scalar(
                        dtype=np.int32,
                        doc='ID of episode in file_path.'
                    ),
                    'has_language': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if language exists in observation, otherwise empty string.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        base_path = '/home/rail-franka/Desktop/cvp/data/'
        return {
            'train': self._generate_examples(path=base_path + 'npy/train/*.npy'),
            'val': self._generate_examples(path=base_path + 'npy/test/*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        _embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            data = np.load(episode_path, allow_pickle=True)  # this is a list of dicts in our case
            print(f"Loaded {episode_path} with {len(data)} steps.")
            for k, example in enumerate(data):
                # assemble episode --> here we're assuming demos so we set reward to 1 at the end
                # example is a list of dicts, each dict is one step
                episode = []
                if 'language' in example[0]:
                    instruction = example[0]['language']  # get the first instruction
                    language_embedding = _embed([instruction])[0].numpy()
                else:
                    instruction = ''
                    language_embedding = np.zeros(512, dtype=np.float32)

                for i in range(example.shape[0]):
                    observation = {
                        'state': example[i]['observation']['proprio'].astype(np.float32),
                        # 'subtask_id': example[i]['observation']['subtask_id'].astype(np.int32),
                    }
                    # gather all the proprioceptive states
                    # for robot_state in ['joint_position', 'joint_velocity']:
                    #     observation[robot_state] = example[i]['action'][robot_state].astype(np.float32)

                    for cam_name in example[i]['observation']['image']:
                        id = camview_to_id[cam_name]
                        observation[f'image_{id}'] = resize_image(example[i]['observation']['image'][cam_name].astype(np.uint8))

                    action = np.concatenate(
                        (example[i]['action']['cartesian_position'], example[i]['action']['gripper_position'][None]),
                        axis=0)

                    episode.append({
                        'observation': observation,
                        'action': action.astype(np.float32),
                        'discount': 1.0,
                        'reward': float(i == (example.shape[0] - 1)),
                        'is_first': i == 0,
                        'is_last': i == (example.shape[0] - 1),
                        'is_terminal': i == (example.shape[0] - 1),
                        'language_instruction': instruction,
                        'language_embedding': language_embedding,
                    })

                # create output data sample
                sample = {
                    'steps': episode,
                    'episode_metadata': {
                        'file_path': episode_path,
                        'episode_id': k,
                    }
                }

                # mark dummy values
                sample['episode_metadata']['has_language'] = bool(instruction)

                # if you want to skip an example for whatever reason, simply return None
                yield episode_path + str(k), sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            for id, sample in _parse_example(sample):
                yield id, sample