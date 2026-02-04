import os
import pathlib

import numpy as np
import tifffile
import yaml

import entarchy

__all__ = ['Animal', 'Recording', 'Layer', 'Roi', 'Phase', 'Suite2PVxPy']


class AnimalCollection(entarchy.Collection):
    pass


class RecordingCollection(entarchy.Collection):
    pass


class RoiCollection(entarchy.Collection):
    pass


class PhaseCollection(entarchy.Collection):
    pass


class Animal(entarchy.Entity):
    collection_type = AnimalCollection


class Recording(entarchy.Entity):
    collection_type = RecordingCollection
    # # Child entity types may be added by name meta // Does not work yet
    # _child_entity_types = ['Roi', 'Phase']
    pass


class Layer(entarchy.Entity):
    pass


class Roi(entarchy.Entity):
    collection_type = RoiCollection
    pass


class Phase(entarchy.Entity):
    collection_type = PhaseCollection
    pass


# Child entity types may be added by method calls
Animal.add_child_entity_type(Recording)
Recording.add_child_entity_type(Layer)
Layer.add_child_entity_type(Roi)
Recording.add_child_entity_type(Phase)


class Suite2PVxPy(entarchy.Entarchy):

    implementation_version = '0.2'

    _hierarchy_root = Animal

    @entarchy.digest_method
    def add_animal(self, path: str):

        with self:

            path = pathlib.Path(path).as_posix()

            print(f'> Add animal from path {path}')

            # Create animal
            path_parts = path.split('/')
            animal_id = path_parts[-1]

            # Create new animal entity
            print(f'>> Create new entity for animal {animal_id}')
            animal = Animal(self, _id=animal_id, _parent=self.root)
            self.add_new_entity(animal)

            # Search for zstacks
            zstack_names = []
            for fn in os.listdir(path):
                _p = os.path.join(path, fn)
                if os.path.isdir(_p):
                    continue
                if 'zstack' in fn:
                    if fn.lower().endswith(('.tif', '.tiff')):
                        zstack_names.append(fn)

            # Add first stack that was detected
            if len(zstack_names) > 0:
                if len(zstack_names) > 1:
                    print(f'WARNING: multiple zstacks detected, using {zstack_names[0]}')

                # Load zstack
                zstack_data = tifffile.imread(os.path.join(path, zstack_names[0]))

                print(f'>> Add zstack {zstack_names[0]} of shape {zstack_data.shape}')
                animal['zstack_fn'] = zstack_names[0]
                animal['zstack'] = zstack_data

        # Add metadata
        add_metadata(animal, path)

        # Search for valid registration path in animal folder
        valid_reg_path = None
        if 'ants_registration' in os.listdir(path):
            for mov_folder in os.listdir(os.path.join(path, 'ants_registration')):
                for ref_folder in os.listdir(os.path.join(path, 'ants_registration', mov_folder)):
                    reg_path = os.path.join(path, 'ants_registration', mov_folder, ref_folder)

                    # If there is a transform file, we'll take it
                    if 'Composite.h5' in os.listdir(reg_path):
                        valid_reg_path = reg_path
                        break

        # Write registration metadata to animal entity
        if valid_reg_path is not None:
            print(f'Loading ANTs registration metadata at {valid_reg_path}')
            ants_metadata = yaml.safe_load(open(os.path.join(valid_reg_path, 'metadata.yaml'), 'r'))
            animal.update({f'ants/{n}': v for n, v in ants_metadata.items()})

        return animal

    def add_recording(self, animal: Animal, path: str, layer_idx: int) -> Recording:

        with self:

            # Create debug folder
            debug_folder_path = os.path.join(path, 'debug')
            if not os.path.exists(debug_folder_path):
                os.mkdir(debug_folder_path)

            # Get recording
            rec_id = f"{pathlib.Path(path).as_posix().split('/')[-1]}_layer{layer_idx}"

            recording = Recording(self, _id=rec_id, _parent=animal)
            self.add_new_entity(recording)

            # Add metadata
            add_metadata(recording, path)

            return recording

    @entarchy.digest_method
    def test_digest(self) -> None:

        import random

        # Use in context to control commits
        #  and speed up adding multiple entities and their attributes
        with self:

            # for i in tqdm.tqdm(range(3), desc='Animals', position=0):
            for i in range(3):
                animal = Animal(self, _id=f'Animal_{i}', _parent=self.root)
                print(f'Add {animal}')
                for k, v in {'age': random.randint(24, 96),
                             'weight': random.randint(20, 100) / 10,
                             'strain': random.choice(['jf1', 'mpn400', 'jf7']),
                             'zstack': np.random.randint(0, 255, size=(50, 512, 512), dtype=np.uint8)}.items():
                    animal[k] = v

                self.add_new_entity(animal)

                # for j in tqdm.tqdm(range(7), desc='Recordings', position=1, leave=False):
                for j in range(random.randint(2, 4)):
                    recording = Recording(self, _id=f'Recording_{j}', _parent=animal)
                    print(f'> Add {recording}')
                    self.add_new_entity(recording)

                    for jj in range(5):
                        recording[f'rec_param_int_{jj}'] = random.randint(0, 1000)
                    for jj in range(5):
                        recording[f'rec_param_float_{jj}'] = random.randint(0, 10000) / 10
                    for jj in range(5):
                        recording[f'rec_param_string_{jj}'] = random.choice(['foo', 'bar', 'baz', 'lorem', 'ipsum', 'dolor'])
                    for jj in range(5):
                        recording[f'rec_param_array_{jj}'] = np.random.rand(random.randint(10, 100))
                    for jj in range(5):
                        recording[f'rec_param_list_{jj}'] = [random.randint(0, 100) for _ in range(random.randint(10, 100))]
                    for jj in range(2):
                        recording[f'rec_param_largelist_{jj}'] = ['abc'] * 20_000_000
                    for jj in range(2):
                        recording[f'rec_param_largearray_{jj}'] = np.random.rand(*np.random.randint(50, 150, size=(3,)))

                    # for p in tqdm.tqdm(range(300), desc='Phases', position=2, leave=False):
                    p_num = 300
                    print(f'>> Add {p_num} Phases')
                    for p in range(p_num):
                        phase = Phase(self, _id=f'Phase_{p}', _parent=recording)
                        self.add_new_entity(phase)

                        for jj in range(5):
                            phase[f'phase_param_int_{jj}'] = random.randint(0, 1000)
                        for jj in range(5):
                            phase[f'phase_param_float_{jj}'] = random.randint(0, 10000) / 10
                        for jj in range(5):
                            phase[f'phase_param_string_{jj}'] = random.choice(['foo', 'bar', 'baz', 'lorem', 'ipsum', 'dolor'])
                        for jj in range(5):
                            phase[f'phase_param_array_{jj}'] = np.random.rand(random.randint(10, 100))
                        for jj in range(5):
                            phase[f'phase_param_list_{jj}'] = [random.randint(0, 100) for _ in range(random.randint(10, 100))]

                    for li in range(5):

                        print('>> Add Layer', li)
                        layer = Layer(self, _id=f'Layer_{li}', _parent=recording)
                        self.add_new_entity(layer)

                        # for r in tqdm.tqdm(range(random.randint(400, 800)), desc='Rois', position=2, leave=False):
                        r_num = random.randint(200, 400)
                        print(f'>>> Add {r_num} Rois')
                        for r in range(r_num):
                            roi = Roi(self, _id=f'Roi_{r}', _parent=layer)
                            self.add_new_entity(roi)

                            for jj in range(5):
                                roi[f'roi_param_int_{jj}'] = random.randint(0, 1000)
                            for jj in range(5):
                                roi[f'roi_param_float_{jj}'] = random.randint(0, 10000) / 10
                            for jj in range(5):
                                roi[f'roi_param_string_{jj}'] = random.choice(['foo', 'bar', 'baz', 'lorem', 'ipsum', 'dolor'])
                            for jj in range(5):
                                roi[f'roi_param_array_{jj}'] = np.random.rand(random.randint(10, 100))
                            for jj in range(5):
                                roi[f'roi_param_list_{jj}'] = [random.randint(0, 100) for _ in range(random.randint(10, 100))]
                            for jj in range(2):
                                roi[f'roi_param_largelist_{jj}'] = ['abc'] * 2_000_000
                            for jj in range(2):
                                roi[f'roi_param_largearray_{jj}'] = np.random.rand(*np.random.randint(1, 70, size=(3,)))

                        # Commit after each layer
                        self.commit()


def add_metadata(entity: entarchy.Entity, folder_path: str):
    """Function searches for and returns metadata on a given folder path

    Function scans the `folder_path` for metadata yaml files (ending in `meta.yaml`)
    and returns a dictionary containing their contents
    """

    meta_files = [f for f in os.listdir(folder_path) if f.endswith('metadata.yaml')]

    print(f'Found {len(meta_files)} metadata files in {folder_path}.')

    metadata = {}
    for f in meta_files:
        with open(os.path.join(folder_path, f), 'r') as stream:
            try:
                metadata.update(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)

    # Add metadata
    unravel_dict(metadata, entity, 'metadata')


def unravel_dict(dict_data: dict, entity: entarchy.Entity, path: str):
    for key, item in dict_data.items():
        if isinstance(item, dict):
            unravel_dict(item, entity, f'{path}/{key}')
            continue
        entity[f'{path}/{key}'] = item


if __name__ == '__main__':
    pass