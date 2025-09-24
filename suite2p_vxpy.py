import sys

import numpy as np
import tqdm

from entarchy import Entarchy, Entity, Collection

__all__ = ['Animal', 'Recording', 'Roi', 'Phase', 'Suite2PVxPy']

from entarchy.backend import MySQLBackend


class Animal(Entity):
    pass


class Recording(Entity):
    # # Child entity types may be added by name meta // Does not work yet
    # _child_entity_types = ['Roi', 'Phase']
    pass


class Roi(Entity):
    pass


class Phase(Entity):
    pass


# Child entity types may be added by method calls
Animal.add_child_entity_type(Recording)
Recording.add_child_entity_type(Roi)
Recording.add_child_entity_type(Phase)


class Suite2PVxPy(Entarchy):

    _implementation_version = '0.2'

    _hierarchy_root = Animal

    def digest(self, raw_data_path: str) -> None:

        import random

        # Use in context to control commits
        #  and speed up adding multiple entities and their attributes
        with self:
            R = 0

            # for i in tqdm.tqdm(range(3), desc='Animals', position=0):
            for i in range(3):
                animal = Animal(self, _id=f'Animal_{i}', _parent=None)
                print(f'Add {animal}')
                for k, v in {'age': random.randint(24, 96),
                             'weight': random.randint(20, 100) / 10,
                             'strain': random.choice(['jf1', 'mpn400', 'jf7'])}.items():
                    animal[k] = v
                self.add_new_entity(animal)

                # for j in tqdm.tqdm(range(7), desc='Recordings', position=1, leave=False):
                for j in range(random.randint(3, 8)):
                    recording = Recording(self, _id=f'Recording_{j}', _parent=animal)
                    print(f'> Add {recording}')
                    self.add_new_entity(recording)

                    for jj in range(5):
                        recording[f'param_int_{jj}'] = random.randint(0, 1000)
                    for jj in range(5):
                        recording[f'param_float_{jj}'] = random.randint(0, 10000) / 10
                    for jj in range(5):
                        recording[f'param_string_{jj}'] = random.choice(['foo', 'bar', 'baz', 'lorem', 'ipsum', 'dolor'])
                    for jj in range(5):
                        recording[f'param_array_{jj}'] = np.random.rand(random.randint(10, 100))
                    for jj in range(5):
                        recording[f'param_list_{jj}'] = [random.randint(0, 100) for _ in range(random.randint(10, 100))]

                    # for p in tqdm.tqdm(range(300), desc='Phases', position=2, leave=False):
                    p_num = 300
                    print(f'>> Add {p_num} Phases')
                    for p in range(p_num):
                        phase = Phase(self, _id=f'Phase_{p}', _parent=recording)
                        self.add_new_entity(phase)

                    # for r in tqdm.tqdm(range(random.randint(400, 800)), desc='Rois', position=2, leave=False):
                    r_num = random.randint(400, 800)
                    print(f'>> Add {r_num} Rois')
                    for r in range(r_num):
                        roi = Roi(self, _id=f'Roi_{r}', _parent=recording)
                        self.add_new_entity(roi)

                        for jj in range(5):
                            roi[f'param_int_{jj}'] = random.randint(1, 999)
                        for jj in range(5):
                            roi[f'param_float_{jj}'] = random.randint(0, 10000) / 10
                        for jj in range(5):
                            roi[f'param_string_{jj}'] = random.choice(['foo', 'bar', 'baz', 'lorem', 'ipsum', 'dolor'])
                        for jj in range(5):
                            roi[f'param_array_{jj}'] = np.random.rand(random.randint(10, 100))
                        for jj in range(5):
                            roi[f'param_list_{jj}'] = [random.randint(0, 100) for _ in range(random.randint(10, 100))]

                    # Commit after each recording
                    self.commit()



if __name__ == '__main__':

    recreate = True
    if 'recreate' in sys.argv:
        recreate = True

    # Recreate entarchy
    if recreate:
        try:
            ente = Suite2PVxPy('./dev_test/path/')
            ente.delete()
        except:
            pass

        # Get connection data
        with open('secret.txt', 'r') as f:
            mysqldata = [f.readline().strip() for _ in range(4)]

        _backend = MySQLBackend(dbhost=mysqldata[0],
                                dbname=f'{mysqldata[1]}_test123',
                                dbuser=mysqldata[2],
                                dbpassword=mysqldata[3])

        ente = Suite2PVxPy.create('./dev_test/path/', _backend)
        ente.digest('')
