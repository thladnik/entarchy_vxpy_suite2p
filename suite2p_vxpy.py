from cmath import phase

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

        for i in range(3):
            animal = Animal(self, _id=f'Animal_{i}', _parent=None)
            self.add_new_entity(animal)
            for j in range(7):
                recording = Recording(self, _id=f'Recording_{j}', _parent=animal)
                self.add_new_entity(recording)
                for p in range(300):
                    phase = Phase(self, _id=f'Phase_{p}', _parent=recording)
                    self.add_new_entity(phase)

                for r in range(random.randint(400, 800)):
                    roi = Roi(self, _id=f'Roi_{r}', _parent=recording)
                    self.add_new_entity(roi)

            self.commit()


if __name__ == '__main__':

    _backend = MySQLBackend(dbhost='172.25.240.200',
                            dbname=f'agarrenberg_test123',
                            dbuser='thladnik',
                            dbpassword=open('pw.txt', 'r').readline().strip(),)

    # try:
    #     my_entarchy = Suite2PVxPy('./my_test/path/')
    #     my_entarchy.delete()
    # except:
    #     pass
    #
    # my_entarchy = Suite2PVxPy.create('./my_test/path/', _backend)
    # my_entarchy.digest('')

    my_entarchy = Suite2PVxPy('./my_test/path/')


    recs = my_entarchy.get(Recording)
    rec = recs[0]

    print('Hello!!')
    print('')

